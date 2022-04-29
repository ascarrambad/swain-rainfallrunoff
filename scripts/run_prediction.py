"""
Run predictors on either the MatrLA or PemsBay traffic datasets.

The underlying graph is generated by using a thresholded Gaussian kernel on the geographic distance of the sensors.
"""

import os
import copy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler, MinMaxScaler
from tsl.data.utils import WINDOW, HORIZON
from tsl.nn.utils import casting
from tsl.predictors import Predictor
from tsl.utils import TslExperiment, ArgParser, parser_utils, numpy_metrics
from tsl.utils.parser_utils import str_to_bool
from tsl.utils.neptune_utils import TslNeptuneLogger

import tsl

from tsl.nn.metrics.metrics import MaskedMSE, MaskedMAE, MaskedMAPE

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np

import pathlib
import datetime
import yaml

from models.dcrnn_model import SWAIN_DCRNNModel
from models.gat_model import SWAIN_GATModel
from models.grawave_model import SWAIN_GraphWaveNetModel

from dataset.lamah import LamaH
from metrics import MaskedNSE, np_masked_nse

import tsl_config

def get_model_class(model_str):
    if model_str == 'dcrnn':
        model = SWAIN_DCRNNModel
    elif model_str == 'gat':
        model = SWAIN_GATModel
    elif model_str == 'gatedgn':
        raise NotImplementedError(f'Model "{model_str}" not available.')
    elif model_str == 'gwnet':
        model = SWAIN_GraphWaveNetModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_name):
    if dataset_name == 'lamah':
        dataset = LamaH()
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset


def add_parser_arguments(parent):
    # Argument parser
    parser = ArgParser(strategy='random_search', parents=[parent], add_help=False)

    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument("--model-name", type=str, default='dcrnn')
    parser.add_argument("--dataset-name", type=str, default='lamah')
    parser.add_argument("--config", type=str, default='dcrnn.yaml')
    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2-reg', type=float, default=0.),
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--val-start', type=str, default='2000-10-1')

    # logging
    parser.add_argument('--save-preds', action='store_true', default=False)
    parser.add_argument('--neptune-logger', action='store_true', default=False)
    parser.add_argument('--project-name', type=str, default="swain")
    parser.add_argument('--tags', type=str, default=tuple())

    known_args, _ = parser.parse_known_args()
    model_cls = get_model_class(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataset.add_argparse_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    return parser


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    tsl.logger.info(f'SEED: {args.seed}')

    model_cls = get_model_class(args.model_name)
    dataset = get_dataset(args.dataset_name)

    tsl.logger.info(args)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(tsl_config.config['logs_dir'],
                          args.dataset_name,
                          args.model_name,
                          exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'tsl_config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    edge_index = dataset.get_connectivity(method='binary',
                                          layout='edge_index',
                                          include_self=False)

    edge_attr = torch.from_numpy(dataset.stream).float()
    edge_scaler = MinMaxScaler(axis=0)
    edge_attr = edge_scaler.fit_transform(edge_attr)

    node_attr = dataset.catchment
    node_scaler = MinMaxScaler(axis=0)
    node_attr = node_scaler.fit_transform(node_attr)

    #############

    torch_dataset = SpatioTemporalDataset(*dataset.numpy(return_idx=True),
                                          exogenous=dataset.exogenous,
                                          mask=dataset.mask,
                                          connectivity=edge_index,
                                          horizon=args.horizon,
                                          window=args.window,
                                          stride=args.stride)
    if args.use_node_attribs != 'none':
        torch_dataset.add_attribute(name='node_attr',
                                    value=node_attr,
                                    node_level=True,
                                    add_to_batch=True)

    torch_dataset.edge_attr = edge_attr

    torch_dataset.set_input_map(x=(['data'], WINDOW),
                                u_w=(['u'], WINDOW),
                                u_h=(['u'], HORIZON))

    dm_conf = parser_utils.filter_args(args, SpatioTemporalDataModule, return_dict=True)
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers={'data': MinMaxScaler(axis=(0, 1)),
                 'u': MinMaxScaler(axis=(0, 1))},
        splitter=dataset.get_splitter(method='at_datetime',
                                      val_start=args.val_start),
        **dm_conf
    )


    ########################################
    # predictor                            #
    ########################################

    additional_model_hparams = dict(n_nodes=torch_dataset.n_nodes,
                                    input_size=torch_dataset.n_channels,
                                    output_size=torch_dataset.n_channels,
                                    horizon=torch_dataset.horizon,
                                    exog_size=torch_dataset.input_map.u_w.n_channels)

    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams},
                                            target_cls=model_cls,
                                            return_dict=True)

    loss_fn = MaskedMSE(compute_on_step=True)

    metrics = {'nse': MaskedNSE(compute_on_step=False),
               'mae': MaskedMAE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mape': MaskedMAPE(compute_on_step=False)}

    # setup predictor
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    predictor = Predictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': args.lr,
                      'weight_decay': args.l2_reg},
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs={
            'eta_min': 0.0001,
            'T_max': args.epochs
        },
        scale_target=True
    )

    ########################################
    # logging options                      #
    ########################################

    # log number of parameters
    args.trainable_parameters = predictor.trainable_parameters

    # add tags
    tags = list(args.tags) + [args.model_name, args.dataset_name]

    if args.neptune_logger:
        logger = TslNeptuneLogger(api_key=tsl_config.config['neptune_token'],
                                  project_name=f"{tsl_config.config['neptune_username']}/{args.project_name}",
                                  experiment_name=exp_name,
                                  tags=tags,
                                  params=vars(args),
                                  offline_mode=False,
                                  upload_stdout=False)
    else:
        logger = TensorBoardLogger(
            save_dir=logdir,
            name=f'{exp_name}_{"_".join(tags)}',

        )
    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor='val_mse',
        patience=args.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=logdir,
        save_top_k=1,
        monitor='val_mse',
        mode='min',
    )

    trainer = pl.Trainer(max_epochs=args.epochs,
                         default_root_dir=logdir,
                         logger=logger,
                         gpus=1 if torch.cuda.is_available() else None,
                         gradient_clip_val=args.grad_clip_val,
                         callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(predictor, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    predictor.load_state_dict(
       torch.load(checkpoint_callback.best_model_path, lambda storage, loc: storage)['state_dict'])

    predictor.freeze()
    trainer.test(predictor, datamodule=dm)

    output = trainer.predict(predictor, dataloaders=dm.test_dataloader())
    output = casting.numpy(output)
    y_hat, y_true, mask = output['y_hat'], \
                          output['y'], \
                          output['mask']
    res = dict(test_nse=np_masked_nse(y_hat, y_true, mask),
               test_mae=numpy_metrics.masked_mae(y_hat, y_true, mask),
               test_rmse=numpy_metrics.masked_rmse(y_hat, y_true, mask),
               test_mape=numpy_metrics.masked_mape(y_hat, y_true, mask))

    output = trainer.predict(predictor, dataloaders=dm.val_dataloader())
    output = casting.numpy(output)
    y_hat, y_true, mask = output['y_hat'], \
                          output['y'], \
                          output['mask']
    res.update(dict(val_nse=np_masked_nse(y_hat, y_true, mask),
                    val_mae=numpy_metrics.masked_mae(y_hat, y_true, mask),
                    val_rmse=numpy_metrics.masked_rmse(y_hat, y_true, mask),
                    val_mape=numpy_metrics.masked_mape(y_hat, y_true, mask)))
    if args.neptune_logger:
        logger.finalize('success')
    return tsl.logger.info(res)

if __name__ == '__main__':
    parser = ArgParser(add_help=False)
    parser = add_parser_arguments(parser)
    exp = TslExperiment(run_fn=run_experiment, parser=parser, config_path=tsl_config.config['config_dir'])
    exp.run()

