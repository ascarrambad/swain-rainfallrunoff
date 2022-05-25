import os
import pickle
from collections import namedtuple
from typing import Mapping, Callable, Optional

import numpy as np
import pandas as pd

from pytorch_lightning import Trainer

from tsl.nn.utils import casting
from tsl.predictors import Predictor
from tsl.data import SpatioTemporalDataModule
from tsl.utils import numpy_metrics as npm

def masked_nse(y_hat, y, mask=None):
    y_hat = np.where(mask, y_hat, np.full_like(y_hat, np.nan))
    y = np.where(mask, y, np.full_like(y, np.nan))

    num = np.nansum(np.square(y - y_hat))
    den = np.nansum(np.square(y - np.nanmean(y)))

    return 1. - num / den if den != 0 else 999.

class SWAIN_Evaluator(object):
    """docstring for SWAIN_Evaluator"""
    def __init__(self,
                 edge_index: np.ndarray,
                 node_attribs: pd.DataFrame,
                 node_idx_map: Mapping[int, int],
                 trainer: Trainer,
                 predictor: Predictor,
                 window_size: int,
                 data_index: pd.Index,
                 datamodule: SpatioTemporalDataModule,
                 log_dir: str,
                 custom_metrics: Optional[Mapping[str, Callable]] = None):
        super(SWAIN_Evaluator, self).__init__()

        self._edge_index = edge_index
        self._node_attribs = node_attribs
        self._node_idx_map = node_idx_map
        self.trainer = trainer
        self.predictor = predictor
        self._window_size = window_size
        self.data_index = data_index
        self.datamodule = datamodule
        self.custom_metrics = custom_metrics if custom_metrics is not None else dict()
        self.log_dir = log_dir

        self.pred_out = dict()
        self._metrics = []
        self._idx_slices = dict()
        self._graphs = dict()

    def _build_graph(self, split):
        # Generate Graph
        node_idxs = list(self._node_idx_map.values())
        attribs = [dict(rID=n,
                        gID=i,
                        n_name=self._node_attribs.loc[n, 'name'],
                        area=self._node_attribs.loc[n, 'area_gov'],
                        elev=self._node_attribs.loc[n, 'elev'],
                        start=self._node_attribs.loc[n, 'obsbeg_day'],
                        impact_type=self._node_attribs.loc[n, 'typimpact'].replace(',', ', '),
                        impact_deg=self._node_attribs.loc[n, 'degimpact'],
                        hydro_mse_cal=float(np.nan_to_num(self._node_attribs.loc[n, 'cal_MSE'], nan=-999)),
                        hydro_nse_cal=float(np.nan_to_num(self._node_attribs.loc[n, 'cal_NSE'], nan=-999)),
                        hydro_mse_val=float(np.nan_to_num(self._node_attribs.loc[n, 'val_MSE'], nan=-999)),
                        hydro_nse_val=float(np.nan_to_num(self._node_attribs.loc[n, 'val_NSE'], nan=-999)),
                        **self._metrics[i][split]) for n,i in self._node_idx_map.items()]

        graph = nx.Graph()
        graph.add_nodes_from(list(zip(node_idxs, attribs)))
        graph.add_edges_from(self._edge_index.T)

        return graph


    def prepare(self, splits):
        # Output generation
        for split in splits:
            out = self.trainer.predict(self.predictor,
                                       dataloaders=getattr(self.datamodule, f'{split}_dataloader')(shuffle=False))
            self.pred_out[split] = casting.numpy(out)
            self._idx_slices[split] = self.data_index[getattr(self.datamodule, f'{split}_slice')][self._window_size:]

        del out

        # Compute metrics
        for i in self._node_idx_map.values():
            n_dict = {split: dict() for split in self.pred_out.keys()}
            for split, outs in self.pred_out.items():
                y_hat, y, mask = outs['y_hat'], \
                                 outs['y'], \
                                 outs['mask']
                # Retrive single node data, mantaining shape
                data = np.squeeze(y_hat[:,:,i,:]), \
                       np.squeeze(y[:,:,i,:]), \
                       np.squeeze(mask[:,:,i,:])

                # Compute standard metrics
                n_dict[split]['mse'] = float(np.nan_to_num(npm.masked_mse(*data), nan=-999))
                n_dict[split]['mae'] = float(np.nan_to_num(npm.masked_mae(*data), nan=-999))
                n_dict[split]['mape'] = float(np.nan_to_num(npm.masked_mape(*data), nan=-999))
                n_dict[split]['rmse'] = float(np.nan_to_num(npm.masked_rmse(*data), nan=-999))
                n_dict[split]['nse'] = float(np.nan_to_num(masked_nse(*data), nan=-999))

                # Compute custom metrics
                for metric, fn in self.custom_metrics.items():
                    n_dict[split][metric] = float(np.nan_to_num(fn(*data), nan=-999))

            self._metrics.append(n_dict)

        # Compute graphs for different splits
        for split in splits:
            self._graphs[split] = self._build_graph(split)

    def dump(self):
        dump_dict = dict(edge_index=self._edge_index,
                        node_attribs=self._node_attribs,
                        node_idx_map=self._node_idx_map,
                        custom_metrics=self.custom_metrics
                        pred_out=self.pred_out
                        metrics=self._metrics
                        idx_slices=self._idx_slices
                        graphs=self._graphs)

        # Save obj
        with open(os.path.join(self.log_dir, 'eval_dump.pickle'), 'wb') as f:
            pickle.dump(dump_dict, f)