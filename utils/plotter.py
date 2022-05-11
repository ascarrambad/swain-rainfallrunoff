import os
import pickle
from typing import Mapping, Callable, Optional

import numpy as np
import pandas as pd
import networkx as nx
from pytorch_lightning import Trainer

from bokeh.layouts import column, row
from bokeh.palettes import Spectral4, RdYlGn11
from bokeh.plotting import figure, from_networkx
from bokeh.io import output_file, save
from bokeh.models import (Range1d,
                          Circle, MultiLine,
                          HoverTool, TapTool,
                          NodesAndLinkedEdges,
                          GraphRenderer, StaticLayoutProvider, LinearColorMapper)

import pydot
from networkx.drawing.nx_pydot import graphviz_layout

from tsl.nn.utils import casting
from tsl.predictors import Predictor
from tsl.data import SpatioTemporalDataModule
from tsl.utils import numpy_metrics as npm

def masked_nse(y_hat, y, mask=None):
    y_hat = np.where(mask, y_hat, np.full_like(y_hat, np.nan))
    y = np.where(mask, y, np.full_like(y, np.nan))

    num = np.nansum(np.square(y - y_hat))
    den = np.nansum(np.square(y - np.nanmean(y)))

    return 1. - num / den

class SWAIN_Plotter(object):
    """docstring for SWAIN_Plotter"""
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
        super(SWAIN_Plotter, self).__init__()

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


    def dump(self):
        # Delete bulky and needless stuff
        del self.trainer
        del self.predictor
        del self.datamodule
        del self.data_index

        # Save obj
        with open(os.path.join(self.log_dir, 'plotter.pickle'), 'wb') as f:
            pickle.dump(self, f)

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
                n_dict[split]['mse'] = npm.masked_mse(*data)
                n_dict[split]['mae'] = npm.masked_mae(*data)
                n_dict[split]['mape'] = npm.masked_mape(*data)
                n_dict[split]['rmse'] = npm.masked_rmse(*data)
                n_dict[split]['nse'] = masked_nse(*data)

                # Compute custom metrics
                for metric, fn in self.custom_metrics.items():
                    n_dict[split][metric] = fn(*data)

            self._metrics.append(n_dict)


    def generate_metrics_map(self, split: str, geo_layout: Optional[bool] = True):
        # Generate Graph
        node_idxs = list(self._node_idx_map.values())
        attribs = [dict(ID=n,
                        gname=self._node_attribs.loc[n, 'name'],
                        area=self._node_attribs.loc[n, 'area_gov'],
                        elev=self._node_attribs.loc[n, 'elev'],
                        start=self._node_attribs.loc[n, 'obsbeg_day'],
                        **self._metrics[i][split]) for n,i in self._node_idx_map.items()]

        graph = nx.Graph()
        graph.add_nodes_from(list(zip(node_idxs, attribs)))
        graph.add_edges_from(self._edge_index.T)

        # Generate plot
        plot = figure(title='LamaH-CE Network',
                      x_range=Range1d(4.150e+6, 5.000e+6),
                      y_range=Range1d(2.500e+6, 3.100e+6),
                      tools='pan,wheel_zoom,box_zoom,save,reset',
                      width=1800,
                      height=950)

        hoover_infos = [('ID', '@ID'),
                        ('Name', '@gname'),
                        ('Area (km^2)', '@area'),
                        ('Elevation (m a.s.l.)', '@elev'),
                        ('Obs start (year)', '@start'),
                        \
                        ('MSE', '@mse'),
                        ('MAE', '@mae'),
                        ('MAPE', '@mape'),
                        ('RMSE', '@rmse'),
                        ('NSE', '@nse')]
        hoover_cstm_metrics = [(k.upper(), f'@{k.lower()}') for k in self.custom_metrics.keys()]

        node_hover_tool = HoverTool(tooltips=hoover_infos + hoover_cstm_metrics)
        plot.add_tools(node_hover_tool, TapTool())

        graph_renderer = from_networkx(graph,
                                       graphviz_layout(graph, prog='fdp'),
                                       scale=2,
                                       center=(0,0))

        if geo_layout:
            x = self._node_attribs.loc[:, 'lon']
            y = self._node_attribs.loc[:, 'lat']
            graph_layout = dict(zip(node_idxs, zip(x, y)))
            graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

        color_map = list(RdYlGn11)
        color_map.reverse()
        node_color_map = LinearColorMapper(palette=color_map,
                                           low=-1,
                                           high=1)

        graph_renderer.node_renderer.glyph = Circle(size=10, fill_color={'field': 'nse',
                                                                         'transform': node_color_map})
        graph_renderer.node_renderer.selection_glyph = Circle(size=10, fill_color=Spectral4[2])
        graph_renderer.node_renderer.hover_glyph = Circle(size=10, fill_color=Spectral4[1])

        graph_renderer.edge_renderer.glyph = MultiLine(line_color='#CCCCCC', line_alpha=0.8, line_width=2)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=2)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=2)

        graph_renderer.selection_policy = NodesAndLinkedEdges()
        plot.renderers.append(graph_renderer)

        output_file(os.path.join(self.log_dir, 'node_metrics.html'))
        save(plot)