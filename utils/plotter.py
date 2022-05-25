import os
from functools import lru_cache

import numpy as np
import pandas as pd
pd.set_option('plotting.backend', 'pandas_bokeh')

from bokeh.layouts import column, row
from bokeh.palettes import Spectral4, RdYlGn11
from bokeh.tile_providers import get_provider, Vendors
from bokeh.plotting import figure, from_networkx
from bokeh.server.server import Server
from bokeh.models import (Range1d,
                          Circle, MultiLine,
                          HoverTool, TapTool,
                          NodesAndLinkedEdges,
                          StaticLayoutProvider, LinearColorMapper,
                          \
                          ColumnDataSource,
                          Panel, Tabs, PreText, Select)

class SWAIN_Plotter(object):

    def __init__(self, results):
        super(SWAIN_Plotter, self).__init__()

        self.results = results
        self.server = Server({'/': main}, num_procs=4)

    ############################################################################
    # Build and update metrics map

    def _plot_metrics_map(self):
        # Generate plot
        self.metrics_map = figure(title='LamaH-CE Network',
                                  x_range=Range1d(4.150e+6, 5.000e+6),
                                  y_range=Range1d(2.500e+6, 3.100e+6),
                                  tools='pan,wheel_zoom,box_zoom,save,reset',
                                  width=1800,
                                  height=950)

        map_provider = get_provider(Vendors.CARTODBPOSITRON)
        self.metrics_map.add_tile(map_provider)

        hoover_infos = [('Real ID', '@rID'),
                        ('Graph ID', '@gID'),
                        ('Name', '@n_name'),
                        ('Area (km^2)', '@area'),
                        ('Elevation (m a.s.l.)', '@elev'),
                        ('Obs start (year)', '@start'),
                        ('Impact type', '@impact_type'),
                        ('Impact deg', '@impact_deg'),
                        \
                        ('Hydro_MSE_calibr', '@hydro_mse_cal'),
                        ('Hydro_NSE_calibr', '@hydro_nse_cal'),
                        ('Hydro_MSE_valid', '@hydro_mse_val'),
                        ('Hydro_NSE_valid', '@hydro_nse_val'),
                        \
                        ('MSE', '@mse'),
                        ('MAE', '@mae'),
                        ('MAPE', '@mape'),
                        ('RMSE', '@rmse'),
                        ('NSE', '@nse')]
        hoover_cstm_metrics = [(k.upper(), f'@{k.lower()}') for k in self.results.custom_metrics.keys()]

        node_hover_tool = HoverTool(tooltips=hoover_infos + hoover_cstm_metrics)
        self.metrics_map.add_tools(node_hover_tool, TapTool())

        # Build graph layout
        node_idxs = list(self.results.node_idx_map.values())
        x = self.results.node_attribs.loc[:, 'lon']
        y = self.results.node_attribs.loc[:, 'lat']
        self._graph_layout = dict(zip(node_idxs, zip(x, y)))

        return self.metrics_map

    def _update_metrics_map(self):
        split = self.splits_ticker.value

        graph_renderer = from_networkx(self.results.graphs[split],
                                       StaticLayoutProvider(graph_layout=self._graph_layout)
                                       scale=2,
                                       center=(0,0))


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

        if len(graph_renderer.node_renderer.data_source.selected.indices) == 0:
            graph_renderer.node_renderer.data_source.selected.indices = [self.results.node_idx_map[399]]
        graph_renderer.node_renderer.data_source.selected.on_change("indices", self._update_ts)
        self.metrics_map.renderers = [graph_renderer]

    ############################################################################

    ############################################################################
    # Build and update timeseries plot

    def _plot_timeseries(self):
        # Data Source and plot tools
        self.ts_source = ColumnDataSource(data=dict(date=[],
                                                    y_hat=[],
                                                    y_true=[],
                                                    residuals=[],
                                                    meanc_resids=[]))
        tools = 'pan,wheel_zoom,box_zoom,save,reset'

        # Figures setup
        self.main_ts = figure(width=1300, height=300, tools=tools, x_axis_type='datetime')
        self.main_ts.line('date', 'y_true', source=self.ts_source, color='blue', legend_label='Observed Runoff (m^2/sec)')
        self.main_ts.line('date', 'y_hat', source=self.ts_source, color='orange', legend_label='Predicted Runoff (m^2/sec)')

        self.resids_ts = figure(width=1300, height=300, tools=tools, x_axis_type='datetime')
        self.resids_ts.x_range = self.main_ts.x_range
        self.resids_ts.line('date', 'residuals', source=self.ts_source, color='red', legend_label='Residuals')

        # ts3 = figure(width=1300, height=300, tools=tools, x_axis_type='datetime')
        # ts3.x_range = self.main_ts.x_range
        # ts3.line('date', 'meanc_resids', source=self.ts_source, color='red', legend_label='Mean-centered Residuals')

        return self.main_ts, self.resids_ts

    def _plot_ts_stats(self):
        self.main_ts_stats = PreText(text='', width=500)
        self.resids_ts_stats = PreText(text='', width=500)
        # stats3 = PreText(text='', width=500)

        return self.main_ts_stats, self.resids_ts_stats

    # Data retrieving function
    @lru_cache()
    def _update_ts_data(self, split, node_idx):
        # Get model outputs
        y_hat, y_true = self.results.pred_out[split]['y_hat'][:,:,node_idx,:], \
                        self.results.pred_out[split]['y'][:,:,node_idx,:]

        # Flatten arrays
        y_hat = np.ravel(y_hat)
        y_true = np.ravel(y_true)

        return pd.DataFrame(dict(date=self.results.idx_slices[split],
                                 y_hat=y_hat,
                                 y_true=y_true,
                                 residuals=y_true - y_hat,
                                 meanc_resids=(y_true - y_hat) - np.nanmean(y_true - y_hat)))

    # Callbacks
    def _update_ts_hist(self, df):
        split = self.splits_ticker.value

        rootLayout = self.plot_doc.get_model_by_name('ts_layout')
        listOfSubLayouts = rootLayout.children

        hist_df = pd.DataFrame({'y_true_all': self.results.pred_out[split]['y'].ravel())
        hist_df = pd.concat([df[['y_hat','y_true']], hist_df], axis=1)

        hist_plot = hist_df.plot.hist(bins=np.linspace(0, 3000, 1000),
                                      vertical_xlabel=True,
                                      hovertool=False,
                                      show_figure=False,
                                      line_color="black")
        if len(listOfSubLayouts) == 3:
            listOfSubLayouts[0][0] = hist_plot
        else:
            listOfSubLayouts[0].insert(0, hist_plot)

        y_hat, y_true = self.results.pred_out[split]['y_hat'].ravel() \
                        self.results.pred_out[split]['y'].ravel()

        hist_df = pd.DataFrame({'residuals_all': y_true - y_hat)
        hist_df = pd.concat([df[['residuals']], hist_df], axis=1)

        hist_plot = hist_df.plot.hist(bins=np.linspace(0, 3000, 1000),
                                      vertical_xlabel=True,
                                      hovertool=False,
                                      show_figure=False,
                                      line_color="black")
        if len(listOfSubLayouts) == 3:
            listOfSubLayouts[1][0] = hist_plot
        else:
            listOfSubLayouts[1].insert(0, hist_plot)


    def _update_ts_stats(self, df):
        self.main_ts_stats.text = str(df[['y_hat', 'y_true']].describe())
        self.resids_ts_stats.text = str(df[['residuals']].describe())
        # stats3.text = str(df[['meanc_resids']].describe())

    def _update_ts(self):
        # Callbacks
        split = self.splits_ticker.value
        node_idx = graph_renderer.node_renderer.data_source.selected.indices[0]
        node = self.results.idx_node_map(node_idx)
        df = self._update_ts_data(split, node_idx)

        self.ts_source.data = df
        self._update_ts_hist(df)
        self._update_ts_stats(df)

        self.main_ts.title.text = f'Predictions for catchment: {node}'
        self.resids_ts.title.text = f'Residuals for catchment: {node}'
        # ts3.title.text = f'Mean-centered Residuals for catchment: {node}'

    def _split_change(self, attrname, old, new):
        self._update_metrics_map()
        self._update_ts()

    ############################################################################

    ############################################################################

    def _build_map_tab(self):
        return row()

    def _build_ts_tab(self):

        splits_label = PreText(text='Select data split: ')

        splits_ticker_options = ['train', 'val', 'test']
        self.splits_ticker = Select(value=splits_ticker_options[0], options=splits_ticker_options)
        self.splits_ticker.on_change('value', self._split_change)

        metrics_map = self._plot_metrics_map()
        main_ts, resids_ts = self._plot_timeseries()
        main_ts_stats, resids_ts_stats = self._plot_ts_stats()

        return column(row(splits_label, self.splits_ticker),
                      row(metrics_map),
                      row(main_ts, main_ts_stats),
                      row(resids_ts, resids_ts_stats))

    def build(self, doc):
        self.plot_doc = doc
        # Widgets setup
        title = PreText(text='SWAIN Project: Day-ahead Rainfall-Runoff predictions')

        map_tab = Panel(child=self._build_map_tab(), 'Map')
        ts_tab = Panel(child=self._build_ts_tab(), 'Timeseries')

        layout = column(title, Tabs(tabs=[map_tab, ts_tab]))

        doc.add_root(layout)
        doc.title = "SWAIN Project: Day-ahead Rainfall-Runoff predictions"


    def run(self):
        self._update_metrics_map()
        self._update_ts()

        self.server.start()

        return self.server