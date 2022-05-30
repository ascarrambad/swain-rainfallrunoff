import os
from functools import lru_cache

import numpy as np
import pandas as pd
import geopandas as gpd
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
                          Panel, Tabs, PreText, Select, RadioGroup, Div)

def getGeometryCoords(row, geom, coord_type, shape_type):
    """
    Returns the coordinates ('x' or 'y') of edges of a Polygon exterior.

    :param: (GeoPandas Series) row : The row of each of the GeoPandas DataFrame.
    :param: (str) geom : The column name.
    :param: (str) coord_type : Whether it's 'x' or 'y' coordinate.
    :param: (str) shape_type
    """

    # Parse the exterior of the coordinate
    if shape_type.lower() == 'polygon':
        if coord_type == 'x':
            # Get the x coordinates of the exterior
            return list(row[geom].exterior.coords.xy[0])
        elif coord_type == 'y':
            # Get the y coordinates of the exterior
            return list(row[geom].exterior.coords.xy[1])
    elif shape_type.lower() == 'point':
        if coord_type == 'x':
            # Get the x coordinates of the exterior
            return row[geom].x
        elif coord_type == 'y':
            # Get the y coordinates of the exterior
            return row[geom].y
    elif shape_type.lower() == 'linestring':
        if coord_type == 'x':
            return list(row[geom].coords.xy[0])
        elif coord_type == 'y':
            return list(row[geom].coords.xy[1])


def convert_GeoPandas_to_Bokeh_format(gdf):
    """
    Function to convert a GeoPandas GeoDataFrame to a Bokeh
    ColumnDataSource object.

    :param: (GeoDataFrame) gdf: GeoPandas GeoDataFrame with polygon(s) under
                                the column name 'geometry.'

    :return: ColumnDataSource for Bokeh.
    """

    shape_type = gdf.iloc[0].geometry.type

    gdf_new = gdf.drop('geometry', axis=1).copy()
    gdf_new['x'] = gdf.apply(getGeometryCoords,
                             geom='geometry',
                             coord_type='x',
                             shape_type=shape_type,
                             axis=1)

    gdf_new['y'] = gdf.apply(getGeometryCoords,
                             geom='geometry',
                             coord_type='y',
                             shape_type=shape_type,
                             axis=1)

    return gdf_new

class SWAIN_Plotter(object):

    def __init__(self, results):
        super(SWAIN_Plotter, self).__init__()

        self.results = results
        self._idx_node_map = {y: x for x, y in self.results.node_idx_map.items()}

        self.server = Server({'/': self.build})

    ############################################################################
    # Build and update metrics map

    def _plot_metrics_map(self):
        # Generate plot
        self.metrics_map = figure(title='LamaH-CE Network',
                                  toolbar_location='above',
                                  x_axis_type='mercator',
                                  y_axis_type='mercator',
                                  tools='pan,wheel_zoom,box_zoom,save,reset',
                                  width=850,
                                  height=475)

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

        coords = gpd.GeoDataFrame(dict(geometry=gpd.points_from_xy(x, y)),
                                  crs='EPSG:3035').to_crs('EPSG:3857')

        coords = convert_GeoPandas_to_Bokeh_format(coords)

        self._graph_layout = dict(zip(node_idxs, zip(coords['x'].to_list(), coords['y'].to_list())))


        # Build commands & infos

        self.metrics_label = Div(text='', width=700)

        def replot_map(attr, old, new):
            self._update_metrics_map()

        node_size_label = PreText(text='Node size by:', width=110)
        self.node_size_ctrl = RadioGroup(labels=['same', 'elev', 'area_gov'], active=0, inline=True)
        self.node_size_ctrl.on_change('active', replot_map)

        node_color_label = PreText(text='Node color by:', width=110)
        self.node_color_ctrl = RadioGroup(labels=['nse', 'mse'], active=0, inline=True)
        self.node_color_ctrl.on_change('active', replot_map)

        return column(self.metrics_map,
                      self.metrics_label,
                      row(node_size_label, self.node_size_ctrl, node_color_label, self.node_color_ctrl))

    def _update_metrics_map(self):
        split = self.splits_ticker.value

        graph_renderer = from_networkx(graph=self.results.graphs[split],
                                       layout_function=self._graph_layout,
                                       scale=2,
                                       center=(0,0))


        color_map = list(RdYlGn11)
        if self.node_color_ctrl.active == 0:
            color_map.reverse()
            node_color_map = LinearColorMapper(palette=color_map,
                                               low=0,
                                               high=1)
        else:
            node_color_map = LinearColorMapper(palette=color_map,
                                               low=0,
                                               high=max([n[split]['mse'] for n in self.results.metrics]))

        node_color_opt = ['nse', 'mse']
        node_size_opt = [10, 'elev', 'area_gov']

        if self.node_size_ctrl.active != 0:
            node_sizes = self.results.node_attribs.loc[list(self.results.node_idx_map.keys()), node_size_opt[self.node_size_ctrl.active]] \
                                                  .to_numpy()
            node_sizes = (node_sizes - np.min(node_sizes)) / (np.max(node_sizes) - np.min(node_sizes))
            node_sizes = node_sizes * (25 - 5) + 5
            graph_renderer.node_renderer.data_source.data['node_size'] = node_sizes

        graph_renderer.node_renderer.glyph = Circle(size='node_size' if self.node_size_ctrl.active != 0 else 10,
                                                    fill_color={'field': node_color_opt[self.node_color_ctrl.active],
                                                                'transform': node_color_map})
        graph_renderer.node_renderer.selection_glyph = Circle(size='node_size' if self.node_size_ctrl.active != 0 else 10,
                                                              fill_color=Spectral4[2])
        graph_renderer.node_renderer.hover_glyph = Circle(size='node_size' if self.node_size_ctrl.active != 0 else 10,
                                                          fill_color=Spectral4[1])

        graph_renderer.edge_renderer.glyph = MultiLine(line_color='#002aff', line_alpha=0.8, line_width=2)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=2)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=2)

        graph_renderer.selection_policy = NodesAndLinkedEdges()

        if len(self.metrics_map.renderers) == 1:
            graph_renderer.node_renderer.data_source.selected.indices = [self.results.node_idx_map[399]]
        else:
            graph_renderer.node_renderer.data_source.selected.indices = self.metrics_map.renderers[1].node_renderer.data_source.selected.indices
        graph_renderer.node_renderer.data_source.selected.on_change("indices", self._update_ts)

        self.metrics_map.renderers = [self.metrics_map.renderers[0]] + [graph_renderer]


    ############################################################################

    ############################################################################
    # Build and update metrics cdfs

    def _update_metrics_cdfs(self):

        split = self.splits_ticker.value
        hydro_split = 'val' if split == 'test' else 'cal'

        # Retrieve metrics data
        nse = pd.DataFrame(dict(nse=[n[split]['nse'] for n in self.results.metrics]),
                           index=list(self.results.node_idx_map.keys()))
        hydro_nse = self.results.node_attribs.loc[:, hydro_split + '_NSE']
        nse_df = pd.concat([nse, hydro_nse], axis=1)

        mse = pd.DataFrame(dict(mse=[n[split]['mse'] for n in self.results.metrics]),
                           index=list(self.results.node_idx_map.keys()))
        hydro_mse = self.results.node_attribs.loc[:, hydro_split + '_MSE']
        mse_df = pd.concat([mse, hydro_mse], axis=1)


        mape_df = pd.DataFrame(dict(mape=[n[split]['mape'] for n in self.results.metrics]),
                               index=list(self.results.node_idx_map.keys()))

        rootLayout = self.plot_doc.get_model_by_name('ts_layout')
        listOfSubLayouts = rootLayout.children

        nse_hist_plot = nse_df.plot.hist(bins=np.linspace(0, 1, 100),
                                         xticks=np.linspace(0, 1, 10),
                                         figsize=(300, 300),
                                         vertical_xlabel=True,
                                         hovertool=False,
                                         normed=True,
                                         toolbar_location='above',
                                         legend='top_left',
                                         show_figure=False)

        nse_cdf_plot = nse_df.plot.hist(bins=np.linspace(0, 1, 100),
                                        xticks=np.linspace(0, 1, 10),
                                        figsize=(300, 300),
                                        vertical_xlabel=True,
                                        hovertool=False,
                                        cumulative=True,
                                        normed=True,
                                        toolbar_location='above',
                                        legend='top_left',
                                        show_figure=False)

        mse_hist_plot = mse_df.plot.hist(bins=np.linspace(0, 100, 100),
                                         xticks=np.linspace(0, 100, 10),
                                         figsize=(300, 300),
                                         vertical_xlabel=True,
                                         hovertool=False,
                                         normed=True,
                                         toolbar_location='above',
                                         show_figure=False)

        mse_cdf_plot = mse_df.plot.hist(bins=np.linspace(0, 100, 100),
                                        xticks=np.linspace(0, 100, 10),
                                        figsize=(300, 300),
                                        vertical_xlabel=True,
                                        hovertool=False,
                                        cumulative=True,
                                        normed=True,
                                        toolbar_location='above',
                                        legend='bottom_right',
                                        show_figure=False)

        mape_hist_plot = mape_df.plot.hist(bins=np.linspace(0, 1, 100),
                                           xticks=np.linspace(0, 1, 10),
                                           figsize=(300, 300),
                                           vertical_xlabel=True,
                                           hovertool=False,
                                           normed=True,
                                           toolbar_location='above',
                                           show_figure=False)

        mape_cdf_plot = mape_df.plot.hist(bins=np.linspace(0, 1, 100),
                                          xticks=np.linspace(0, 1, 10),
                                          figsize=(300, 300),
                                          vertical_xlabel=True,
                                          hovertool=False,
                                          cumulative=True,
                                          normed=True,
                                          toolbar_location='above',
                                          legend='bottom_right',
                                          show_figure=False)

        plots = row(column(nse_hist_plot, nse_cdf_plot),
                    column(mse_hist_plot, mse_cdf_plot),
                    column(mape_hist_plot, mape_cdf_plot))

        if len(listOfSubLayouts[1].children) == 2:
            listOfSubLayouts[1].children[1] = plots
        else:
            listOfSubLayouts[1].children.append(plots)


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
        self.main_ts = figure(width=1100,
                              height=300,
                              tools=tools,
                              x_axis_type='datetime',
                              toolbar_location='above')
        self.main_ts.line('date', 'y_true', source=self.ts_source, color='blue', legend_label='Observed Runoff (m^2/sec)')
        self.main_ts.line('date', 'y_hat', source=self.ts_source, color='orange', legend_label='Predicted Runoff (m^2/sec)')

        self.resids_ts = figure(width=1100,
                                height=300,
                                tools=tools,
                                x_axis_type='datetime',
                                toolbar_location='above')
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

        # hist_df = pd.DataFrame({'y_true_all': self.results.pred_out[split]['y'].ravel()})
        hist_df = df[['y_hat','y_true']]

        hist_plot = hist_df.plot.hist(bins=np.linspace(0, hist_df.max().max(), 100),
                                      xticks=np.linspace(0, hist_df.max().max(), 10),
                                      figsize=(400, 300),
                                      vertical_xlabel=True,
                                      hovertool=False,
                                      toolbar_location='above',
                                      show_figure=False)
        if len(listOfSubLayouts[2].children) == 3:
            listOfSubLayouts[2].children[0] = hist_plot
        else:
            listOfSubLayouts[2].children.insert(0, hist_plot)

        y_hat, y_true = self.results.pred_out[split]['y_hat'].ravel(), \
                        self.results.pred_out[split]['y'].ravel()

        # hist_df = pd.DataFrame({'residuals_all': y_true - y_hat})
        hist_df = df[['residuals']]

        hist_plot = hist_df.plot.hist(bins=np.linspace(0, hist_df.max().max(), 100),
                                      xticks=np.linspace(0, hist_df.max().max(), 10),
                                      figsize=(400, 300),
                                      vertical_xlabel=True,
                                      hovertool=False,
                                      toolbar_location='above',
                                      show_figure=False)
        if len(listOfSubLayouts[3].children) == 3:
            listOfSubLayouts[3].children[0] = hist_plot
        else:
            listOfSubLayouts[3].children.insert(0, hist_plot)


    def _update_ts_stats(self, df):
        self.main_ts_stats.text = str(df[['y_hat', 'y_true']].describe())
        self.resids_ts_stats.text = str(df[['residuals']].describe())
        # stats3.text = str(df[['meanc_resids']].describe())

    def _update_ts_metrics(self):
        metrics_label = ''
        i = self.metrics_map.renderers[1].node_renderer.data_source.selected.indices[0]
        for metric, value in self.results.metrics[i][self.splits_ticker.value].items():
            metrics_label += f'<b>{metric.upper()}</b>: {value:.2f}, '

        self.metrics_label.text = metrics_label[:-2]

    def _update_ts(self, attr, old, new):
        # Callbacks
        split = self.splits_ticker.value
        if len(self.metrics_map.renderers[1].node_renderer.data_source.selected.indices) == 0:
            node_idx = self.results.node_idx_map[399]
        else:
            node_idx = self.metrics_map.renderers[1].node_renderer.data_source.selected.indices[0]
        node = self._idx_node_map[node_idx]
        df = self._update_ts_data(split, node_idx)

        self.ts_source.data = df
        self._update_ts_metrics()
        self._update_ts_hist(df)
        self._update_ts_stats(df)

        self.main_ts.title.text = f'Predictions for catchment: {node}'
        self.resids_ts.title.text = f'Residuals for catchment: {node}'
        # ts3.title.text = f'Mean-centered Residuals for catchment: {node}'

    def _split_change(self, attrname, old, new):
        self._update_metrics_map()
        self._update_metrics_cdfs()
        self._update_ts(attrname, old, new)

    ############################################################################

    ############################################################################

    def _build_map_tab(self):
        return row()

    def _build_ts_tab(self):

        splits_label = PreText(text='Select data split: ')

        splits_ticker_options = ['train', 'val', 'test']
        self.splits_ticker = Select(value=splits_ticker_options[2], options=splits_ticker_options)
        self.splits_ticker.on_change('value', self._split_change)

        metrics_map = self._plot_metrics_map()
        main_ts, resids_ts = self._plot_timeseries()
        main_ts_stats, resids_ts_stats = self._plot_ts_stats()

        return column(row(splits_label, self.splits_ticker),
                      row(metrics_map),
                      row(main_ts, main_ts_stats),
                      row(resids_ts, resids_ts_stats), name='ts_layout')

    def build(self, doc):
        self.plot_doc = doc
        # Widgets setup
        title = Div(text='<h2>SWAIN Project: Day-ahead Rainfall-Runoff predictions</h2>')

        map_tab = self._build_map_tab()
        ts_tab = self._build_ts_tab()

        map_tab = Panel(child=map_tab, title='Map')
        ts_tab = Panel(child=ts_tab, title='Timeseries')

        layout = column(title, Tabs(tabs=[map_tab, ts_tab]))

        doc.add_root(layout)
        doc.title = "SWAIN Project: Day-ahead Rainfall-Runoff predictions"

        self._update_metrics_map()
        self._update_metrics_cdfs()
        self._update_ts('','','')


    def run(self):
        self.server.start()
        return self.server