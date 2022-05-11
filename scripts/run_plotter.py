import os
import sys
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd

from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, PreText, Select

from tsl.utils import ArgParser


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

parser = ArgParser(add_help=False)
parser.add_argument("--plotter-path", type=str, default='')
parser.add_argument("--split", type=str, default='')

known_args, _ = parser.parse_known_args()

# Load obj
with open(known_args.plotter_path, 'rb') as file:
    plotter = pickle.load(file)

split=known_args.split


# Data Source and plot tools
source = ColumnDataSource(data=dict(date=[],
                                    y_hat=[],
                                    y_true=[],
                                    residuals=[],
                                    meanc_resids=[]))
tools = 'pan,wheel_zoom,box_zoom,save,reset'

# Figures setup
ts1 = figure(width=1300, height=300, tools=tools, x_axis_type='datetime')
ts1.line('date', 'y_true', source=source, color='blue', legend_label='True runoff (m^2/sec)')
ts1.line('date', 'y_hat', source=source, color='orange', legend_label='Predicted runoff (m^2/sec)')

ts2 = figure(width=1300, height=300, tools=tools, x_axis_type='datetime')
ts2.x_range = ts1.x_range
ts2.line('date', 'residuals', source=source, color='red', legend_label='Residuals')

ts3 = figure(width=1300, height=300, tools=tools, x_axis_type='datetime')
ts3.x_range = ts1.x_range
ts3.line('date', 'meanc_resids', source=source, color='red', legend_label='Mean-centered Residuals')

# Widgets setup
ticker_label = PreText(text='Select catchment ID: ')
stats1 = PreText(text='', width=500)
stats2 = PreText(text='', width=500)
stats3 = PreText(text='', width=500)

ticker_options = [str(i) for i in plotter._node_idx_map.keys()]
ticker = Select(value=ticker_options[0], options=ticker_options)

# Data retrieving function
@lru_cache()
def get_data(ticker_opt):
    ticker_opt = int(ticker_opt)

    # Get model outputs
    y_hat, y_true = plotter._outs[split]['y_hat'][:,:,plotter._node_idx_map[ticker_opt],:], \
                    plotter._outs[split]['y'][:,:,plotter._node_idx_map[ticker_opt],:]

    # Flatten arrays
    y_hat = np.ravel(y_hat)
    y_true = np.ravel(y_true)

    return pd.DataFrame(dict(date=plotter._idx_slices[split],
                             y_hat=y_hat,
                             y_true=y_true,
                             residuals=y_true - y_hat,
                             meanc_resids=(y_true - y_hat) - np.nanmean(y_true - y_hat)))

# Callbacks
def update_stats(df):
    stats1.text = str(df[['y_hat', 'y_true']].describe())
    stats2.text = str(df[['residuals']].describe())
    stats3.text = str(df[['meanc_resids']].describe())

def update(selected=None):
    df = get_data(ticker.value)

    source.data = df
    update_stats(df)

    ts1.title.text = f'Predictions for catchment: {ticker.value}'
    ts2.title.text = f'Residuals for catchment: {ticker.value}'
    ts3.title.text = f'Cumulative Residuals for catchment: {ticker.value}'


def ticker_change(attrname, old, new):
    update()

ticker.on_change('value', ticker_change)

# Configure layout
widgets = row(ticker_label, ticker)
preds = row(ts1, stats1)
resids = row(ts2, stats2)
meanc_resids = row(ts3, stats3)

layout = column(widgets, preds, resids, meanc_resids)

# Generate plot
update()
curdoc().add_root(layout)
curdoc().title = "SWAIN Project: LamaH-CE Day-ahead Rainfall-Runoff predictor"