import pandas as pd
import numpy as np
from bokeh.models import *
from bokeh.colors.named import (
    lime as BULL_COLOR,
    tomato as BEAR_COLOR
)
# from bokeh.io import show
from bokeh.layouts import gridplot

from math import pi
from bokeh.plotting import figure, show

from utils import *

def visualize(df: pd.DataFrame, broker: Broker, strategy: Strategy):
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    INIT_CANDLES = 100
    MARGIN_MULTIPLIER_OHLC = 0.03
    MARGIN_MULTIPLIER_VOL = 0.1
    
    bull = df.Close >= df.Open
    bear = df.Close < df.Open
    candle_width = np.min(np.diff(df.Datetime)/2)
    
    # OHLC plot
    # Initially display the most recent INIT_CANDLES candles
    start_ind = len(df) - INIT_CANDLES
    ohlc_plot = figure(x_axis_type="datetime", width = 1200, height = 600, tools=TOOLS, title = "Candlesticks", x_range = (df.Datetime[start_ind], df.Datetime[len(df)-1]), y_range = (min(df.Low[start_ind:])*(1-MARGIN_MULTIPLIER_OHLC), max(df.High[start_ind:])*(1+MARGIN_MULTIPLIER_OHLC)))
    ohlc_plot.xaxis.major_label_orientation = pi/4
    ohlc_plot.grid.grid_line_alpha=0.3
    
    ohlc_plot.segment(df.Datetime, df.High, df.Datetime, df.Low, color="black")
    ohlc_plot.vbar(df.Datetime[bull], candle_width, df.Open[bull], df.Close[bull], fill_color=BULL_COLOR, line_color="black")
    ohlc_plot.vbar(df.Datetime[bear], candle_width, df.Open[bear], df.Close[bear], fill_color=BEAR_COLOR, line_color="black")
    
    # Volume plot
    vol_plot = figure(x_axis_type="datetime", width = 1200, height = 200, tools=TOOLS, title = "Volume", x_range=ohlc_plot.x_range, y_range=(0, max(df.Volume[-INIT_CANDLES:])*(1+MARGIN_MULTIPLIER_VOL)))

    vol_plot.xaxis.major_label_orientation = pi/4
    vol_plot.grid.grid_line_alpha=0.3

    vol_plot.segment(df.Datetime, df.High, df.Datetime, df.Low, color="black")
    vol_plot.vbar(df.Datetime[bull], candle_width, df.Volume[bull], fill_color=BULL_COLOR, line_color="black")
    vol_plot.vbar(df.Datetime[bear], candle_width, df.Volume[bear],fill_color=BEAR_COLOR, line_color="black")
    
    
    # Plot Indicators
    for indicator in strategy.indicators:
        if indicator.overlay:
            ohlc_plot.line(x=df.Datetime, y = indicator.values)
    
    # Plot Trades
    # TODO: only debug for now, need to implement for real
    # ohlc_plot.scatter(np.array([df.Datetime[trade] for trade in broker.closed_trades]), np.array([df.Close[trade] for trade in broker.closed_trades]), size=10)
    
    
    # Display
    plots = [ohlc_plot, vol_plot]
    
    for p in plots:
        p.add_tools(CrosshairTool(dimensions='both'))
    
    fig = gridplot(
        plots,
        ncols=1,
        toolbar_location='right',
        toolbar_options=dict(logo=None),
        merge_tools=True
    )
    show(fig)
