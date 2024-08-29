import pandas as pd
import numpy as np
from bokeh.models import *
from bokeh.colors.named import (
    lime as BULL_COLOR,
    tomato as BEAR_COLOR
)

import bokeh.palettes

dark_colors = bokeh.palettes.Dark2_8

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
    m1 = ohlc_plot.vbar(df.Datetime[bull], candle_width, df.Open[bull], df.Close[bull], fill_color=BULL_COLOR, line_color="black")
    m2 = ohlc_plot.vbar(df.Datetime[bear], candle_width, df.Open[bear], df.Close[bear], fill_color=BEAR_COLOR, line_color="black")
    
    
    callback_hovertool_x = CustomJS(code="""
    var tooltips = document.getElementsByClassName('bk-Tooltip');

    tooltips[0].style.left = '5px';

    """)
    ohlc_plot.add_tools(HoverTool(
        point_policy="follow_mouse",
        mode='vline',
        renderers=[m1, m2],
        tooltips="""
        <div>
            <span style="font-size: 16px; color: black;">$y{0,0.00}</span>
        </div>
        """,
        callback=callback_hovertool_x,

    ))

    callback_hovertool_y = CustomJS(code="""
        var tooltips = document.getElementsByClassName('bk-Tooltip');

        tooltips[1].style.top = '480px';

    """)
    ohlc_plot.add_tools(HoverTool(
        point_policy="follow_mouse",
        mode='vline',
        renderers=[m1, m2],
        tooltips="""
        <div>
            <span style="font-size: 16px; color: black;">$x</span>
        </div>
        """,
        formatters={
            '$x' : 'datetime'
        },
        callback=callback_hovertool_y,
    ))


    # Volume plot
    vol_plot = figure(x_axis_type="datetime", width = 1200, height = 200, tools=TOOLS, title = "Volume", x_range=ohlc_plot.x_range, y_range=(0, max(df.Volume[-INIT_CANDLES:])*(1+MARGIN_MULTIPLIER_VOL)))

    vol_plot.xaxis.major_label_orientation = pi/4
    vol_plot.grid.grid_line_alpha=0.3

    vol_plot.segment(df.Datetime, df.High, df.Datetime, df.Low, color="black")
    vol_plot.vbar(df.Datetime[bull], candle_width, df.Volume[bull], fill_color=BULL_COLOR, line_color="black")
    vol_plot.vbar(df.Datetime[bear], candle_width, df.Volume[bear],fill_color=BEAR_COLOR, line_color="black")
    
    
    # Plot Indicators
    for ix, indicator in enumerate(strategy.indicators):
        if indicator.overlay:
            ohlc_plot.line(x=df.Datetime, y=indicator.values, line_color=dark_colors[ix], legend_label=indicator.name)
        else:
            pass # TODO: implement RSI, MACD etc separate graphs 
        
        
    # Plot Trades
    # TODO: only debug for now, need to implement for real
    # Entry points
    entry_t, entry_prices = [], []
    for trade in broker.closed_trades:
        entry_t.extend(df.Datetime[trade.entry_indices])
        entry_prices.extend(trade.entry_prices)
    ohlc_plot.scatter(np.array(entry_t), np.array(entry_prices), size=10, marker="triangle")
    
    # Exit points
    exit_t, exit_prices = [], []
    for trade in broker.closed_trades:
        exit_t.extend(df.Datetime[trade.exit_indices])
        exit_prices.extend(trade.exit_prices)
    ohlc_plot.scatter(np.array(exit_t), np.array(exit_prices), size=10, marker="inverted_triangle")
   
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
