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
from bokeh.transform import factor_cmap

from utils import *

def add_crosshair_labels(plot, renderer):
    # https://github.com/bokeh/bokeh/issues/3000#issuecomment-2028617843
    # https://docs.bokeh.org/en/2.4.3/docs/user_guide/tools.html#crosshairtool
    # https://docs.bokeh.org/en/2.4.3/docs/reference/models/formatters.html#bokeh.models.DatetimeTickFormatter
    callback_hovertool_x = CustomJS(code="""
        var tooltips = document.getElementsByClassName('bk-Tooltip');
        tooltips[0].style.left = '5px';
    """)
    plot.add_tools(HoverTool(
        point_policy="follow_mouse",
        renderers=[renderer],
        mode='vline',
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
    plot.add_tools(HoverTool(
        point_policy="follow_mouse",
        renderers=[renderer],
        mode='vline',
        tooltips="""
        <div>
            <span style="font-size: 16px; color: black;">@Datetime{%Y-%m-%d %T}</span>
        </div>
        """,
        formatters= {
            "@Datetime" : "datetime"
        },
        callback=callback_hovertool_y,
    ))

def visualize(df: pd.DataFrame, broker: Broker, strategy: Strategy):
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    INIT_CANDLES = 100
    MARGIN_MULTIPLIER_OHLC = 0.03
    MARGIN_MULTIPLIER_VOL = 0.1

    candle_width = np.min(np.diff(df.Datetime)/2)
    source = ColumnDataSource(df)
    source.add((df.Close >= df.Open).values.astype(np.uint8).astype(str), 'inc')
    
    # TODO: Fix gaps on non-business days and hours
    # TODO: if sub-day chart, shade in background of after and pre-market candles
    # OHLC plot
    # Initially display the most recent INIT_CANDLES candles
    start_ind = len(df) - INIT_CANDLES
    ohlc_plot = figure(
        x_axis_type="datetime", 
        width = 1200, 
        height = 600, 
        tools=TOOLS, 
        title = "Candlesticks", 
        x_range = (df.Datetime[start_ind], df.Datetime[len(df)-1]), 
        y_range = (min(df.Low[start_ind:])*(1-MARGIN_MULTIPLIER_OHLC), max(df.High[start_ind:])*(1+MARGIN_MULTIPLIER_OHLC))
    )
    ohlc_plot.xaxis.major_label_orientation = pi/4
    ohlc_plot.grid.grid_line_alpha=0.3
    
    ohlc_plot.segment("Datetime", "High", "Datetime", "Low", source=source, color="black")
    m_ohlc = ohlc_plot.vbar("Datetime", candle_width, "Open", "Close", source=source, line_color="black",
                   fill_color=factor_cmap('inc', [BEAR_COLOR, BULL_COLOR], ['0', '1']))

    # NBSP = '\N{NBSP}' * 4
    # ohlc_tooltips = [
    #     ("Datetime", "@Datetime{%Y-%m-%d}"),
    #     ('OHLC', NBSP.join(('@Open{0,0.00}',
    #                         '@High{0,0.00}',
    #                         '@Low{0,0.00}',
    #                         '@Close{0,0.00}'))),
    #     ('Volume', '@Volume{0,0}')]

    # ohlc_plot.add_tools(HoverTool(
    #     point_policy="follow_mouse",
    #     mode='vline',
    #     renderers=[m_ohlc],
    #     tooltips=ohlc_tooltips,
    #     formatters={
    #         '@Datetime' : 'datetime'
    #     }
    # ))
    
    add_crosshair_labels(ohlc_plot, m_ohlc)
    
    # Volume plot
    vol_plot = figure(
        x_axis_type="datetime", 
        width = 1200, 
        height = 200, 
        tools=TOOLS, 
        title = "Volume", 
        x_range=ohlc_plot.x_range, 
        y_range=(0, max(df.Volume[-INIT_CANDLES:])*(1+MARGIN_MULTIPLIER_VOL))
    )

    vol_plot.xaxis.major_label_orientation = pi/4
    vol_plot.grid.grid_line_alpha=0.3

    vol_plot.segment("Datetime", "High", "Datetime", "Low", source=source, color="black")
    m_vol = vol_plot.vbar("Datetime", candle_width, "Volume", source=source, line_color="black",
                   fill_color=factor_cmap('inc', [BEAR_COLOR, BULL_COLOR], ['0', '1']))
    
    add_crosshair_labels(vol_plot, m_vol)
    
    # Plot Indicators
    for ix, indicator in enumerate(strategy.indicators):
        if indicator.overlay:
            ohlc_plot.line(x=df.Datetime, y=indicator.values, line_color=dark_colors[ix], legend_label=indicator.name)
        else:
            pass # TODO: implement RSI, MACD etc separate graphs 
        
        
    # Plot Trades
    buy_t, buy_prices = [], []
    sell_t, sell_prices = [], []
    for trade in broker.closed_trades:
        if trade.num_shares > 0: # Long trades
            buy_t.extend(df.Datetime[trade.entry_indices])
            buy_prices.extend(trade.entry_prices)
            sell_t.extend(df.Datetime[trade.exit_indices])
            sell_prices.extend(trade.exit_prices)
        else: # Short trades
            sell_t.extend(df.Datetime[trade.entry_indices])
            sell_prices.extend(trade.entry_prices)
            buy_t.extend(df.Datetime[trade.exit_indices])
            buy_prices.extend(trade.exit_prices)
    ohlc_plot.scatter(np.array(buy_t), np.array(buy_prices), size=10, marker="triangle")
    ohlc_plot.scatter(np.array(sell_t), np.array(sell_prices), size=10, marker="inverted_triangle")
   
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
