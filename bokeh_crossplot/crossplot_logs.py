"""
============================================================
Log crossplot
============================================================

 Author:    Paul de Groot <paul.degroot@dgbes.com>
 Copyright: dGB Beheer BV
 Date:      March 2019
 License:   GPL v3
 Credits:   Based in parts on Crossfilter example of www.bokeh.org.

@author: paul
"""

import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models.tickers import FixedTicker
from bokeh.models import Range1d, LinearAxis, ColumnDataSource, ColorBar, LinearColorMapper, HoverTool
from bokeh.layouts import gridplot
from sklearn import linear_model

def create_plots(alldata):
    (source, data) = getsource(alldata)
    plottype = alldata['plottype']
    if (plottype.value == 'Bubbles'):
        xplot = create_bubblecrossplot(alldata, source, data)
    else:
        xplot = create_scattercrossplot(alldata, source, data)
    a = {}
    plotlist = []
    for nr in range(2):
        key = 'plot'+str(nr)
        a[key] = create_logplot(alldata, nr, source, data)
        plotlist.append(a[key])
        if (key == 'plot0'):
            y_range = a[key].y_range
        else:
            a[key].y_range = y_range
    grid = gridplot([plotlist],toolbar_location="left")
    return(grid, xplot)

def prepare_data(alldata):
    data = alldata['data']
    x = alldata['x']
    y = alldata['y']
    data = data.dropna(subset=[x.value, y.value])
    mindepth = alldata['mindepth']
    maxdepth = alldata['maxdepth']
    minidx, maxidx = (0,len(data))
    for i in range(len(data)):
        if ( abs(data.iloc[i,0] >= mindepth)):
            minidx = i
            break
    for i in range(len(data)):
        if (abs(data.iloc[i,0] >= maxdepth)):
            maxidx = i+1
            break
    datarange = data.iloc[minidx:maxidx, :].copy()
    datarange.reset_index()
    return(datarange)

def getsource(alldata):
    data = prepare_data(alldata)
    depthdata = -data.iloc[:, 0]
    x = alldata['x']
    y = alldata['y']
    size = alldata['size']
    color = alldata['color']
    N_SIZES = alldata['N_SIZES']
    SIZES = alldata['SIZES']
    N_COLORS = alldata['N_COLORS']
    COLORS = alldata['COLORS']
    xs = data[x.value].values
    ys = data[y.value].values

    sz = 9
    szlist = [sz]
    sizes = np.asarray(szlist * xs.size)
    szs = data[x.value].values # dummy to ensure column is filled even if None
    if size.value != 'None':
        szs = data[size.value].values
        if len(set(data[size.value])) > N_SIZES:
            groups = pd.qcut(data[size.value].values,
                             N_SIZES, duplicates='drop')
        else:
            groups = pd.Categorical(data[size.value])
        sizes = np.asarray([SIZES[xx] for xx in groups.codes])

    col = "#31AADE"
    colist = [col]
    cols = np.asarray(colist * xs.size)
    cs = data[x.value].values # dummy to ensure column is filled even if None
    if color.value != 'None':
        cs = data[color.value].values
        if len(set(data[color.value])) > N_SIZES:
            groups = pd.qcut(data[color.value].values,
                             N_COLORS, duplicates='drop')
        else:
            groups = pd.Categorical(data[color.value])
        cols = np.asarray([COLORS[xx] for xx in groups.codes])
    source = ColumnDataSource(data=dict(depthdata=depthdata,
                                        xsrc=xs, ysrc=ys, csrc=cs, ssrc=szs,
                                        cols=cols, sizes=sizes))
    return(source, data)

def create_histogramplots(alldata, source, data, p):
    # create the horizontal histogram
    x= alldata['x']
    y = alldata['y']
    xs = data[x.value].values
    ys = data[y.value].values
    hhist, hedges = np.histogram(xs, bins=20)
#    hzeros = np.zeros(len(hedges)-1)
    hmax = max(hhist)*1.1

    ph = figure(toolbar_location=None, plot_width=p.plot_width,
                plot_height=200, x_range=p.x_range,
                y_range=(0, hmax), min_border=10, min_border_left=50,
                y_axis_location="right")
    ph.xgrid.grid_line_color = None
    ph.yaxis.major_label_orientation = np.pi/4
    ph.background_fill_color = "#f0f0f0"

    ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:],
            top=hhist, line_color="darkorange",
            line_width=2, fill_color='darkgray')

    # create the vertical histogram
    vhist, vedges = np.histogram(ys, bins=20)
#    vzeros = np.zeros(len(vedges)-1)
    vmax = max(vhist)*1.1

    pv = figure(toolbar_location=None, plot_width=200,
                plot_height=p.plot_height, x_range=(0, vmax),
                y_range=p.y_range, min_border=10, y_axis_location="right")
    pv.ygrid.grid_line_color = None
    pv.xaxis.major_label_orientation = np.pi/4
    pv.background_fill_color = "#f0f0f0"

    pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist,
            line_color="royalblue", fill_color='darkgray',
            line_width=2)

    return (ph, pv)

def create_bubblecrossplot(alldata, source, data):
    x = alldata['x']
    y = alldata['y']
    colorlog = alldata['color']
    stats = alldata['stats']
    stats.text = ' '
    x_title = x.value
    y_title = y.value
    headers = alldata['headers']

    kw = dict()
    kw['title'] = "%s vs %s" % (x_title, y_title)

    TOOLTIPS = [
      (x_title,"$x{0.000}"),
      (y_title,"$y{0.000}")
    ]

    p = figure(plot_height=600, plot_width=800,
               background_fill_color="#f0f0f0",
               tools='pan,box_zoom,lasso_select,box_select,hover,reset',
               tooltips=TOOLTIPS,
               title = kw['title'])
    p.xaxis.axis_label = x_title
    p.yaxis.axis_label = y_title

    if x.value in headers:
        p.xaxis.major_label_orientation = pd.np.pi / 4

    minval = 0
    maxval = 1
    title = "Color bar"
    if colorlog.value != 'None':
        minval = min(data[colorlog.value].values)
        maxval = max(data[colorlog.value].values)
        title = colorlog.value

    p.circle('xsrc', 'ysrc', color='cols', size='sizes', line_color="white",
             source=source,
             alpha=0.6, hover_color='white', hover_alpha=0.5,
             selection_color="red", nonselection_alpha=0.1,
             selection_alpha=0.4)
    (ph,pv) = create_histogramplots(alldata, source, data, p)
    color_mapper = LinearColorMapper(palette=alldata['COLORS'],
                                  low=minval, high=maxval)
    color_bar = ColorBar(color_mapper=color_mapper,
                     label_standoff=12, border_line_color=None,
                     location=(0,0))
    color_bar_plot = figure(title=title, title_location="right",
                        height=200, width=160,
                        toolbar_location=None, min_border=0,
                        outline_line_color=None)
    color_bar_plot.add_layout(color_bar, 'right')
    xplot = gridplot([[p, pv], [ph, color_bar_plot], [stats, None]],
                     merge_tools=False)
    return (xplot)


def create_scattercrossplot(alldata, source, data):
    x= alldata['x']
    y = alldata['y']
    headers = alldata['headers']
    xs = data[x.value].values
    ys = data[y.value].values
    x_title = x.value
    y_title = y.value
    stats = alldata['stats']

    TOOLTIPS = [
      (x_title,"$x{0.000}"),
      (y_title,"$y{0.000}")
    ]

    kw = dict()
    kw['title'] = "%s vs %s" % (x_title, y_title)
    p = figure(plot_height=600, plot_width=800,
               background_fill_color="#f0f0f0",
               tools='pan,box_zoom,lasso_select, box_select, hover, reset',
               tooltips=TOOLTIPS,
               title = kw['title'])
    p.xaxis.axis_label = x_title
    p.yaxis.axis_label = y_title

    if x.value in headers:
        p.xaxis.major_label_orientation = pd.np.pi / 4

    # Fit line using all data
    lr = linear_model.LinearRegression()
    X = xs.reshape(-1,1)
    lr.fit(X, ys)
    r2full = round(lr.score(X,ys), 4)
    coeffull = round(float(lr.coef_) ,  4)
    interceptfull = round(float(lr.intercept_) , 4)

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, ys)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    r2ransac = round(ransac.score(X[inlier_mask], ys[inlier_mask]), 4)
    coefransac = round(float(ransac.estimator_.coef_) ,  4)
    interceptransac = round(float(ransac.estimator_.intercept_) , 4)

    # Predict data of estimated models
    xmin = X.min()
    xmax = X.max()
    line_X = []
    line_X.append(xmin)
    line_X.append(xmax)
    line_X = (np.asarray(line_X)).reshape(-1,1)
    line_y = lr.predict(line_X)
    line_y_ransac = ransac.predict(line_X)

    lw = 2
    sz = 3
    p.scatter('xsrc', 'ysrc', source = source,
             color='yellowgreen', marker='square', size=sz,
             selection_color="red", nonselection_alpha=0.1,
             selection_alpha=0.4)
    p.scatter(xs[outlier_mask], ys[outlier_mask],
             color='gold', marker='square', size=sz)
    X = np.reshape(line_X, len(line_X))
    p.line(X, line_y, color='navy', line_width=lw, legend='Full range')
    p.line(X, line_y_ransac, color='cornflowerblue', line_width=lw,
           legend='Robust')
    p.legend.location = 'bottom_right'
    p.legend.click_policy= 'hide'

    stats.text = ("Linear regression statistics: \n \n" +
                    "Full range: \n" +
                    "slope = " + str(coeffull) +
                    "    intercept = " + str(interceptfull) +
                    "    r2 = " + str(r2full) + "\n \n" +
                    "Robust (RANSAC algorithm): \n" +
                    "slope = " + str(coefransac) +
                    "    intercept = " + str(interceptransac) +
                    "    r2 = " + str(r2ransac) + "\n \n" )
    (ph,pv) = create_histogramplots(alldata, source, data, p)
    xplot = gridplot([[p, pv], [ph, None], [stats, None]], merge_tools=False)

    return (xplot)

#Make a Bokeh plot
def create_logplot(alldata, nr, source, data):
    depthdata = -data.iloc[:, 0]
    x= alldata['x']
    y = alldata['y']
    size = alldata['size']
    color = alldata['color']
    logwidth = 200
    logheight = 800
    mindepth = alldata['mindepth']
    maxdepth = alldata['maxdepth']
    depthticks = 50
    depthminorticks = 10
    linecolors = ['darkorange', 'royalblue', 'indigo', 'forestgreen']
    ylabel = data.columns[0]
    lognames = []
    minvalues = []
    maxvalues = []
    source = source
    if (nr == 0):
        lognames.append(x.value)
        lognames.append(y.value)
        linecolor1 = linecolors[0]
        linecolor2 = linecolors[1]

    else:
        if (color.value != 'None'):
            lognames.append(color.value)
        if (size.value != 'None'):
            lognames.append(size.value)
        linecolor1 = linecolors[2]
        linecolor2 = linecolors[3]
        ylabel = ''

    for i in range (len(lognames)):
        if (lognames[i] != 'None'):
            minval = data[lognames[i]].dropna().min()
            maxval = data[lognames[i]].dropna().max()
            minval = minval - (maxval - minval) / 20
            maxval = maxval + (maxval - minval) / 20
            minvalues.append(minval)
            maxvalues.append(maxval)

    if (len(lognames) == 0):
        x_axis_label = 'None'
    else:
        x_axis_label = lognames[0]

    TOOLTIPS = [
      ("Name","$name"),
      ("Value","$x{0.000}"),
      ("MD","$y{0.00}")
    ]

    hover = HoverTool( tooltips=TOOLTIPS, mode="mouse" )

    plot = figure(plot_width=logwidth,
                  plot_height=logheight,
                  x_axis_label = x_axis_label,
                  x_axis_location = 'above',
                  background_fill_color="#f0f0f0",
                  tools='ypan,ywheel_zoom,box_select,reset',
                  y_axis_label=ylabel)
    plot.add_tools( hover )

    ticker = []
    for i in range(0,-10000,-depthticks):
        ticker.append(i)
    minorticks = []
    for i in range(0,-10000,-depthminorticks):
        minorticks.append(i)
    plot.yaxis.ticker = FixedTicker(ticks=ticker,
                                    minor_ticks = minorticks)
    plot.ygrid.grid_line_color = 'navy'
    plot.ygrid.grid_line_alpha = 0.2
    plot.ygrid.minor_grid_line_color = 'navy'
    plot.ygrid.minor_grid_line_alpha = 0.1
    plot.y_range = Range1d(-maxdepth , -mindepth)
    if (nr == 0):
        plot.title.text = 'X-axis & Y-axis logs'
        plot.line(data[lognames[0]], depthdata[:], legend=lognames[0],
                  color= linecolor1,
                  line_width=2, name=lognames[0])
        plot.x_range = Range1d(float(minvalues[0]), float(maxvalues[0]))
        plot.circle('xsrc', 'depthdata', color=linecolor1, size=2,
                 source=source, alpha=0.6,
                 selection_color="red", nonselection_alpha=0,
                 selection_alpha=0.4, name=lognames[0])
        xrange2 = lognames[1]
        plot.add_layout(LinearAxis(x_range_name=xrange2,
                axis_label=xrange2), 'above')
        plot.extra_x_ranges={xrange2: Range1d(float(minvalues[1]),
                                       float(maxvalues[1]))}
        plot.line(data[lognames[1]], depthdata[:], legend=lognames[1],
                  x_range_name = xrange2,
                  color=linecolor2,
                  line_width=2, name=lognames[1])
        plot.circle('ysrc', 'depthdata', color=linecolor2, size=2,
                 x_range_name = xrange2,
                 source=source, alpha=0.6,
                 selection_color="red", nonselection_alpha=0,
                 selection_alpha=0.4, name=lognames[1])

    else:
        plot.title.text = 'Color & Size logs'
        if (len(lognames) >= 1):
                plot.line(data[lognames[0]], depthdata[:], legend=lognames[0],
                          color= linecolor1,
                          line_width=2, name=lognames[0])
                plot.x_range = Range1d(float(minvalues[0]), float(maxvalues[0]))
                if (color.value == lognames[0]):
                    plot.circle('csrc', 'depthdata', color=linecolor1, size=2,
                             source=source, alpha=0.6,
                             selection_color="orange", nonselection_alpha=0.1,
                             selection_alpha=0.4, name=lognames[0])
                if (size.value == lognames[0]):
                    plot.circle('ssrc', 'depthdata', color=linecolor1, size=2,
                             source=source, alpha=0.6,
                             selection_color="red", nonselection_alpha=0.1,
                             selection_alpha=0.4, name=lognames[0])
        if (len(lognames) == 2):
            if (lognames[1] != 'None'):
                xrange2 = lognames[1]
                plot.add_layout(LinearAxis(x_range_name=xrange2,
                        axis_label=xrange2), 'above')
                plot.extra_x_ranges={xrange2: Range1d(float(minvalues[1]),
                                               float(maxvalues[1]))}
                plot.line(data[lognames[1]], depthdata[:], legend=lognames[1],
                          x_range_name = xrange2,
                          color=linecolor2,
                          line_width=2, name=lognames[1])
                plot.circle('ssrc', 'depthdata', color=linecolor2, size=2,
                         x_range_name = xrange2,
                         source=source, alpha=0.6,
                         selection_color="red", nonselection_alpha=0.1,
                         selection_alpha=0.4, name=lognames[1])

    return (plot)
