# -*- coding: utf-8 -*-
"""
============================================================
Log plotting
============================================================

 Author:    Paul de Groot <paul.degroot@dgbes.com>
 Copyright: dGB Beheer BV
 Date:      January 2020
 License:   GPL v3


@author: paul
"""
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.palettes import all_palettes
from bokeh.models.tickers import FixedTicker
from bokeh.models import (LinearAxis, Range1d, LogAxis,
                              LabelSet, ColumnDataSource)
from bokeh.layouts import gridplot
from collections import OrderedDict

scaledlog = pd.DataFrame()

def removeNone(lst):
    return [x for x in lst if x != 'None']

def getShadingBounds(inputlog):
    minbound = inputlog.min()
    maxbound = inputlog.max()
    extension = (maxbound - minbound) / 20
    minbound = minbound - extension
    maxbound =  maxbound + extension
    return(minbound, maxbound)

def getScaledLog(nr, minval, maxval, minvalues, maxvalues, data, logname):
    if (nr <= 1) :
        scalingfactor = 1
        scalingintercept = 0
    else:
        targetrange = float(maxvalues[0]) - float(minvalues[0])
        logrange = float(maxval) - float(minval)
        scalingfactor = targetrange / logrange
        scalingintercept = (-targetrange * float(minval) / logrange
                            + float(minvalues[0]))
        minvalues[nr] = minvalues[0]
        maxvalues[nr] = maxvalues[0]
    scaledlog = (data.loc[:, logname].copy() * scalingfactor + scalingintercept)
    return (scaledlog, minvalues, maxvalues)

def create_gridplot(alldata):
    nrlogplots = int(alldata['nrlogplots'].value)
    plotmarkers = alldata['plotmarkers'].value
    plotmarkersyesno = plotmarkers[0]
    a = {}
    plotlist = []
    for nr in range(nrlogplots):
        key = 'plot'+str(nr)
        a[key] = create_plot(alldata, nr, plotmarkersyesno)
        plotlist.append(a[key])
        if (key == 'plot0'):
            y_range = a[key].y_range
        else:
            a[key].y_range = y_range
    if (plotmarkersyesno != 'None'):
        plotmarkersyesno = 'Last'
        markerplot = create_plot(alldata, nr, plotmarkersyesno)
        plotlist.append(markerplot)
    grid = gridplot([plotlist])
    return(grid)

# convert hexadecimal to RGB tuple
def hex_to_rgb(hex):
    red = ''.join(hex.strip('#')[0:2])
    green = ''.join(hex.strip('#')[2:4])
    blue = ''.join(hex.strip('#')[4:6])
    return (int(red, 16), int(green, 16), int(blue,16))

def rgb_from_list(colorname):
    colordict ={'darkred': [139,0,0],
                'navy' : [0,0,128],
                'orange': [255,165,0], 
                'darkgreen':[0,100,0] , 
                'black': [0,0,0],
                'cyan': [[0,255,255]], 
                'darkslategrey': [47,79,79], 
                'gold' :[255,215,0], 
                'lime': [0,255,0], 
                'magenta': [255,0,255]}
    return(colordict[colorname])

def get_index(depthdata, firstdepth, lastdepth):
    nrdepths = len(depthdata)
    tolerance = 10 * (abs(lastdepth)-abs(firstdepth)) / nrdepths
    uniqueidx = pd.Index(list(depthdata))
    firstidx = uniqueidx.get_loc(firstdepth, 
                                 method='nearest', tolerance = tolerance)
    lastidx = uniqueidx.get_loc(lastdepth, 
                                method='nearest', tolerance = tolerance)
    return(firstidx, lastidx)

def add_shading(plot, scaledlogs, depthdata, inputmindepth,
                inputmaxdepth, lognames, xrange2, 
                shadingtype, shadinglogs, linestyles, mypalette,
                colorfillpalette, colorfill, shadingcolor):
    (firstidx, lastidx) = get_index(depthdata, inputmindepth, inputmaxdepth)
    leftbound = pd.Series()
    rightbound = pd.Series()
    shadingdifferencelog1 = 'None'
    shadingdifferencelog2 = 'None'
    tmpcurve = scaledlogs[shadinglogs[0]][firstidx:lastidx+1]
    firstcurve = tmpcurve.reset_index()
    del firstcurve['index']
    firstcurve = firstcurve.squeeze()
    (minbound, maxbound) = getShadingBounds(firstcurve)
    if (shadingtype == 'Difference 2 logs shading color'):
        if (len(shadinglogs) >= 2):
            shadingdifferencelog1 = shadinglogs[0]
            shadingdifferencelog2 = shadinglogs[1]
            tmpcurve = scaledlogs[shadinglogs[1]][firstidx:lastidx+1]
            secondcurve = tmpcurve.reset_index()
            del secondcurve['index']
            secondcurve = secondcurve.squeeze()
            print ("Warning: a difference plot cannot be combined with",
                   "curve plots. Selected curves are ignored.",
                   "Similarly, the shading is applied between the",
                   "first and second log; additional selections are ignored.")
        else:
            print('error: select 2 logs for difference shading')
    if ((shadingtype == 'Left palette') or
            (shadingtype == 'Left shading color')):
        leftbound = pd.Series(minbound for x in range(len(firstcurve)))
        rightbound = firstcurve
    if ((shadingtype == 'Right palette') or
            (shadingtype == 'Right shading color')):
        leftbound = firstcurve
        rightbound = pd.Series(maxbound for x in range(len(firstcurve)))
    if (shadingtype == 'Column wide palette'):
        leftbound = pd.Series(minbound for x in range(len(firstcurve)))
        rightbound = pd.Series(maxbound for x in range(len(firstcurve)))
    if (shadingtype == 'Difference 2 logs shading color'):
        leftbound = firstcurve
        rightbound = secondcurve       
        plot.line(leftbound, depthdata[firstidx:lastidx+1], 
                  name=shadingdifferencelog1, 
                  line_color=mypalette[-2], 
                  line_width=2,
                  line_dash =  linestyles[-2]) 
        plot.line(rightbound, depthdata[firstidx:lastidx+1], 
                  name=shadingdifferencelog2, 
                  line_color=mypalette[-1], 
                  line_width=2,
                  line_dash =  linestyles[-1])
    (minvalue, maxvalue) = (minbound, maxbound)
    scalefactor = (float(255)/(float(maxvalue)-float(minvalue)))
    N = lastidx+1 - firstidx
    M = 255 # width of the RGBA image
    rgbadata = np.empty((N, M, 4), dtype=np.uint8)
    (red, green, blue) = rgb_from_list(shadingcolor)
    (rgbadata[:,:,0],rgbadata[:,:,1], rgbadata[:,:,2]) = (
                red, green, blue)                          
    if (colorfill == 'palette'):
        for x in range(N):
            if not (np.isnan(firstcurve[x])):
                value = firstcurve[x] - minvalue
                (rgbadata[x,:,0],rgbadata[x,:,1], rgbadata[x,:,2]) = (
                     hex_to_rgb(
                     all_palettes[colorfillpalette][256][int(value * scalefactor)])
                     ) 
    samplerate = (abs(inputmaxdepth - inputmindepth) / N)
    dh = N * samplerate
    dw = abs(maxvalue - minvalue)
    rgbadata[:,:,3] = 0
    if (shadingtype == 'Difference 2 logs shading color'):
        for x in range(N):
            if not (np.isnan(firstcurve[x])):
                minvalx = min(leftbound[x], rightbound[x]) - minvalue
                maxvalx = max(leftbound[x], rightbound[x]) - minvalue
                rgbadata[x, max(0,int(minvalx * scalefactor)) : 
                             min(int(maxvalx * scalefactor),M), 3] = 125
    else:
        for x in range(N):
            if not (np.isnan(firstcurve[x])):
                minvalx = leftbound[x] - minvalue
                maxvalx = rightbound[x] - minvalue
                rgbadata[x, max(0,int(minvalx * scalefactor)) : 
                              min(int(maxvalx * scalefactor),M), 3] = 125
    rgbaflip = np.flipud(rgbadata)
    plot.x_range = Range1d(minvalue , maxvalue)
    plot.y_range = Range1d(inputmaxdepth, inputmindepth)
    plot.image_rgba(image=[rgbaflip], 
                        x=minvalue, y=inputmaxdepth, dw=dw, dh=dh, level="image")    
    return(plot)

def add_markers(plot, markers, plotmarkers, plotmarkersyesno,
                minbound, maxbound):
    markerdepths = list(markers['MD'])
    markerdepths = [m * -1 for m in markerdepths]
    markernames = list(markers['Name'])
    markercolors =  list(markers['Color'])
    xseries = [minbound, maxbound]
    xlabels = []
    ylabels = []
    textlabels = []
    linecolors = []
    if (plotmarkers[0] == 'All'):
        plotmarkers = list(markers['Name'])
    for j in range(len(plotmarkers)):
        for i in range(len(markerdepths)):
            if (plotmarkers[j] == markernames[i]):
                yseries = [markerdepths[i], markerdepths[i]]
                xlabels.append(minbound)
                ylabels.append(markerdepths[i])
                textlabels.append(markernames[i])
                linecolors.append(markercolors[i])
    if (plotmarkersyesno == 'Last'):
        plot.outline_line_color = None
        plot.xaxis.visible = False
        plot.xgrid.visible = False
        plot.ygrid.visible = False
        plot.background_fill_color = None
        plot.border_fill_color = None
        plot.title.text = 'Markers'
        source = ColumnDataSource(data=dict(xlabel=xlabels,
                           ylabel = ylabels,
                           textlabel = textlabels ))
        labels = LabelSet(x='xlabel', y='ylabel',
                          text='textlabel', level='glyph',
                          x_offset=15, y_offset=5, source=source,
                          render_mode='canvas')
        plot.add_layout(labels)
        for i in range(len(plotmarkers)):
            yseries = [ylabels[i], ylabels[i]]
            plot.line(xseries[:], yseries[:],
                      color=linecolors[i],
                      line_width=2)
    if (plotmarkersyesno != 'None'):
        for i in range(len(plotmarkers)):
            yseries = [ylabels[i], ylabels[i]]
            plot.line(xseries[:], yseries[:],
                  color=linecolors[i],
                  line_width=2)
    return(plot)

#Make a Bokeh plot
def create_plot(alldata, nr, plotmarkersyesno):
    data = alldata['data']
    headers = alldata['headers']
    markers = alldata['markers']
    plotmarkers = alldata['plotmarkers'].value
    xlogscales = alldata['xlogscales']
    nrcurves = alldata['nrcurves']
    xaxistype = (alldata['rowofcolcurves'].children[nr].
                    children[0].active)
    selmullogs = (alldata['rowofcolcurves'].children[nr].
                    children[1])
    shadingtype = (alldata['rowofcolcurves'].children[nr].
                    children[2].value)
    shadinglogs = list((alldata['rowofcolcurves'].children[nr].
                    children[3]).value)
    shadingcolor =  alldata['shadingcolor'].value
    colorfillpalette = alldata['colorfillpalette'].value
    colorfill = 'single'
    xrange2 = 'dummy'
    rowofcol = alldata['rowofcol']
    if (((shadingtype == 'Left palette') or
         (shadingtype == 'Right palette') or
         (shadingtype == 'Column wide palette'))):
        colorfill = 'palette'
    shadingdifferencelog1 = 'None'
    shadingdifferencelog2 = 'None'
    if (shadingtype == 'Difference 2 logs shading color'):
        if (len(shadinglogs) >= 2):
            shadingdifferencelog1 = shadinglogs[0]
            shadingdifferencelog2 = shadinglogs[1]
        else:
            print('error: select 2 logs for difference shading')
    depthlogstr = headers[0]
    depthdata = data.loc[:, depthlogstr]
    logwidth = int(alldata['logwidth'].value)
    logheight = int(alldata['logheight'].value)
    inputmindepth = float(alldata['inputmindepth'].value)
    inputmaxdepth = float(alldata['inputmaxdepth'].value)
    depthticks = int(alldata['depthticks'].value)
    depthminorticks = int(alldata['depthminorticks'].value)
    ylabel = depthlogstr
    lognames = list(selmullogs.value)
    if ((shadingtype != 'None') and
            (shadinglogs[0] != 'None')):
        lognames.append(shadinglogs[0])
    if (shadingtype == 'Difference 2 logs shading color'):
        lognames = shadinglogs # No other logs allowed for difference shading
    lognames = removeNone(lognames)
    lognames = list(OrderedDict.fromkeys(lognames)) # remove double entries
    xaxistype = xlogscales[0]
    mypalette = []
    scaledlogs = pd.DataFrame()
    counter = []
    minvalues = []
    maxvalues = []
    linestyles = []
    for i in range(len(lognames)):
        for nm in range(nrcurves):
            if (lognames[i] == (rowofcol.children[0].
                                 children[nm].value)):
                minval = (rowofcol.children[1].
                                 children[nm].value)
                maxval = (rowofcol.children[2].
                                 children[nm].value)
                minvalues.append(minval)
                maxvalues.append(maxval)
                (scaledlog, minvalues, maxvalues) = getScaledLog(i,
                                        minval, maxval,
                                        minvalues, maxvalues,
                                        data, lognames[i])
                scaledlogs = pd.concat([scaledlogs, scaledlog], axis=1)
                linestyles.append(rowofcol.children[3].
                                 children[nm].value)
                mypalette.append((rowofcol.children[4].
                                 children[nm].value))
                break
        counter.append(i)

    if (len(lognames) <=2):
        plot = figure(plot_width=logwidth, plot_height=logheight,
                     x_axis_type = xaxistype,
                     x_axis_label = lognames[0],
                     x_axis_location = 'above',
                     background_fill_color="#f0f0f0",
                     tools='ypan,ywheel_zoom,reset,hover',
                     y_axis_label=ylabel,
#                     output_backend='webgl' // gives weird look when zoomed out
                     )
    else:
        xaxislabel = lognames[0]
        for i in range(2, len(lognames)):
            xaxislabel = xaxislabel + "," + lognames[i] + "(rescaled)"
        plot = figure(plot_width=logwidth, plot_height=logheight,
                     x_axis_type = xaxistype,
                     x_axis_label = xaxislabel,
                     x_axis_location = 'above',
                     background_fill_color="#f0f0f0",
                     tools='ypan,ywheel_zoom,reset,hover',
                     y_axis_label=ylabel,
#                     output_backend='webgl' // gives weird look when zoomed out
                     )
    ticker = []
    for i in range(0,10000,depthticks):
        ticker.append(i)
    minorticks = []
    for i in range(0,10000,depthminorticks):
        minorticks.append(i)
    plot.yaxis.ticker = FixedTicker(ticks=ticker,
                                    minor_ticks = minorticks)
    plot.ygrid.grid_line_color = 'navy'
    plot.ygrid.grid_line_alpha = 0.2
    plot.ygrid.minor_grid_line_color = 'navy'
    plot.ygrid.minor_grid_line_alpha = 0.1
    plot.title.text = 'Plot ' + str(nr+1)
    plot.y_range = Range1d(inputmaxdepth , inputmindepth) 
    plot.x_range = Range1d(float(minvalues[0]), float(maxvalues[0]))
    if (plotmarkersyesno != 'Last'):
        for count, name, minvalue, maxvalue, linestyle, color in zip(counter,
                                      lognames,
                                      minvalues,
                                      maxvalues,
                                      linestyles,
                                      mypalette):
            if (count == 0):
                if ((name != shadingdifferencelog1) and (name != shadingdifferencelog2)):            
                    plot.line(scaledlogs[name], depthdata[:], 
                              name=name, 
                              line_color=color, 
                              line_width=2,
                              line_dash = linestyle)
            if (count == 1):
                xrange2 = name
                plot.extra_x_ranges={xrange2: Range1d(float(minvalues[1]),
                                                   float(maxvalues[1]))}
                if (shadingtype == 'Difference 2 logs shading color'):
                    plot.extra_x_ranges={xrange2: Range1d(float(minvalues[0]),
                                                       float(maxvalues[0]))}
                if (xaxistype == 'linear'):
                     plot.add_layout(LinearAxis(x_range_name=xrange2,
                            axis_label=name), 'above')
                if (xaxistype == 'log'):
                      plot.add_layout(LogAxis(x_range_name=xrange2,
                            axis_label=name), 'above')
                if ((name != shadingdifferencelog1) and (name != shadingdifferencelog2)):            
                    plot.line(scaledlogs[name], depthdata[:], 
                              name=name, 
                              line_color=color, 
                              x_range_name = name,
                              line_width = 2,
                              line_dash =  linestyle)
            if (count > 1):
                if ((name != shadingdifferencelog1) and (name != shadingdifferencelog2)):            
                    plot.line(scaledlogs[name], depthdata[:], 
                              name=name, 
                              line_color=color, 
                              line_width=2,
                              line_dash =  linestyle)
                    print ("Warning: only two separate X-ranges supported.",
                           "Additional curves are scaled to the range of the first curve.")

    # shading plots
        if ((shadingtype != 'None') and
            (shadinglogs[0] != 'None')):
            plot = add_shading(plot, scaledlogs, depthdata, inputmindepth,
                            inputmaxdepth, lognames, xrange2,
                            shadingtype, shadinglogs, linestyles, mypalette,
                            colorfillpalette, colorfill, shadingcolor)
# marker plots
    minval = float(minvalues[0])
    maxval = float(maxvalues[0])
    extension = (maxval - minval) / 20
    minbound = minval - extension
    maxbound =  maxval + extension

    if len(markers)>0:
      plot = add_markers(plot, markers, plotmarkers, plotmarkersyesno,
                       minbound, maxbound)
    return (plot)

