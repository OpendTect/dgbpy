# -*- coding: utf-8 -*-
"""

@author: paul, arnaud
"""

# load and plot faces

import numpy as np

from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper

from dgbpy import keystr as dgbkeys
from dgbpy import mlio as dgbmlio

# load the current cubelet
def load_data(plotparams):
    slicepos = plotparams['slicepos']
    inlnr = slicepos[0]
    crlnr = slicepos[1]
    znr = slicepos[-1]
    cubenr = plotparams['cubeidx']
    iattr = plotparams['attribidx']
    info = plotparams[dgbkeys.infodictstr]
    surveynm = plotparams[dgbkeys.surveydictstr]
    collnm = plotparams[dgbkeys.collectdictstr]
    learntype = info[dgbkeys.learntypedictstr]
    
    xslices = [] # inline, crossline, zslice from input cubelet
    yslices = [] # inline, crossline, zslice from target cubelet
    datadict = dgbmlio.getTrainingDataByInfo(info, dsetsel={ surveynm: {collnm: [cubenr]} } )
    xdata = datadict[dgbkeys.xtraindictstr][0,iattr]
    ydata = datadict[dgbkeys.ytraindictstr][0]
    if len(ydata.shape) == 1:
        ydata = ydata[0]
    else:
        ydata = ydata[iattr]
    xslices.append(xdata[inlnr , :, :].transpose())
    xslices.append(xdata[: , crlnr , :].transpose())
    xslices.append(xdata[: , : , znr ].transpose())
    if learntype == dgbkeys.seisimgtoimgtypestr:
        yslices.append(ydata[inlnr , :, :].transpose())
        yslices.append(ydata[: , crlnr , :].transpose())
        yslices.append(ydata[: , : , znr ].transpose())
    elif learntype == dgbkeys.seisclasstypestr:
        yslices = ydata
    return (xslices, yslices)
 
# plot a list of loaded slices
def plot_slices(plotparams):
    (xslices, yslices) = load_data(plotparams)
    slicepos = plotparams['slicepos']
    mypalette = plotparams['palette']
    zoom = plotparams['zoom']
    dim =  plotparams['nrdims']
    cubenr = plotparams['cubeidx']
    titlelist = plotparams['titles']
    info = plotparams[dgbkeys.infodictstr]
    learntype = info[dgbkeys.learntypedictstr]
    a = {}
    plotlist = []
    titlepart1 = 'Input '
    data = xslices
    if dim == 3 and learntype == dgbkeys.seisimgtoimgtypestr:
        for t in range(2):
            if t == 1:
                data = yslices
                titlepart1 = ' Target '
            for i in range(dim):
                nr = slicepos[i]
                titlepart2 = titlelist[i]
                title = titlepart1 + titlepart2 + str(nr)
                flipped = np.flipud(data[i])
                key = 'plot'+str(i)
                n = flipped.shape[0]
                m = flipped.shape[1]
                maxval = np.amax(flipped)
                minval = np.amin(flipped)
                color_mapper = LinearColorMapper(palette=mypalette, 
                                                 low=minval, high=maxval)
                color_bar = ColorBar(color_mapper=color_mapper, 
                                     ticker=BasicTicker(),
                                     location=(0,0))
                a[key] = figure(frame_height=int(zoom * n), frame_width=int(zoom * m),
                                x_range=(0,m), y_range=(0,n),
                                min_border_right=64, title = title)
                a[key].image([flipped], x=0, y=0, dw=m, dh=n, color_mapper=color_mapper)
                a[key].add_layout(color_bar, 'right')
                plotlist.append(a[key])
        return gridplot(plotlist, ncols = 3)
    if dim == 2 and learntype == dgbkeys.seisimgtoimgtypestr:
        for t in range(2):
            if (t == 1): 
                data = yslices
                titlepart1 = ' Target '
            nr = cubenr + 1
            titlepart2 = titlelist[0]
            title = titlepart1 + titlepart2 + str(nr)
            flipped = np.flipud(data[0])
            n = flipped.shape[0]
            m = flipped.shape[1]
            maxval = np.amax(flipped)
            minval = np.amin(flipped)
            color_mapper = LinearColorMapper(palette=mypalette, 
                                             low=minval, high=maxval)          
            color_bar = ColorBar(color_mapper=color_mapper, 
                                 ticker=BasicTicker(),
                                 location=(0,0))
            plot = figure(frame_height=int(zoom * n), frame_width=int(zoom * m),
                            x_range=(0,m), y_range=(0,n),
                            min_border_right=64, title = title)
            plot.image([flipped], x=0, y=0, dw=m, dh=n, color_mapper=color_mapper)         
            plot.add_layout(color_bar, 'right')
            plotlist.append(plot)
        return gridplot(plotlist, ncols=1)
    if dim == 3 and learntype == dgbkeys.seisclasstypestr:
        for i in range(dim):
            nr = cubenr + 1
            classnr = yslices
            sectionnr = slicepos[i]
            title = titlelist[i] \
                     + str(sectionnr) + '; output: ' \
                     + str(info[dgbkeys.classnmdictstr][classnr])
            flipped = np.flipud(data[i])
            key = 'plot'+str(i)
            n = flipped.shape[0]
            m = flipped.shape[1]
            maxval = np.amax(flipped)
            minval = np.amin(flipped)
            color_mapper = LinearColorMapper(palette=mypalette, 
                                             low=minval, high=maxval)
            color_bar = ColorBar(color_mapper=color_mapper, 
                                 ticker=BasicTicker(),
                                 location=(0,0))
            a[key] = figure(frame_height=int(zoom * n), frame_width=int(zoom * m),
                            x_range=(0,m), y_range=(0,n),
                             min_border_right=64, title = title)
            a[key].image([flipped], x=0, y=0, dw=m, dh=n, color_mapper=color_mapper )
            a[key].add_layout(color_bar, 'right')
            plotlist.append(a[key])
        return gridplot(plotlist, ncols=len(plotlist))
    if dim == 2 and learntype == dgbkeys.seisclasstypestr:
        nr = cubenr + 1
        classnr = yslices
        title = 'Input direction is: ' + titlelist[0] \
                + '; Example nr. is: '+ str(nr) + ';    '  \
                + 'Output is: ' + str(info[dgbkeys.classnmdictstr][classnr])
        flipped = np.flipud(data[0])
        n = flipped.shape[0]
        m = flipped.shape[1]
        maxval = np.amax(flipped)
        minval = np.amin(flipped)
        color_mapper = LinearColorMapper(palette=mypalette, 
                                         low=minval, high=maxval)          
        color_bar = ColorBar(color_mapper=color_mapper, 
                             ticker=BasicTicker(),
                             location=(0,0))
        plot = figure(frame_height=int(zoom * n), frame_width=int(zoom * m),
                        x_range=(0,m), y_range=(0,n),
                        min_border_right=64, title = title)
        plot.image([flipped], x=0, y=0, dw=m, dh=n, color_mapper=color_mapper )
        plot.add_layout(color_bar, 'right')
        plotlist.append(plot)
        return gridplot(plotlist, ncols=1)
    return None
