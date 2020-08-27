# -*- coding: utf-8 -*-
"""
============================================================
Cubelets Plotting GUI
============================================================

 Author:    Paul de Groot, Arnaud Huck
 Copyright: dGB Beheer BV
 Date:      May 2020
 License:   GPL v3


@author: paul
"""

from functools import partial
import numpy as np

from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Select, Slider, TextInput
from bokeh.server.server import Server

from odpy.common import log_msg
from dgbpy import hdf5 as dgbhdf5
from dgbpy import mlio as dgbmlio
from dgbpy import keystr as dgbkeys

from plot_examples import plot_slices

exfilenm = ''

palettelist = ['Inferno256', 'Viridis256', 'Turbo256', 'Greys256']

def re_plot(plotparams, gridcol):
    gridcol.children[0] = plot_slices(plotparams)

def survChgCB(plotparams, gridcol, collslider, cubeslider, attr, old, new):
    newsurveynm = new
    plotparams.update({dgbkeys.surveydictstr: newsurveynm})
    info = plotparams[dgbkeys.infodictstr]
    dsets = info[dgbkeys.datasetdictstr][newsurveynm]
    oldcollnm = None
    if dgbkeys.collectdictstr in plotparams:
        oldcollnm = plotparams[dgbkeys.collectdictstr]
    collnms = dsets.keys()
    if oldcollnm != None and oldcollnm in collnms:
        newcollnm = oldcollnm
    else:
        newcollnm = next(iter(collnms))
    collslider.options = list(collnms)
    collslider.value = newcollnm
    collslider.disabled = len(collnms) < 2
    collslider.on_change('value', partial(collChgCB,plotparams,gridcol,cubeslider))
    collChgCB(  plotparams, gridcol, cubeslider, 'value', oldcollnm, newcollnm )
    
def collChgCB(plotparams, gridcol, cubeslider, attr, old, new):
    collnm = new
    plotparams.update({dgbkeys.collectdictstr: collnm})
    surveynm = plotparams[dgbkeys.surveydictstr]
    info = plotparams[dgbkeys.infodictstr]
    maxnrcubes = len(info[dgbkeys.datasetdictstr][surveynm][collnm])
    oldnrcube = cubeslider.value
    cubeslider.end = maxnrcubes
    newnrcube = oldnrcube
    if oldnrcube > maxnrcubes:
        newnrcube = maxnrcubes
        cubeslider.value = newnrcube
    else:
        loadCubeCB( plotparams, gridcol, 'value', oldnrcube, newnrcube )
    
def loadCubeCB(plotparams, gridcol, attr, old, new):
    plotparams['cubeidx'] = new-1
    re_plot(plotparams, gridcol)
    
def attribChgCB(plotparams, gridcol, attr, old, new):
    plotparams['attribidx'] = new-1
    re_plot(plotparams, gridcol)

def inlSliceCB(plotparams, gridcol, attr, old, new):
    plotparams['slicepos'][0] = new-1
    re_plot(plotparams, gridcol)

def crlSliceCB(plotparams, gridcol, attr, old, new):
    plotparams['slicepos'][1] = new-1
    re_plot(plotparams, gridcol)

def zSliceCB(plotparams, gridcol, attr, old, new):
    plotparams['slicepos'][-1] = new-1
    re_plot(plotparams, gridcol)

def colorPaletteCB(plotparams, gridcol, attr, old, new):
    plotparams['palette'] = new
    re_plot(plotparams, gridcol)

def zoomFactorCB(plotparams, gridcol, attr, old, new):
    plotparams['zoom'] = new
    re_plot(plotparams, gridcol)

def exampleplot_app(doc):
    # initialize parameters
    info = dgbmlio.getInfo( exfilenm )
    surveys = list(info[dgbkeys.exampledictstr].keys())
    titlelist = ['inline-slice ','crossline-slice ', 'z-slice ']
    nrinputs = dgbhdf5.getNrAttribs(info)
    shape = info[dgbkeys.inpshapedictstr]
    dim = len(shape)
    if shape[0] == 1:
        titlelist = ['inline ']
        dim = 2
    elif shape[1] == 1:
        titlelist = ['crossline ']
        dim = 2
    elif shape[-1] == 1:
        titlelist = ['z-slice ']
        dim = 2
    initpos = np.divide(shape,2).astype(np.int).tolist()
    defzoom = np.floor( 200/np.max(shape) )
    if defzoom < 1:
        defzoom = 1
    plotparams = {
        dgbkeys.infodictstr: info,
        'palette': palettelist[-1],
        'nrdims': dim,
        'slicepos': initpos,
        'cubeidx': 0,
        'attribidx': 0,
        'zoom': defzoom,
        'titles': titlelist
    }

    # Add controls
    survslider = Select(title='Survey',options=surveys, value=surveys[0],
                        disabled=len(surveys)<2)
    collslider = Select(title='Collection')
    attribslider = Slider(start=1, end=2, value=1,
                          visible=nrinputs>1,title='Attribute')
    if attribslider.visible:
        attribslider.end = nrinputs
    inlslider = None
    crlslider = None
    zslider = None
    controls = None
    filenametxt = TextInput(title='File Name', value=exfilenm)
    possliders = None
    cubeslider = Slider(start=1, end=2, value=1, title='Cubelet Number')
    if dim == 3:
        inlslider = Slider(start=1, end=shape[0], value=initpos[0], title='Inline Number')
        crlslider = Slider(start=1, end=shape[1], value=initpos[1], title='Crossline Number')
        zslider = Slider(start=1, end=shape[-1], value=initpos[-1], title='Z-slice Number')
        possliders = (cubeslider,inlslider,crlslider,zslider)
    elif dim == 2:
        cubeslider.title = 'Image Number'
        possliders = (cubeslider,)

    colorpalette = Select(title='Color palette:',
                  value = plotparams['palette'],
                  options=palettelist
                  )
    zoomslider = Slider(start=1, end=20, value=plotparams['zoom'], title='Zoom factor')
    controls = column([filenametxt, survslider, collslider, attribslider,
                       *possliders,
                       colorpalette,zoomslider], width=250)
        
    gridcol = column( gridplot([]) )
    layoutplots = row(controls,gridcol)
    
    survslider.on_change('value', partial(survChgCB,plotparams,gridcol,collslider,cubeslider))
    attribslider.on_change( 'value', partial(attribChgCB,plotparams,gridcol))
    cubeslider.on_change('value_throttled', partial(loadCubeCB,plotparams,gridcol))    
    if dim == 3:
        inlslider.on_change('value_throttled', partial(inlSliceCB,plotparams,gridcol))
        crlslider.on_change('value_throttled', partial(crlSliceCB,plotparams,gridcol))
        zslider.on_change('value_throttled', partial(zSliceCB,plotparams,gridcol))
    colorpalette.on_change('value', partial(colorPaletteCB,plotparams,gridcol))
    zoomslider.on_change('value_throttled', partial(zoomFactorCB,plotparams,gridcol))

    doc.add_root(layoutplots)
    doc.title = 'Examine Machine Learning examples'
    survChgCB( plotparams, gridcol, collslider, cubeslider, 'value', '', survslider.options[0] )

def main():
  server = Server({'/' : exampleplot_app})
  server.start()
  server.io_loop.add_callback(server.show, "/")
  server.io_loop.start()

if __name__ == "__main__":
  main()
