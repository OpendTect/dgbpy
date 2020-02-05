# -*- coding: utf-8 -*-
"""
============================================================
Log plotting GUI
============================================================

 Author:    Paul de Groot <paul.degroot@dgbes.com>
 Copyright: dGB Beheer BV
 Date:      January 2020
 License:   GPL v3


@author: paul
"""

from os import path
import argparse
import pandas as pd
from bokeh.layouts import layout, column, row
from bokeh.models import Button
from bokeh.models.widgets import (MultiSelect, Panel, Tabs, Select,
                                  TextInput, Div, Slider,
                                  RadioButtonGroup)
from bokeh.plotting import curdoc, figure
import plot_logs
import itertools
import odpy.wellman as wellman
from dgbpy.bokehserver import StartBokehServer, DefineBokehArguments

parser = argparse.ArgumentParser(
            description='Dashboard for well log plotting')
parser.add_argument( '-v', '--version',
            action='version', version='%(prog)s 1.0' )
datagrp = parser.add_argument_group( 'Data' )
datagrp.add_argument( '--dataroot',
            dest='dtectdata', metavar='DIR', nargs=1,
            help='Survey Data Root' )
datagrp.add_argument( '--survey',
            dest='survey', nargs=1,
            help='Survey name' )
datagrp.add_argument( '--well',
            dest='wellid', nargs=1,
            help='Well ID' )

parser = DefineBokehArguments(parser)

args = vars(parser.parse_args())
reload = False

wellid = args['wellid'][0]
wellnm = wellman.getName( wellid, reload, args )

data = pd.DataFrame()
headers = []
loglist = [] # headers with None added and depth removed
logonlylist = [] # headers with depth removed
markers = []
undef = 1e30
mindepth = 0
maxdepth = 0
nolog = 'None'


def readLogs( wellnm, undefvalue ):
    lognames = wellman.getLogNames( wellnm, reload, args )
    logdata = pd.DataFrame()
    if not lognames:
      return (lognames,logdata)

    for nm in lognames:
      ld = wellman.getLog( wellnm, nm, reload, args )
      lddf = pd.DataFrame( ld ).transpose()
      lddf.columns = ['MD',nm]
      if ( logdata.empty ):
        logdata = lddf
      else:
        logdata = pd.merge( logdata, lddf, on='MD', how='outer', sort=True )

    logdata = logdata.replace(to_replace=undefvalue, value=float('NaN'))
    lognames = ['MD'] + lognames

    if ( logdata.empty ):
      mindepth = 0
      maxdepth = 500
    else:
      mindepth = logdata.iloc[0][0]
      maxdepth = logdata.iloc[-1][0]

    return (lognames,logdata)

def updateLogList(val, lst, option = 0): # option = 1 add None
  if (option == 1):
    lst = ['None'] + lst
  return [x for x in lst if x != val]


(headers,data) = readLogs( wellnm, undef )
depthlogstr = headers[0]
loglist = updateLogList(depthlogstr, headers, option = 1)
logonlylist = updateLogList(depthlogstr, headers, option = 0)


def logplot_app(doc):

  plot = figure()
  rowofcolcurves = row()
  nrlogplots = 1 # first time only one log plot

  xlogscales = ['linear', 'log']
  shadingtypes = ['None', 'Left shading color', 'Left palette',
                'Right shading color', 'Right palette',
                'Column wide palette', 'Difference 2 logs shading color']
  logwidth = '250'
  logheight = '800'
  columnwidth = 300
  colorlist = ['darkred', 'navy', 'orange', 'darkgreen', 'black',
             'cyan', 'darkslategrey', 'gold', 'lime', 'magenta']
  colors = itertools.cycle(colorlist)
  palettelist = ['Inferno', 'Magma', 'Plasma', 'Viridis']
  linetypelist = ['solid', 'dashed', 'dotted', 'dashdot', 'dotdash']
  types = itertools.cycle(linetypelist)


  def update():
    layoutplots.children[1] = plot_logs.create_gridplot(alldata)

  def getMinMaxLogs(loglist):
    minmaxlogs = []
    for i in range(len(loglist)):
      minvalue = data[loglist[i]].dropna().min()
      maxvalue = data[loglist[i]].dropna().max()
      extension = (maxvalue - minvalue) / 20
      minvalue = minvalue - extension
      maxvalue =  maxvalue + extension
      addrow = [loglist[i], minvalue, maxvalue]
      minmaxlogs.append(addrow)
    return(minmaxlogs)

  minmaxlogs = getMinMaxLogs(logonlylist)
  textgeneral = Div(text="""<b>General</b>""", width=columnwidth)
  plotbutton = Button(label='Update Plot', button_type='success')
  plotbutton.on_click(update)
#  markernames = ['None', 'All'] + list(markers['Name'])
  markernames = ['None', 'All']
  plotmarkers = MultiSelect(title='Plot markers:',
		    value=[markernames[0]],
		    options=markernames[:])
  inputwellname = TextInput(title='Well name', value=wellnm)
  maxnrplots = 5
  nrlogplots = Slider(start=1, end=maxnrplots, value=1, step=1,
			title='Number of log plots')
  logheight = TextInput(title='log plot height', value= logheight)
  logwidth = TextInput(title='log plot width', value= logwidth)
  textplot1 = Div(text="""<b>Plot 1</b>""", width=columnwidth)
  textplot2 = Div(text="""<b>Plot 2</b>""", width=columnwidth)
  textplot3 = Div(text="""<b>Plot 3</b>""", width=columnwidth)
  textplot4 = Div(text="""<b>Plot 4</b>""", width=columnwidth)
  textplot5 = Div(text="""<b>Plot 5</b>""", width=columnwidth)
  rowdict = {}
  coldict = {}
  for i in range(0,9):
    r = 'r' + str(i)
    rowdict[r] = []

  for i in range(0,5):
    rowdict['r0'].append(RadioButtonGroup(
		    labels=['linear', 'logarithmic'],
		    active=0))
    rowdict['r1'].append(MultiSelect(title='Curve plots:',
		    value=[loglist[1]],
		    options=loglist[:]
		    ))
    rowdict['r2'].append(Select(title='Shading type:',
		    value=shadingtypes[0],
		    options=shadingtypes[:]
		    ))
    rowdict['r3'].append(MultiSelect(title='Shading log(s):',
		    value=[loglist[0]],
		    options=loglist[:]
		    ))

  for i in range(0,5):
    c = 'c' + str(i)
    coldict[c] = column([rowdict['r0'][i], rowdict['r1'][i],
			      rowdict['r2'][i], rowdict['r3'][i]
			      ])
  rowofcolcurves = row(coldict['c0'], coldict['c1'],
		   coldict['c2'], coldict['c3'],
		   coldict['c4'])

  #tab Parameters
  textparams = Div(text="""<b>Ranges, Scaling & Colors</b>""",
		     width=columnwidth)
  plotbutton1 = Button(label='Update Plot',
			  button_type='success')
  plotbutton1.on_click(update)
  inputmindepth = TextInput(title='Minimum depth', value=str(mindepth))
  inputmaxdepth = TextInput(title='Maximum depth', value=str(maxdepth))
  depthticks = TextInput(title='Depth ticks', value="50")
  depthminorticks = TextInput(title='Depth minor ticks', value="10")
  shadingcolor = Select(title='Shading color',
		     value = colorlist[0],
		     options=colorlist[:]
		     )
  colorfillpalette = Select(title='Color fill palette:',
		     value = palettelist[0],
		     options=palettelist[:]
		     )
  curvetext = (Div(text = """<b>Curve</b>""", width=columnwidth))
  minscaletext = (Div(text = """<b>Min. scale</b>""", width=columnwidth))
  maxscaletext = (Div(text = """<b>Max. scale</b>""", width=columnwidth))
  linetypetext = (Div(text = """<b>Line style</b>""", width=columnwidth))
  linecolortext = (Div(text = """<b>Line color</b>""", width=columnwidth))
  nrcurves = len(logonlylist)

  lst0 = []
  lst1 = []
  lst2 = []
  lst3 = []
  lst4 = []
  colorcycle = []
  for n, color in zip(range(nrcurves), colors ):
    colorcycle.append(color)

  for n, color in zip(range(nrcurves), colorcycle ):
    lst0.append(TextInput(value=logonlylist[n]))
    lst1.append(TextInput(value =
		    str(round(minmaxlogs[n][1], 1))))
    lst2.append(TextInput(value =
		    str(round(minmaxlogs[n][2], 1))))
    linestyle = Select(value = linetypelist[0],
			 options= linetypelist[:])
    lst3.append(linestyle)
    linecolor = Select(value = color,
			 options=colorcycle[:])
    lst4.append(linecolor)

  col0 = column(lst0)
  col1 = column(lst1)
  col2 = column(lst2)
  col3 = column(lst3)
  col4 = column(lst4)
  rowofcol = row(col0, col1, col2, col3, col4)

  alldata = { 'data': data,
	       'headers': headers,
	       'markers': markers,
	       'plotmarkers': plotmarkers,
	       'wellname': wellnm,
	       'nrcurves' : nrcurves,
	       'rowofcolcurves' : rowofcolcurves,
	       'shadingcolor' : shadingcolor,
	       'colorfillpalette' : colorfillpalette,
	       'nrlogplots' : nrlogplots,
	       'logwidth': logwidth,
	       'logheight': logheight,
	       'xlogscales' : xlogscales,
	       'firstdepth' : mindepth,
	       'lastdepth' : maxdepth,
	       'depthticks' : depthticks,
	       'rowofcol'  : rowofcol,
	       'depthminorticks' : depthminorticks
	       }

  layoutcurves = layout()
  layoutplots = layout()
  # put the button and plot in a layout and add to the document
  layoutcurves = layout( children = [ [textgeneral, plotbutton],
					[inputwellname,plotmarkers],
					[nrlogplots,logwidth,logheight],
					[textplot1, textplot2, textplot3,
					 textplot4, textplot5],
					[[rowofcolcurves.children]]
					] )
  layoutparams = layout( children = [ [textparams, plotbutton1],
					[inputmindepth, inputmaxdepth,
					 depthticks, depthminorticks],
					[shadingcolor, colorfillpalette],
					[curvetext, minscaletext, maxscaletext,
					 linetypetext, linecolortext],
					[[rowofcol.children]]
					] )

  wellnmfld = TextInput()
  wellnmfld.value = wellnm
  grid = plot_logs.create_gridplot(alldata)

  layoutplots = layout( children = [ [wellnmfld], [grid] ] )

  tab1 = Panel(child=layoutcurves, title='Curves')
  tab2 = Panel(child=layoutparams, title='Parameters')
  tab3 = Panel(child=layoutplots, title='Plots')
  tabs = Tabs(tabs=[ tab1, tab2, tab3 ] )

  doc.add_root(tabs)
  doc.title = "Plot well logs"

#logplot_app( curdoc() )
StartBokehServer({'/': logplot_app}, args)
