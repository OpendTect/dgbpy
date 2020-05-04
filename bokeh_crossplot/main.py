"""
============================================================
Log crossplot GUI
============================================================

 Author:    Paul de Groot <paul.degroot@dgbes.com>
 Copyright: dGB Beheer BV
 Date:      March 2019
 License:   GPL v3
 Credits:   Based in parts on Crossfilter example of www.bokeh.org.

@author: paul
"""

import sys
import argparse
from dgbpy.bokehserver import StartBokehServer, DefineBokehArguments

parser = argparse.ArgumentParser(
            description='Dashboard for well log crossplotting')
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
datagrp.add_argument( '--wlfile',
            dest='welllogfile', nargs=1,
            help='Log file' )
loggrp = parser.add_argument_group( 'Logging' )
loggrp.add_argument( '--proclog',
            dest='logfile', metavar='file', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='Progress report output' )
loggrp.add_argument( '--syslog',
            dest='sysout', metavar='stdout', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='Standard output' )

parser = DefineBokehArguments(parser)

args = vars(parser.parse_args())

import odpy.common as odcommon
odcommon.initLogging( args )

from os.path import dirname,join
import numpy as np
import pandas as pd 
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import MISSING_RENDERERS
from bokeh.layouts import column, row
from bokeh import themes
from bokeh.models.widgets import Select, PreText, TextInput
from bokeh.palettes import Viridis

import odpy.wellman as wellman
import crossplot_logs

silence(MISSING_RENDERERS, True)
reload = False

wellid = args['wellid'][0]
wellnm = wellman.getName( wellid, reload, args )
filenm = args['welllogfile'][0]

data = pd.DataFrame
undef = 1e30
mindepth = 0
maxdepth = 0
headers = []
nolog = 'None'
logx = nolog
logy = nolog
xoptions = [nolog]
yoptions = [nolog]
logcol = nolog
logsz = nolog

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
    return (lognames,logdata)


def readFromFile( filenm, undefval ):
    logdata = pd.read_csv( filenm, delimiter='\t' )
    logdata = logdata.replace( to_replace=undefval, value=float('NaN') )
    headers = list( logdata.columns.values )
    return (headers,logdata)


def prepareForPlot( wellnm, filenm ):
    (lognames,logdata) = readFromFile( filenm, undef )
#    (lognames,logdata) = readLogs( wellnm, undef )
    global data, headers, mindepth, maxdepth
    global logx, logy, logcol, logsz, xoptions, yoptions
    data = logdata
    headers = [nolog] + lognames
    logx = logy = logcol = logsz = nolog
    if ( len(lognames) == 2 ):
      logx = lognames[0]
      logy = lognames[1]
    else:
      if ( len(lognames) > 1 ):
        logx = lognames[1]
      if ( len(lognames) > 2 ):
        logy = lognames[2]

    if not lognames:
        xoptions = yoptions = [nolog]
    else:
        xoptions = yoptions = lognames

    if ( data.empty ):
        mindepth = 0
        maxdepth = 500
    else:
        mindepth = data.iloc[0][0]
        maxdepth = data.iloc[-1][0]


prepareForPlot( wellnm, filenm )
  
def crossplot_app(doc):
  SIZES = list(range(6, 25, 1))
  COLORS = Viridis[10]
  N_SIZES = len(SIZES)
  N_COLORS = len(COLORS)
  columns = data.columns

  def update(attr, old, new):
    (layout.children[1], layout.children[2]) = (
                    crossplot_logs.create_plots(alldata) )

  def minDepthChangeCB(attr, old, new):
    if (float(new) <= data.iloc[0][0]):
        alldata['mindepth'] = data.iloc[0][0]
    else:
        alldata['mindepth'] = float(new)
    update(attr, old, alldata['mindepth'])

  def maxDepthChangeCB(attr, old, new):
    if (float(new) >= data.iloc[-1][0]):
        alldata['maxdepth'] = data.iloc[-1][0]
    else:
        alldata['maxdepth'] = float(new)
    update(attr, old, alldata['maxdepth'])

  w = TextInput(title='Well', value=wellnm )
  w.disabled = True

  x = Select(title='X-Axis', value=logx, options=xoptions )
  x.on_change('value', update)

  y = Select(title='Y-Axis', value=logy, options=yoptions )
  y.on_change('value', update)

  size = Select(title='Size', value=logsz, options= headers)
  size.on_change('value', update)

  color = Select(title='Color', value=logcol, options=headers)
  color.on_change('value', update)

  plottype = Select(title='Cross plot type:', value='Bubbles',
                 options=['Bubbles', 'Scatter + Regression'])
  plottype.on_change('value', update)

  inputmindepth = TextInput(title='Minimum depth', value=str(mindepth))
  inputmindepth.on_change('value', minDepthChangeCB)

  inputmaxdepth = TextInput(title='Maximum depth', value=str(maxdepth))
  inputmaxdepth.on_change('value', maxDepthChangeCB)

  controls = column([w, x, y, color, size, plottype,
                   inputmindepth, inputmaxdepth], width=250)

  stats = PreText(text='', width=800)

  alldata = { 'data': data,
              'headers': headers,
              'x': x,
              'y': y,
              'size': size,
              'color': color,
              'plottype': plottype,
              'stats' : stats,
              'N_SIZES': N_SIZES,
              'SIZES': SIZES,
              'N_COLORS': N_COLORS,
              'COLORS': COLORS,
              'mindepth': mindepth,
              'maxdepth': maxdepth,
           }
  (grid, xplot) = crossplot_logs.create_plots(alldata)
  layout = row( controls, grid, xplot )

  doc.add_root(layout)
#  doc.theme = themes.Theme( filename=join(dirname(__file__),'theme.yaml') )
  doc.title = "Crossplot well logs"

StartBokehServer({'/': crossplot_app}, args)

