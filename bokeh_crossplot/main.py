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

from os import path
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models.widgets import Select, PreText, TextInput
from bokeh.plotting import curdoc
from bokeh.palettes import Viridis
import crossplot_logs
import odpy.wellman as wellman

run_dict = {
     "dir_path" : "/tmp/",
     "file_name" : "One_well_logs.dat",   # input file in pseudo-las format
     "undef" : 1e30, # undefined value in logs
     'wellname': 'Mywell'
}

# Unpack the dictionary of training parameters
dir_path = run_dict['dir_path']
file_name = run_dict['file_name']
undef = run_dict['undef']
wellname = run_dict['wellname']

file_name = path.join( dir_path, file_name )

def prepareData(inputlogfile, inputwellname, undefvalue,
                nrlogplots):
    data = pd.read_csv(file_name, delimiter='\t')
    data = data.replace(to_replace = undef, value = float('NaN'))
    headers = list(data.columns.values)
    return(data, headers)

(data, headers) = prepareData(
        inputlogfile = file_name,
        inputwellname = wellname, 
        undefvalue = undef, 
        nrlogplots = 1)
headers = ['None'] + headers
wells = wellman.getNames()
mindepth = data.iloc[0][0]
maxdepth = data.iloc[-1][0]
SIZES = list(range(6, 25, 1))
COLORS = Viridis[10]
N_SIZES = len(SIZES)
N_COLORS = len(COLORS)
columns = data.columns

def update(attr, old, new):
    (layout.children[0], layout.children[2]) = (
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

def updateLogsCB(attr, old, new):
    headers = ['None'] + wellman.getLogNames(new)
    x.options = headers[1:]
    y.options = headers[1:]
    size.options = headers
    color.options = headers


w = Select(title='Well', value=wells[0], options=wells)
w.on_change('value', updateLogsCB)

x = Select(title='X-Axis', value= headers[2], options= headers[1:])
x.on_change('value', update)

y = Select(title='Y-Axis', value= headers[3], options= headers[1:])
y.on_change('value', update)

size = Select(title='Size', value= headers[0], options= headers)
size.on_change('value', update)

color = Select(title='Color', value= headers[0], options=headers)
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
layout = row(grid, controls, xplot)

curdoc().add_root(layout)
curdoc().title = "Crossplot well logs"
