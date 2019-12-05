# -*- coding: utf-8 -*-
"""
============================================================
Log plotting GUI
============================================================

 Author:    Paul de Groot <paul.degroot@dgbes.com>
 Copyright: dGB Beheer BV
 Date:      March 2019
 License:   GPL v3


@author: paul
"""

from os import path
import pandas as pd
from bokeh.layouts import layout, column, row
from bokeh.models import Button
from bokeh.models.widgets import (MultiSelect, Panel, Tabs, Select, 
                                  TextInput, Div, Slider,
                                  RadioButtonGroup)
from bokeh.plotting import curdoc, figure
import plot_logs
import itertools  

#import plot_logs as plot_logs

run_dict = {
#     "dir_path" : "c:/dev/Python_Scripts/Bokeh/", # directory path for in- and output
     "dir_path" : "/dsk/d101/nanne/Python/Paul/Bokeh/",
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
data = pd.DataFrame()
depthdata = pd.DataFrame()
headers = []
loglist = [] # headers with None added and depth removed
logonlylist = [] # headers with depth removed
plot = figure()
logcol = 1 # default log for the first plot
xlogscales = ['linear', 'log']
logwidth = '200'
logheight = '1000'
depthname = []
mindepth = str()
maxdepth = str()
colorlist = ['darkred', 'navy', 'orange', 'darkgreen', 'black',
             'cyan', 'darkslategrey', 'gold', 'lime', 'magenta']
colors = itertools.cycle(colorlist)
palettelist = ['Inferno', 'Magma', 'Plasma', 'Viridis']
linetypelist = ['solid', 'dashed', 'dotted', 'dashdot', 'dotdash']
types = itertools.cycle(linetypelist)
#param_col = column(height=10, width=100, sizing_mode = 'scale_both')
alldata = { }
layoutcurves = layout()
layoutplots = layout()
wellnmfld = TextInput()
wellnmfld.value = wellname
rowofcolcurves = row()
nrlogplots = Slider()

# create a callback that will add a number in a random location
def update():
#    alldata['depthdata'] = seldepthlog.value
    layoutplots.children[1] = plot_logs.create_gridplot(alldata)
        
def inputFileCB(attr, oldtype, newtype):
    print( 'Not implemented' )

def wellNameCB(attrname, old, new):
    wellnmfld.value = inputwellname.value

def depthLogCB(attrname, old, new):
    update()
   
def updateLogList(val, lst, option = 0): # option = 1 add None
    if (option == 1):
        lst = ['None'] + lst
    return [x for x in lst if x != val]

def prepareData(inputlogfile, inputwellname, undefvalue, 
                nrlogplots, logheight, logwidth):
    data = pd.read_csv(file_name, delimiter='\t')
    data = data.replace(to_replace = undef, value = float('NaN'))
    headers = list(data.columns.values)
    return(data, headers)

def getMinMaxLogs(logonlylist):
    minmaxlogs = []
    for i in range(len(logonlylist)):
        minvalue = data[logonlylist[i]].dropna().min()
        maxvalue = data[logonlylist[i]].dropna().max()
        extension = (maxvalue - minvalue) / 20
        minvalue = minvalue - extension 
        maxvalue =  maxvalue + extension 
        addrow = [logonlylist[i], minvalue, maxvalue]
        minmaxlogs.append(addrow)
    return(minmaxlogs)
    
(data, headers) = prepareData(
        inputlogfile = file_name,
        inputwellname = wellname, 
        undefvalue = undef, 
        nrlogplots = 1, 
        logheight = logheight, 
        logwidth = logwidth)

depthlogstr = headers[0] # assuming first column is depth
loglist = updateLogList(depthlogstr, headers, option = 1)
logonlylist = updateLogList(depthlogstr, headers, option = 0)
minmaxlogs = getMinMaxLogs(logonlylist)
seldepthlog = Select(title='Depth column:',
                 value = headers[0],
                 options=headers[:]
                 )
seldepthlog.on_change('value', depthLogCB)
textgeneral = Div(text="""<b>General</b>""")
plotbutton = Button(label='Update Plot',
                      button_type='success')
plotbutton.on_click(update)

inputlogfile = TextInput(title='Input log file', value=file_name)
inputwellname = TextInput(title='Well name', value=wellname)
inputwellname.on_change('value', wellNameCB)
maxnrplots = 5 
nrlogplots = Slider(start=1, end=maxnrplots, value=1, step=1, 
                    title='Number of log plots')
undefvalue = TextInput(title='Undefined value', value=str(undef))
logheight = TextInput(title='log plot height', value= logheight)
logwidth = TextInput(title='log plot width', value= logwidth)
textplot1 = Div(text="""<b>Plot 1</b>""")
textplot2 = Div(text="""<b>Plot 2</b>""")
textplot3 = Div(text="""<b>Plot 3</b>""")
textplot4 = Div(text="""<b>Plot 4</b>""")
textplot5 = Div(text="""<b>Plot 5</b>""")
rowdict = {}
coldict = {}
for i in range(0,7):
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
    rowdict['r2'].append(Select(title='Shading left:',
                value = loglist[0],
                options=loglist[:]
                ))
    rowdict['r3'].append(Select(title='Shading right:',
                 value = loglist[0],
                 options=loglist[:]
                 ))
    rowdict['r4'].append(Select(title='Full width (litho-facies):',
                 value = loglist[0],
                 options=loglist[:]
                 ))
    rowdict['r5'].append(Select(title='Shading band left:',
                 value = loglist[0],
                 options=loglist[:]    
                 ))
    rowdict['r6'].append(Select(title='Shading band right:',
                 value = loglist[0],
                 options=loglist[:]
                 ))

for i in range(0,5):
    c = 'c' + str(i)
    coldict[c] = column([rowdict['r0'][i], rowdict['r1'][i],
                          rowdict['r2'][i], rowdict['r3'][i],
                          rowdict['r4'][i], rowdict['r5'][i],
                          rowdict['r6'][i]
                          ])    
rowofcolcurves = row(coldict['c0'], coldict['c1'], 
               coldict['c2'], coldict['c3'],
               coldict['c4'])

#tab Parameters
textparams = Div(text="""<b>Ranges, Scaling & Colors</b>""")
plotbutton1 = Button(label='Update Plot',
                      button_type='success')
plotbutton1.on_click(update)
mindepth = TextInput(title='Minimum depth', value=mindepth)
maxdepth = TextInput(title='Maximum depth', value=maxdepth)
depthticks = TextInput(title='Depth ticks', value="50")
depthminorticks = TextInput(title='Depth minor ticks', value="10")
shadingcolor = Select(title='Shading color',
                 value = colorlist[0],
                 options=colorlist[:]
                 )
fullwidthpalette = Select(title='Full width (litho-facies) palette:',
                 value = palettelist[0],
                 options=palettelist[:]
                 )
curvetext = (Div(text = """<b>Curve</b>"""))
minscaletext = (Div(text = """<b>Min. scale</b>"""))
maxscaletext = (Div(text = """<b>Max. scale</b>"""))
linetypetext = (Div(text = """<b>Line style</b>"""))
linecolortext = (Div(text = """<b>Line color</b>"""))
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
           'wellname': wellname,
           'nrcurves' : nrcurves,
           'rowofcolcurves' : rowofcolcurves,
           'shadingcolor' : shadingcolor,
           'fullwidthpalette' : fullwidthpalette,
           'nrlogplots' : nrlogplots,
           'logwidth': logwidth,
           'logheight': logheight,
           'seldepthlog': seldepthlog,
           'xlogscales' : xlogscales,
           'mindepth' : mindepth,
           'maxdepth' : maxdepth,
           'depthticks' : depthticks,
           'rowofcol'  : rowofcol,
           'depthminorticks' : depthminorticks
           }

grid = plot_logs.create_gridplot(alldata)
  
# put the button and plot in a layout and add to the document
layoutcurves = layout( children = [ [textgeneral, plotbutton],
                                    [inputlogfile, inputwellname,
                                     seldepthlog, undefvalue],
                                    [nrlogplots,
                                    logwidth, logheight],
                                    [textplot1, textplot2, textplot3,
                                     textplot4, textplot5],
                                    [[rowofcolcurves.children]]
                                    ] )
layoutparams = layout( children = [ [textparams, plotbutton1],
                                    [mindepth, maxdepth, 
                                     depthticks, depthminorticks],
                                    [shadingcolor, fullwidthpalette],
                                    [curvetext, minscaletext, maxscaletext,
                                     linetypetext, linecolortext],
                                    [[rowofcol.children]]
                                    ] )
layoutplots = layout( children =  [ [wellnmfld], 
                                    [grid]
                                    ] )

tab1 = Panel(child=layoutcurves, title='Curves')
tab2 = Panel(child=layoutparams, title='Parameters')
tab3 = Panel(child=layoutplots, title='Plots')
tabs = Tabs(tabs=[ tab1, tab2, tab3 ] )

curdoc().add_root(tabs)
