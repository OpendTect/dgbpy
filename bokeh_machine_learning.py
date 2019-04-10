#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Jan 2019
#
# _________________________________________________________________________a

import sys
import os
import argparse
import subprocess

from bokeh.layouts import row, column, layout
from bokeh.models import Spacer
from bokeh.models.widgets import (Button, CheckboxGroup, Panel, Select, Slider,
                                  Tabs, TextInput)
from bokeh.plotting import curdoc
from bokeh.server import callbacks
from bokeh.util import logconfig

import odpy.common as odcommon
from odpy.oscommand import getPythonCommand, execCommand, kill, isRunning
import dgbpy.keystr as dgbkeys
from dgbpy import mlapply as dgbmlapply
from dgbpy import dgbkeras, dgbscikit

parser = argparse.ArgumentParser(
            description='Select parameters for machine learning model training')
parser.add_argument( '-v', '--version',
            action='version', version='%(prog)s 1.0' )
parser.add_argument( 'h5file',
            type=argparse.FileType('r'),
            help='HDF5 file containing the training data' )
datagrp = parser.add_argument_group( 'Data' )
datagrp.add_argument( '--dataroot',
            dest='dtectdata', metavar='DIR', nargs=1,
            help='Survey Data Root' )
datagrp.add_argument( '--survey',
            dest='survey', nargs=1,
            help='Survey name' )
odappl = parser.add_argument_group( 'OpendTect application' )
odappl.add_argument( '--dtectexec',
            metavar='DIR', nargs=1,
            help='Path to OpendTect executables' )
odappl.add_argument( '--qtstylesheet',
            metavar='qss', nargs=1,
            type=argparse.FileType('r'),
            help='Qt StyleSheet template' )
loggrp = parser.add_argument_group( 'Logging' )
loggrp.add_argument( '--proclog',
            dest='logfile', metavar='file', nargs='?',
            type=argparse.FileType('a'), default=sys.stdout,
            help='Progress report output' )
loggrp.add_argument( '--syslog',
            dest='sysout', metavar='stdout', nargs='?',
            type=argparse.FileType('a'), default=sys.stdout,
            help='Standard output' )
args = vars(parser.parse_args())
odcommon.initLogging( args )
odcommon.proclog_logger.setLevel( 'DEBUG' )

trainscriptfp = os.path.join(os.path.dirname(__file__),'mlapplyrun.py')

but_width = 80
but_height = 32
but_spacer = 5

traintabnm = 'Training'
paramtabnm = 'Parameters'

trainpanel = Panel(title=traintabnm)
parameterspanel = Panel(title=paramtabnm)
mainpanel = Tabs(tabs=[trainpanel,parameterspanel])

ML_PLFS = []
ML_PLFS.append( dgbkeras.platform )
ML_PLFS.append( dgbscikit.platform )

platformfld = Select(title="Machine learning platform:",options=ML_PLFS)
outputnmfld = TextInput(title='Output model:',value=dgbkeys.modelnm)

def setActiveTab( tabspanelwidget, tabnm ):
  tabs = tabspanelwidget.tabs
  for i in range( len(tabs) ):
    if tabs[i].title == tabnm:
      tabspanelwidget.active = i
      return

def integerListContains( listobj, index ):
  for itm in listobj:
    if itm == index:
      return True
  return False

def doDecimate( fldwidget, index=0 ):
  return integerListContains( fldwidget.active, index )

def doKeras():
  return platformfld.value == dgbkeras.getMLPlatform()

def doScikit():
  return platformfld.value == dgbscikit.getMLPlatform()

def setTrainigTabCB():
  setActiveTab( mainpanel, traintabnm )

def setParsTabCB():
  setActiveTab( mainpanel, paramtabnm )

def getKerasParsGrp():
  dict = dgbkeras.keras_dict
  dodecimatefld = CheckboxGroup( labels=['Decimate input'], active=[] )
  decimatefld = Slider(start=0.1,end=99.9,value=dict['dec']*100, step=0.1,
                title='Decimation (%)')
  iterfld = Slider(start=1,end=100,value=dict['iters'],step=1,
              title='Iterations')
  epochfld = Slider(start=1,end=100,value=dict['epoch'],step=1,
              title='Epochs')
  batchfld = Slider(start=1,end=100,value=dict['batch'],step=1,
            title='Number of Batch')
  patiencefld = Slider(start=1,end=100,value=dict['patience'],step=1,
                title='Patience')
  return (dodecimatefld,decimatefld,iterfld,epochfld,batchfld,patiencefld,{
    'tabname': dgbkeras.getUIMLPlatform(),
    'grp' : column(epochfld,batchfld,patiencefld,dodecimatefld,decimatefld,iterfld)
  })

def getScikitParsGrp():
  dict = dgbscikit.scikit_dict
  nbparfld = Slider(start=1,end=100,value=dict['nb'],step=1,title='Number')
  return (nbparfld,{
    'tabname': dgbscikit.getUIMLPlatform(),
    'grp': column(nbparfld)
  })

def getButtonsGrp():
  runbut = Button(label="Run",button_type="success",
                  width=but_width,height=but_height)
  stopbut = Button(label="Stop",disabled=True,width=but_width,height=but_height)
  return ( runbut, stopbut,
           row(Spacer(width=110),runbut,Spacer(width=but_spacer),stopbut) )

platformparsbut = Button(label=paramtabnm,width=but_width,height=but_height)

(dodecimatefld,decimatefld,iterfld,epochfld,batchfld,patiencefld,kerasparsgrp) = getKerasParsGrp()
(nbparfld,scikitparsgrp) = getScikitParsGrp()

parsgroups = (kerasparsgrp,scikitparsgrp)
parsbackbut = Button(label="Back",width=but_width,height=but_height)

(runbut,stopbut,buttonsgrp) = getButtonsGrp()
trainpanel.child = column( platformfld, platformparsbut, outputnmfld,
                           buttonsgrp )

def mlchgCB( attrnm, old, new):
  selParsGrp( new )

def selParsGrp( platformnm ):
  for platform,parsgroup in zip(ML_PLFS,parsgroups):
    if platform[0] == platformnm:
      curdoc().clear()
      parameterspanel.child = column( parsgroup['grp'], parsbackbut )
      curdoc().add_root(mainpanel)
      return

def decimateCB( widgetactivelist ):
  decimate = integerListContains( widgetactivelist, 0 )
  decimatefld.visible = decimate
  iterfld.visible = decimate

def getParams():
  if doKeras():
    return dgbkeras.getParams( doDecimate(dodecimatefld), decimatefld.value/100,
                               iterfld.value, epochfld.value, batchfld.value,
                               patiencefld.value )
  elif doScikit():
    return dgbscikit.getParams( nbparfld.value )
  return {}

def getProcArgs( platfmnm, pars, outnm ):
  ret = {
    'posargs': [args['h5file'].name],
    'odargs': odcommon.getODArgs( args ),
    'dict': {
      'platform': platfmnm,
      'output': outnm,
      'parameters': pars
    }
  }
  return ret

trainstate = {
  'proc': None,
  'cb': None
}

def trainMonitorCB():
  proc = trainstate['proc']
  if proc == None:
    return
  elif not isRunning(proc):
    trainstate['cb'] = curdoc().remove_periodic_callback( trainstate['cb'] )
    trainstate['proc'] = None
    runbut.disabled = False
    stopbut.disabled = True

def acceptOK():
  runbut.disabled = True
  stopbut.disabled = False
  scriptargs = getProcArgs( platformfld.value, getParams(), outputnmfld.value )
  cmdtorun = getPythonCommand( trainscriptfp, scriptargs['posargs'], \
                               scriptargs['dict'], scriptargs['odargs'] )
  trainstate['proc'] = execCommand( cmdtorun, background=True )
  trainstate['cb'] = curdoc().add_periodic_callback(trainMonitorCB,2000)

def rejectOK():
  proc = trainstate['proc']
  if isRunning(proc):
    trainstate['proc'] = kill( trainstate['proc'] )
    trainstate['cb'] = curdoc().remove_periodic_callback( trainstate['cb'] )
  trainstate['proc'] = None
  runbut.disabled = False
  stopbut.disabled = True

platformfld.on_change('value',mlchgCB)
platformparsbut.on_click(setParsTabCB)
runbut.on_click(acceptOK)
stopbut.on_click(rejectOK)
dodecimatefld.on_click(decimateCB)
parsbackbut.on_click(setTrainigTabCB)

def initWin():
  platformfld.value = ML_PLFS[0][0]
  mlchgCB( 'value', 0, platformfld.value )
  decimateCB( dodecimatefld.active )

initWin()
