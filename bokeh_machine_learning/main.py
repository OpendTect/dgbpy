#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Jan 2019
#
# _________________________________________________________________________

import sys
from os import path
import argparse
import psutil
from functools import partial

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models.widgets import Panel, Select, Tabs, TextInput

import odpy.common as odcommon
from odpy.oscommand import (getPythonCommand, execCommand, kill,
                            isRunning, pauseProcess, resumeProcess)
import dgbpy.keystr as dgbkeys
from dgbpy import mlapply as dgbmlapply
from dgbpy import uibokeh, uikeras, uisklearn
from dgbpy import mlio as dgbmlio

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
traingrp = parser.add_argument_group( 'Training' )
traingrp.add_argument( '--modelfnm',
            dest='model', nargs=1,
            help='Input model file name' )
traingrp.add_argument( '--mldir',
            dest='mldir', nargs=1,
            help='Machine Learning Logging Base Directory' )
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

examplefilenm =  args['h5file'].name
trainscriptfp = path.join(path.dirname(path.dirname(__file__)),'mlapplyrun.py')

traintabnm = 'Training'
paramtabnm = 'Parameters'

trainpanel = Panel(title=traintabnm)
parameterspanel = Panel(title=paramtabnm)
mainpanel = Tabs(tabs=[trainpanel,parameterspanel])

ML_PLFS = []
ML_PLFS.append( uikeras.getPlatformNm(True) )
ML_PLFS.append( uisklearn.getPlatformNm(True) )

platformfld = Select(title="Machine learning platform:",options=ML_PLFS)
platformparsbut = uibokeh.getButton(paramtabnm,\
    callback_fn=partial(uibokeh.setTabFromButton,panelnm=mainpanel,tabnm=paramtabnm))
outputnmfld = TextInput(title='Output model:')

info = dgbmlio.getInfo( examplefilenm )
keraspars = uikeras.getUiPars( info['estimatedsize'] )
sklearnpars = uisklearn.getUiPars( info['classification'] )
parsgroups = (keraspars,sklearnpars)
parsbackbut = uibokeh.getButton('Back',\
    callback_fn=partial(uibokeh.setTabFromButton,panelnm=mainpanel,tabnm=traintabnm))

def mlchgCB( attrnm, old, new):
  selParsGrp( new )

def nameChgCB( attrnm, old, new):
  if len(new) < 1:
    return

  curbg = outputnmfld.background
  (exists,sametrl,sameformat,sametyp) = \
                  dgbmlio.modelNameExists( new,info['type'], \
                  args=args,reload=False)
  if dgbmlio.modelNameIsFree(new,info['type'],args=args,reload=False):
    if exists and sametrl and sameformat and sametyp != None and sametyp:
      outputnmfld.background = '#FFFF00'
    else:
      outputnmfld.background = None
  else:
    outputnmfld.background = '#FF0000'
  outputnmfld.value = new
  if outputnmfld.background == curbg:
    return
  curdoc().clear()
  curdoc().add_root(mainpanel)

def getParsGrp( platformnm ):
  for platform,parsgroup in zip(ML_PLFS,parsgroups):
    if platform[0] == platformnm:
      return parsgroup['grp']
  return None

def selParsGrp( platformnm ):
  parsgrp = getParsGrp( platformnm )
  if parsgrp == None:
    return
  curdoc().clear()
  parameterspanel.child = column( parsgrp, parsbackbut )
  curdoc().add_root(mainpanel)

def getUiParams():
  parsgrp = getParsGrp( platformfld.value )
  if platformfld.value == uikeras.getPlatformNm():
    return uikeras.getUiParams( keraspars )
  elif platformfld.value == uisklearn.getPlatformNm():
    return uisklearn.getUiParams( sklearnpars )
  return {}

def getProcArgs( platfmnm, pars, outnm ):
  traintype = dgbmlapply.TrainType.New
  ret = {
    'posargs': [examplefilenm],
    'odargs': odcommon.getODArgs( args ),
    'dict': {
      'platform': platfmnm,
      'parameters': pars,
      'output': outnm
    }
  }
  dict = ret['dict']
  if 'model' in args:
    model = args['model']
    if model != None and len(model)>0:
      dict.update({'model': model[0]})
      traintype = dgbmlapply.TrainType.Resume #TODO: from GUI
  if 'mldir' in args:
    mldir = args['mldir']
    if mldir != None and len(mldir)>0:
      dict.update({'logdir': mldir[0]})
  dict.update({dgbkeys.typedictstr: traintype.name})
  return ret

def doRun( cb = None ):
  modelnm = outputnmfld.value
  canwrite = dgbmlio.modelNameIsFree(modelnm,info['type'],args=args,reload=True)
  if not canwrite:
    odcommon.log_msg( 'Output model is not writable, please provide another name' )
    #TODO: replace with netpacket
    return False
  
  scriptargs = getProcArgs( platformfld.value, getUiParams(), \
                            modelnm )
  cmdtorun = getPythonCommand( trainscriptfp, scriptargs['posargs'], \
                               scriptargs['dict'], scriptargs['odargs'] )
  return execCommand( cmdtorun, background=True )

def doAbort( proc ):
  if isRunning(proc):
    proc = kill( proc )
  return None

def doPause( proc ):
  pauseProcess( proc )
  return proc

def doResume( proc ):
  resumeProcess( proc )
  return proc

def trainMonitorCB( proc ):
  if proc == None or isRunning(proc):
    return True
  try:
    stat = proc.status()
  except psutil.NoSuchProcess:
    if not odcommon.batchIsFinished( odcommon.get_log_file() ):
      odcommon.log_msg( '\nProcess is no longer running (crashed or terminated).' )
      odcommon.log_msg( 'See OpendTect log file for more details (if available).' )
    return False
  return True

platformfld.on_change('value',mlchgCB)
outputnmfld.on_change('value_input',nameChgCB)
buttonsgrp = uibokeh.getRunButtonsBar( doRun, doAbort, doPause, doResume, trainMonitorCB )
trainpanel.child = column( platformfld, platformparsbut, outputnmfld, buttonsgrp )

def initWin():
  platformfld.value = ML_PLFS[0][0]
  mlchgCB( 'value', 0, platformfld.value )
  nameChgCB( 'value', 0, dgbkeys.modelnm )
  curdoc().title = 'Machine Learning'

initWin()
