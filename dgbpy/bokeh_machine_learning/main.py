#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Jan 2019
#
# _________________________________________________________________________

import os
import sys
import argparse
from dgbpy.bokehserver import StartBokehServer, DefineBokehArguments

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
            type=argparse.FileType('r'),
            help='Input model file name' )
traingrp.add_argument( '--transfer', '--Transfer', dest='transfer',
            action='store_true', default=False,
            help='Do transfer training' )
traingrp.add_argument( '--trainmodelnm',
            dest='trainmodelnm', nargs='?', default='',
            help='Output trained model dataset name' )
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

parser = DefineBokehArguments(parser)

args = vars(parser.parse_args())

import odpy.common as odcommon
odcommon.initLogging( args )
odcommon.proclog_logger.setLevel( 'DEBUG' )

import dgbpy.servicemgr as dgbservmgr
srcfile = __file__


def training_app(doc):
# Keep all lengthy operations below
  from functools import partial
  import logging
  logging.getLogger('bokeh.bokeh_machine_learning.main').setLevel(logging.DEBUG)
  odcommon.proclog_logger = logging.getLogger('bokeh.bokeh_machine_learning.main')

  class MsgHandler(logging.StreamHandler):
    def __init__(self, msgstr, servmgr, msgkey, msgjson):
      logging.StreamHandler.__init__(self)
      self.msginfo = {}
      self.add(msgstr, msgkey, msgjson)
      self.servmgr = servmgr

    def add(self, msgstr, msgkey, msgjson):
      self.msginfo[msgstr] = {'msgkey': msgkey, 'jsonobj': msgjson}

    def emit(self, record):
      try:
        logmsg = self.format(record)
        for msgstr in self.msginfo.keys():
          if msgstr in logmsg:
             doc.add_next_tick_callback(partial(self.sendmsg, msgnm=msgstr))
      except (KeyboardInterrupt, SystemExit):
          raise
      except:
          self.handleError(record)

    def sendmsg(self, msgnm):
      self.servmgr.sendObject(self.msginfo[msgnm]['msgkey'], self.msginfo[msgnm]['jsonobj'])

  odcommon.log_msg( 'Start training UI')

  from os import path
  import psutil

  from bokeh.layouts import column, row
  from bokeh.models.widgets import Panel, Select, Tabs
  from bokeh.models import CheckboxGroup

  from odpy.oscommand import (getPythonCommand, execCommand, kill,
                            isRunning, pauseProcess, resumeProcess)
  import dgbpy.keystr as dgbkeys
  from dgbpy import mlapply as dgbmlapply
  from dgbpy import uibokeh, uisklearn, uitorch, uikeras
  from dgbpy import mlio as dgbmlio

  examplefilenm = args['h5file'].name
  trainingcb = None
  traintype =  dgbmlapply.TrainType.New
  doabort = False
  if 'model' in args:
    model = args['model']
    if model != None and len(model)>0:
      model = model[0].name
      if args['transfer']:
        traintype = dgbmlapply.TrainType.Transfer
      else:
        traintype = dgbmlapply.TrainType.Resume
  outmodelnm = None
  if 'trainmodelnm' in args:
      outmodelnm = args['trainmodelnm']

  trainscriptfp = path.join(path.dirname(path.dirname(srcfile)),'mlapplyrun.py')

  with dgbservmgr.ServiceMgr(args['bsmserver'],args['ppid'],args['port'],args['bokehid']) as this_service:
    traintabnm = 'Training'
    paramtabnm = 'Parameters'
    adparamtabnm = "Advanced"

    mh = MsgHandler('--Training Started--', this_service, 'ml_training_msg',
                   {'training_started': ''})
    mh.add('--Epoch0End--', 'ml_training_msg', {'show tensorboard': ''})
    mh.setLevel(logging.DEBUG)
    odcommon.proclog_logger.addHandler(mh)

    trainpanel = Panel(title=traintabnm)
    parameterspanel = Panel(title=paramtabnm)
    adparameterspanel = Panel(title=adparamtabnm)
    mainpanel = Tabs(tabs=[trainpanel,parameterspanel,adparameterspanel])

    ML_PLFS = []
    ML_PLFS.append( uikeras.getPlatformNm(True) )
    ML_PLFS.append( uitorch.getPlatformNm(True) )
    ML_PLFS.append( uisklearn.getPlatformNm(True) )

    platformfld = Select(title="Machine learning platform:",options=ML_PLFS)

    info = None
    keraspars = None
    torchpars = None  
    sklearnpars = None
    parsgroups = None
    kerasadvpars = None
    traininglogfilenm = args['logfile'].name

    def makeUI(examplefilenm):
      nonlocal info
      nonlocal keraspars
      nonlocal torchpars
      nonlocal sklearnpars
      nonlocal parsgroups
      nonlocal kerasadvpars
      info = dgbmlio.getInfo( examplefilenm, quick=True )
      uikeras.info = info
      uitorch.info = info
      uisklearn.info = info
      keraspars = uikeras.getUiPars()
      torchpars = uitorch.getUiPars()
      sklearnpars = uisklearn.getUiPars()
      parsgroups = (keraspars,torchpars,sklearnpars)
      kerasadvpars = uikeras.getAdvancedUiPars()

    def updateUI():
      nonlocal info
      nonlocal keraspars
      nonlocal torchpars
      nonlocal platformfld
      nonlocal ML_PLFS
      if info[dgbkeys.learntypedictstr] == dgbkeys.seisimgtoimgtypestr:
          platformfld.options.remove( uisklearn.getPlatformNm(True) )
      else:
          platformfld.options = ML_PLFS

      keraspars['uiobjects']['dodecimatefld'].active = []
      keraspars['uiobjects']['sizefld'].text = uikeras.getSizeStr(info[dgbkeys.estimatedsizedictstr])

    makeUI( examplefilenm )
    updateUI()

    def resetUiFields(cb):
      nonlocal keraspars
      nonlocal sklearnpars
      nonlocal torchpars
      platformnm = platformfld.value
      if platformnm == uikeras.getPlatformNm():
        keraspars = uikeras.getUiPars(keraspars)
      elif platformnm == uitorch.getPlatformNm():
        torchpars = uitorch.getUiPars(torchpars)
      elif platformnm == uisklearn.getPlatformNm():
        sklearnpars = uisklearn.getUiPars(sklearnpars)

    def resetAdvancedUiFields(cb):
      nonlocal kerasadvpars
      platformnm = platformfld.value
      if platformnm == uikeras.getPlatformNm():
        kerasadvpars = uikeras.getAdvancedUiPars(kerasadvpars)

    parsresetbut = uibokeh.getButton('Reset', callback_fn=resetUiFields)
    parsAdvancedResetBut = uibokeh.getButton('Reset', callback_fn=resetAdvancedUiFields)
    parsbackbut = uibokeh.getButton('Back',\
      callback_fn=partial(uibokeh.setTabFromButton,panelnm=mainpanel,tabnm=traintabnm))

    def procArgChgCB( paramobj ):
      nonlocal examplefilenm
      nonlocal model
      nonlocal traintype
      nonlocal info
      nonlocal traininglogfilenm
      for key, val in paramobj.items():
        if key=='Training Type':
          odcommon.log_msg(f'Change training type to "{val}".')
          if val == dgbmlapply.TrainType.New.name:
            traintype = dgbmlapply.TrainType.New
            model = None
          elif val == dgbmlapply.TrainType.Resume.name:
            traintype = dgbmlapply.TrainType.Resume
          elif val == dgbmlapply.TrainType.Transfer.name:
            traintype = dgbmlapply.TrainType.Transfer
        elif key=='Input Model File':
          odcommon.log_msg(f'Change pretrained input model to "{val}".')
          if os.path.isfile(val):
            model = val
          else:
            model = None
            traintype = dgbmlapply.TrainType.New
        elif key=='ProcLog File':
          odcommon.log_msg(f'Change log file name to "{val}".')
          traininglogfilenm = val
        elif key=='Output Model File':
          odcommon.log_msg(f'Change output model to "{val}".')
          doRun( doTrain(val) )
        elif key=='Examples File':
          odcommon.log_msg(f'Change input example data to "{val}".')
          if examplefilenm != val:
            examplefilenm = val
            info = dgbmlio.getInfo( examplefilenm, quick=True )
            uikeras.info = info
            doc.add_next_tick_callback(partial(updateUI))
      return dict()

    this_service.addAction('BokehParChg', procArgChgCB )

    def mlchgCB( attrnm, old, new):
      nonlocal adparameterspanel
      if new==uikeras.getPlatformNm(True)[0]:
        adparameterspanel.disabled = False
      else:
        adparameterspanel.disabled = True
      selParsGrp( new )

    def getParsGrp( platformnm ):
      for platform,parsgroup in zip(ML_PLFS,parsgroups):
        if platform[0] == platformnm:
          return parsgroup['grp']
      return None

    def selParsGrp( platformnm ):
      parsgrp = getParsGrp( platformnm )
      if parsgrp == None:
        return
      doc.clear()
      parameterspanel.child = column( parsgrp, row(parsresetbut, parsbackbut))
      doc.add_root(mainpanel)
      this_service.sendObject('ml_training_msg', {'platform_change': platformnm})

    def getUiParams():
      if platformfld.value == uikeras.getPlatformNm():
        return uikeras.getUiParams( keraspars, kerasadvpars )
      elif platformfld.value == uisklearn.getPlatformNm():
        return uisklearn.getUiParams( sklearnpars )
      elif platformfld.value == uitorch.getPlatformNm():
        return uitorch.getUiParams( torchpars )
      return {}

    def getProcArgs( platfmnm, pars, outnm ):
      ret = {
        'posargs': [examplefilenm],
        'odargs': odcommon.getODArgs( args ),
        'dict': {
          'platform': platfmnm,
          'parameters': pars,
          'output': outnm
        }
      }
      dict = ret['odargs']
      dict.update({'proclog': traininglogfilenm})
      dict = ret['dict']
      if model != None:
        dict.update({'model': model})

      if 'mldir' in args:
        mldir = args['mldir']
        if mldir != None and len(mldir)>0:
          dict.update({'logdir': mldir[0]})
          dict.update({'cleanlogdir': len(kerasadvpars['uiobjects']['cleartensorboardfld'].active)!=0})
      dict.update({dgbkeys.learntypedictstr: traintype.name})
      return ret

    def doRun( cb = None ):
      nonlocal trainingcb
      nonlocal doabort
      nonlocal outmodelnm
      doabort = False
      if cb == None and this_service.can_connect():
        this_service.sendObject('ml_training_msg', {'training can start request': ''})
        return True
      elif cb == False:
        doabort = True
        return False
      elif this_service.can_connect():
        trainingcb = {uibokeh.timerkey: cb}
      elif outmodelnm != None:
        trainingcb = {uibokeh.timerkey: doTrain(outmodelnm) }
      return True

    def doTrain( trainedfnm ):
      if len(trainedfnm) < 1:
        return False
      if platformfld.value==uikeras.getPlatformNm() and 'divfld' in keraspars['uiobjects']:
        odcommon.log_msg('\nNo Keras models found for this workflow.')
        return False

      modelnm = trainedfnm

      scriptargs = getProcArgs( platformfld.value, getUiParams(), \
                                modelnm )
      cmdtorun = getPythonCommand( trainscriptfp, scriptargs['posargs'], \
                              scriptargs['dict'], scriptargs['odargs'] )

      if platformfld.value == uikeras.getPlatformNm():
          this_service.sendObject('ml_training_msg', {'start tensorboard': ''})

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

    def trainMonitorCB( rectrainingcb ):
      proc = rectrainingcb[uibokeh.timerkey]
      nonlocal trainingcb
      nonlocal doabort
      if doabort:
        return (False,rectrainingcb)
      if proc == None:
        if trainingcb != None and uibokeh.timerkey in trainingcb:
          rectrainingcb[uibokeh.timerkey] = trainingcb[uibokeh.timerkey]
        return (True,rectrainingcb)
      if isRunning(proc):
        return (True,rectrainingcb)
      try:
        proc.status()
      except psutil.NoSuchProcess:
        if not odcommon.batchIsFinished( traininglogfilenm ):
          odcommon.log_msg( '\nProcess is no longer running (crashed or terminated).' )
          odcommon.log_msg( 'See OpendTect log file for more details (if available).' )
        else:
          this_service.sendObject('ml_training_msg', {'training_finished': ''})
        rectrainingcb[uibokeh.timerkey] = None
        trainingcb[uibokeh.timerkey] = None
        return (False,rectrainingcb)
      return (True,rectrainingcb)

    platformfld.on_change('value',mlchgCB)
    buttonsgrp = uibokeh.getRunButtonsBar( doRun, doAbort, doPause, doResume, trainMonitorCB )
    trainpanel.child = column( platformfld, buttonsgrp )
    adparameterspanel.child = column(kerasadvpars['grp'], row(parsAdvancedResetBut, parsbackbut))

    def initWin():
      mllearntype = info[dgbkeys.learntypedictstr]
      if mllearntype == dgbkeys.loglogtypestr or \
        mllearntype == dgbkeys.logclustertypestr or \
        mllearntype == dgbkeys.seisproptypestr:
        platformfld.value = uisklearn.getPlatformNm(True)[0]
      else:
        platformfld.value = uikeras.getPlatformNm(True)[0]
      mlchgCB( 'value', 0, platformfld.value )
      doc.title = 'Machine Learning'

    initWin()

StartBokehServer({'/': training_app}, args)
