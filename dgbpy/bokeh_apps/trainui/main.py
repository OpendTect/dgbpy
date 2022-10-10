#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Jan 2019
#
# _________________________________________________________________________

import os
import sys
import psutil
import json
from functools import partial
import logging
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models.widgets import Panel, Select, Tabs
from bokeh.models import CheckboxGroup

from odpy.oscommand import (getPythonCommand, execCommand, kill,
                            isRunning, pauseProcess, resumeProcess)
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
from dgbpy import mlapply as dgbmlapply
from dgbpy import uibokeh, uisklearn, uitorch, uikeras
from dgbpy import mlio as dgbmlio
from dgbpy.servicemgr import ServiceMgr
from dgbpy.bokehserver import get_request_id

import odpy.common as odcommon

from trainui import get_default_info, MsgHandler, get_platforms, get_default_platform, uinoplfm
logging.getLogger('bokeh.bokeh_machine_learning.main').setLevel(logging.DEBUG)
odcommon.proclog_logger = logging.getLogger('bokeh.bokeh_machine_learning.main')
odcommon.proclog_logger.setLevel( 'DEBUG' )

odcommon.log_msg( 'Before start training UI')

srcfile = __file__

def training_app(doc):
  odcommon.log_msg( 'Start training UI')
  trainingpars = {
    'Examples File': None,
    'Training Type': dgbmlapply.TrainType.New,
    'Input Model File': None,
    'Output Model File': None,
    'Proc Log File': None,
    'ComArgs': None
  }
  
  info = get_default_info()

  def set_info():
    uikeras.info = info
    uitorch.info = info
    uisklearn.info = info

  def trainingParChgCB( paramobj ):
    nonlocal trainingpars
    nonlocal info
    dorun = False
    for key, val in paramobj.items():
      if key=='Training Type':
        odcommon.log_msg(f'Change training type to "{val}".')
        if val == dgbmlapply.TrainType.New.name:
          trainingpars['Training Type'] = dgbmlapply.TrainType.New
          trainingpars['Input Model File'] = None
        elif val == dgbmlapply.TrainType.Resume.name:
          trainingpars['Training Type'] = dgbmlapply.TrainType.Resume
        elif val == dgbmlapply.TrainType.Transfer.name:
          trainingpars['Training Type'] = dgbmlapply.TrainType.Transfer
      elif key=='Input Model File':
        odcommon.log_msg(f'Change pretrained input model to "{val}".')
        if os.path.isfile(val):
          trainingpars['Input Model File'] = val
        else:
          trainingpars['Input Model File'] = None
          trainingpars['Training Type'] = dgbmlapply.TrainType.New
      elif key=='ProcLog File':
        odcommon.log_msg(f'Change log file name to "{val}".')
        trainingpars['Proc Log File'] = val
      elif key=='Output Model File':
        odcommon.log_msg(f'Change output model to "{val}".')
        trainingpars['Output Model File'] = val
        dorun = True
      elif key=='Examples File':
        odcommon.log_msg(f'Change input example data to "{val}".')
        if trainingpars['Examples File'] != val:
          trainingpars['Examples File'] = val
          if get_default_platform() != uinoplfm().getPlatformNm(True)[0]:
            info = dgbmlio.getInfo( trainingpars['Examples File'], quick=True )
          set_info()
          doc.add_next_tick_callback(partial(updateUI))
      elif key=='ComArgs':
        odcommon.log_msg(f'Change command line args to "{val}".')
        trainingpars['ComArgs'] = {}
        for key,item in val.items():
          trainingpars['ComArgs'].update({key: item.split(',')})

    if dorun:
      doRun( doTrain(trainingpars['Output Model File']) )
    return dict()

  keraspars = None
  torchpars = None
  sklearnpars = None
  parsgroups = None
  advparsgroups = None
  kerasadvpars = None
  torchadvpars = None

  def makeUI(examplefilenm):
    nonlocal info
    nonlocal keraspars
    nonlocal torchpars
    nonlocal sklearnpars
    nonlocal parsgroups
    nonlocal advparsgroups
    nonlocal kerasadvpars
    nonlocal torchadvpars
    if examplefilenm and get_default_platform() != uinoplfm().getPlatformNm(True)[0]:
      info = dgbmlio.getInfo( examplefilenm, quick=True )

  trainingcb = None
  doabort = False
  trainscriptfp = os.path.join(os.path.dirname(os.path.dirname(srcfile)),'..','mlapplyrun.py')
  traintabnm = 'Training'
  paramtabnm = 'Parameters'
  adparamtabnm = "Advanced"
  trainpanel = Panel(title=traintabnm)
  parameterspanel = Panel(title=paramtabnm)
  adparameterspanel = Panel(title=adparamtabnm)
  mainpanel = Tabs(tabs=[trainpanel,parameterspanel,adparameterspanel])
  platformfld = Select(title="Machine learning platform:",options=get_platforms())
  tensorboardfld = CheckboxGroup(labels=['Clear Tensorboard log files'], inline=True,
                                   active=[], visible=True)
  this_service = None
  parsresetbut = None
  parsbackbut = None

  def getParsGrp( platformnm ):
    for platform,parsgroup,advparsgroup in zip(get_platforms(),parsgroups,advparsgroups):
      if platform[0] == platformnm:
        if parsgroup and advparsgroup:
          return parsgroup['grp'], advparsgroup['grp']
        if parsgroup and not advparsgroup:
          return parsgroup['grp'], None
    return None,None

  def setAdvPanel (advparsgrp):
    if advparsgrp:
      adparameterspanel.child = column(advparsgrp, row(parsAdvancedResetBut, parsbackbut))
    else:
      adparameterspanel.child = row(parsAdvancedResetBut, parsbackbut)

  def selParsGrp( platformnm ):
    parsgrp,advparsgrp = getParsGrp( platformnm )
    if not parsresetbut or not parsbackbut :
      return
    doc.clear()
    if not parsgrp:
      parameterspanel.child = row(parsAdvancedResetBut, parsbackbut)
      adparameterspanel.child = row(parsAdvancedResetBut, parsbackbut)
      parameterspanel.disabled = True
      adparameterspanel.disabled = True
    else:
      parameterspanel.child = column( parsgrp, row(parsresetbut, parsbackbut))
      setAdvPanel(advparsgrp)
    doc.add_root(mainpanel)
    if this_service:
      this_service.sendObject('bokeh_app_msg', {'platform_change': platformnm})

  def getKerasUiPars():
    _keraspars, _kerasadvpars=None,None
    if uikeras.hasKeras():
      _keraspars = uikeras.getUiPars()
      _kerasadvpars = uikeras.getAdvancedUiPars()
    return _keraspars, _kerasadvpars
  
  def getTorchUiPars():
    _torchpars, _torchadvpars=None,None
    if uitorch.hasTorch():
      _torchpars = uitorch.getUiPars()
      _torchadvpars = uitorch.getAdvancedUiPars()
    return _torchpars, _torchadvpars

  def getSklearnUiPars():
    _sklearnpars, _sklearnadvpars=None,None
    if uisklearn.hasScikit():
      _sklearnpars = uisklearn.getUiPars()
    return _sklearnpars, _sklearnadvpars

  def setPlatformGrp(*args):
    """
      Returns a tuple of parameters for all available platforms
      -- Parameters should be in the order of keras, torch, sklearn.
    """
    grps = ()
    for arg in args:
      if arg:
        grps+=arg,
    if not bool(grps):
      return (None,)
    return grps

  def mlchgCB( attrnm, old, new):
    nonlocal info
    nonlocal keraspars
    nonlocal torchpars
    nonlocal sklearnpars
    nonlocal parsgroups
    nonlocal advparsgroups
    nonlocal kerasadvpars
    nonlocal torchadvpars
    nonlocal tensorboardfld
    nonlocal adparameterspanel
    set_info()
    keraspars, kerasadvpars = getKerasUiPars()
    torchpars, torchadvpars = getTorchUiPars()
    sklearnpars, sklearnadvpars = getSklearnUiPars()
    parsgroups = setPlatformGrp(keraspars,torchpars,sklearnpars)
    advparsgroups = (kerasadvpars, torchadvpars, sklearnadvpars)
    if keraspars:
      keraspars['uiobjects']['dodecimatefld'].active = []
      keraspars['uiobjects']['sizefld'].text = uikeras.getSizeStr(info[dgbkeys.estimatedsizedictstr])
    if new==uikeras.getPlatformNm(True)[0] or (new==uitorch.getPlatformNm(True)[0] and not dgbhdf5.isLogOutput(uitorch.info)):
      adparameterspanel.disabled = False
    else:
      adparameterspanel.disabled = True
    selParsGrp( new )

  def updateUI():
    nonlocal platformfld

    if info[dgbkeys.learntypedictstr] == dgbkeys.seisimgtoimgtypestr:
      platformfld.options.remove( uisklearn.getPlatformNm(True) )
      if platformfld.value == uisklearn.getPlatformNm(False):
        platformfld.value = get_default_platform(info[dgbkeys.learntypedictstr])
    else:
      platformfld.options = get_platforms()
    platformfld.value = get_default_platform(info[dgbkeys.learntypedictstr])
    mlchgCB('value', 0, platformfld.value)

  makeUI( trainingpars['Examples File'] )
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
    nonlocal torchadvpars
    platformnm = platformfld.value
    if platformnm == uikeras.getPlatformNm():
      kerasadvpars = uikeras.getAdvancedUiPars(kerasadvpars)
    if platformnm == uitorch.getPlatformNm():
      torchadvpars = uitorch.getAdvancedUiPars(torchadvpars)

  parsresetbut = uibokeh.getButton('Reset', callback_fn=resetUiFields)
  parsAdvancedResetBut = uibokeh.getButton('Reset', callback_fn=resetAdvancedUiFields)
  parsbackbut = uibokeh.getButton('Back',\
    callback_fn=partial(uibokeh.setTabFromButton,panelnm=mainpanel,tabnm=traintabnm))

  def getUiParams():
    if platformfld.value == uikeras.getPlatformNm():
      odcommon.log_msg(uikeras.getUiParams(keraspars,kerasadvpars))
      return uikeras.getUiParams( keraspars, kerasadvpars )
    elif platformfld.value == uisklearn.getPlatformNm():
      return uisklearn.getUiParams( sklearnpars )
    elif platformfld.value == uitorch.getPlatformNm():
      odcommon.log_msg(uitorch.getUiParams(torchpars,torchadvpars))
      return uitorch.getUiParams( torchpars, torchadvpars )
    return {}

  def getProcArgs( platfmnm, pars, outnm ):
    nonlocal trainingpars
    ret = {
      'posargs': [trainingpars['Examples File']],
      'odargs': trainingpars['ComArgs'].copy(),
      'dict': {
        'platform': platfmnm,
        'parameters': pars,
        'output': outnm
      }
    }
    dict = ret['odargs']
    dict.update({'proclog': trainingpars['Proc Log File']})
    mldir = dict.pop('logdir', None)

    dict = ret['dict']
    if trainingpars['Input Model File']:
      dict.update({'model': trainingpars['Input Model File']})

    if mldir:
      dict.update({'logdir': mldir[0]})
      if platfmnm == uikeras.getPlatformNm():
        dict.update({'cleanlogdir': len(kerasadvpars['uiobjects']['cleartensorboardfld'].active)!=0})
      elif platfmnm == uitorch.getPlatformNm():
        dict.update({'cleanlogdir': len(torchadvpars['uiobjects']['cleartensorboardfld'].active)!=0})

    dict.update({dgbkeys.learntypedictstr: trainingpars['Training Type'].name})
    return ret

  def doRun( cb = None ):
    nonlocal trainingpars
    nonlocal trainingcb
    nonlocal doabort
    doabort = False
    if cb == None and this_service and this_service.can_connect():
      this_service.sendObject('bokeh_app_msg', {'training can start request': ''})
      return True
    elif cb == False:
      doabort = True
      return False
    elif this_service and this_service.can_connect():
      trainingcb = {uibokeh.timerkey: cb}
    elif trainingpars['Output Model File'] != None:
      trainingcb = {uibokeh.timerkey: doTrain(trainingpars['Output Model File']) }
    return True

  def doTrain( trainedfnm ):
    if len(trainedfnm) < 1:
      return False
    if platformfld.value==uikeras.getPlatformNm():
      if 'divfld' in keraspars['uiobjects']:
        odcommon.log_msg('\nNo Keras models found for this workflow.')
        return False

    modelnm = trainedfnm

    scriptargs = getProcArgs( platformfld.value, getUiParams(), \
                                modelnm )
    cmdtorun = getPythonCommand( trainscriptfp, scriptargs['posargs'], \
                            scriptargs['dict'], scriptargs['odargs'] )

    if (platformfld.value == uikeras.getPlatformNm() or platformfld.value == uitorch.getPlatformNm())  and this_service:
      this_service.sendObject('bokeh_app_msg', {'start tensorboard': ''})

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
      if not odcommon.batchIsFinished( trainingpars['Proc Log File'] ):
        odcommon.log_msg( '\nProcess is no longer running (crashed or terminated).' )
        odcommon.log_msg( 'See OpendTect log file for more details (if available).' )
      elif this_service:
        this_service.sendObject('bokeh_app_msg', {'training_finished': ''})
      rectrainingcb[uibokeh.timerkey] = None
      trainingcb[uibokeh.timerkey] = None
      return (False,rectrainingcb)
    return (True,rectrainingcb)

  platformfld.on_change('value',mlchgCB)
  buttonsgrp = uibokeh.getRunButtonsBar( doRun, doAbort, doPause, doResume, trainMonitorCB )
  trainpanel.child = column( platformfld, buttonsgrp )

  def initWin():
    nonlocal info
    nonlocal platformfld
    platformfld.value = get_default_platform(info[dgbkeys.learntypedictstr])
    mlchgCB( 'value', 0, platformfld.value )
    doc.title = 'Machine Learning'

  args = curdoc().session_context.server_context.application_context.application.metadata
  if args:
    this_service = ServiceMgr(args['bsmserver'],args['ppid'],args['port'],get_request_id())
    this_service.addAction('BokehParChg', trainingParChgCB )
    mh = MsgHandler('--Training Started--', this_service, 'bokeh_app_msg', {'training_started': ''})
    mh.add('--ShowTensorboard--', 'bokeh_app_msg', {'show tensorboard': ''})
    mh.setLevel(logging.DEBUG)
    odcommon.proclog_logger.addHandler(mh)
  else:
    if len(sys.argv)>1:
      data = json.loads(sys.argv[1])
      trainingParChgCB(data)

  initWin()

training_app(curdoc())

