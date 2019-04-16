#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Apr 2019
#
# _________________________________________________________________________

from enum import Enum
from functools import partial

from bokeh.core import enums
from bokeh.layouts import row
from bokeh.models import Spacer
from bokeh.models.widgets import Button
from bokeh.plotting import curdoc

but_width = 80
but_height = 32
but_spacer = 5
go_lbl = '▶ Run'
stop_lbl = '◼ Abort'
pause_lbl = '❚❚ Pause'
resume_lbl = '► Resume'

RunState = Enum( 'RunState', 'Ready Running Pause', module=__name__ )

def getButton(nm,type=enums.ButtonType.default,callback_fn=None):
  ret = Button(label=nm,button_type=type,width=but_width,height=but_height)
  if callback_fn != None:
    ret.on_click(partial(callback_fn,cb=ret))
  return ret

def getRunStopButton(callback_fn=None):
  return getButton(go_lbl,type=enums.ButtonType.success,callback_fn=callback_fn)

def getPauseResumeButton(callback_fn=None):
  return getButton(pause_lbl,type=enums.ButtonType.primary,callback_fn=callback_fn)

def getRunButtonsBar(runact,abortact,pauseact,resumeact,timercb):
  runstopbut = getRunStopButton()
  pauseresumebut = getPauseResumeButton()
  pauseresumebut.visible = False
  buttonsgrp = row(pauseresumebut,Spacer(width=but_spacer),runstopbut,width_policy='min')
  buttonsfld = row(Spacer(width=but_spacer),buttonsgrp, sizing_mode='stretch_width')
  ret = {
    'run': runstopbut,
    'pause': pauseresumebut,
    'state': RunState.Ready,
    'timerobj': None
  }
  runstopbut.on_click(partial(startStopCB,cb=ret,run_fn=runact,abort_fn=abortact,
                              timer_fn=timercb) )
  pauseresumebut.on_click(partial(pauseResumeCB,cb=ret,pause_fn=pauseact,resume_fn=resumeact))
  return buttonsfld

def startStopCB( cb, run_fn, abort_fn, timer_fn ):
  if isReady( cb ):
    setRunning( cb )
    cb['timerobj'] = run_fn( cb['timerobj'] )
    cb.update({'cb': curdoc().add_periodic_callback(partial(timerCB,cb=cb,timer_fn=timer_fn),2000)})
  else:
    setReady( cb )
    cb['timerobj'] = abort_fn( cb['timerobj'] )

def pauseResumeCB( cb, pause_fn, resume_fn ):
  if isRunning( cb ):
    setPaused( cb )
    cb['timerobj'] = pause_fn( cb['timerobj'] )
  else:
    setResumed( cb )
    cb['timerobj'] = resume_fn( cb['timerobj'] )

def timerCB( cb, timer_fn ):
  if not timer_fn( cb['timerobj'] ):
    setReady( cb )

def isReady( runbutbar ):
  return runbutbar['state'] == RunState.Ready

def isRunning( runbutbar ):
  return runbutbar['state'] == RunState.Running

def setRunning( runbutbar ):
  runbutbar['state'] = RunState.Running
  runbutbar['run'].label = stop_lbl
  runbutbar['run'].button_type = enums.ButtonType.danger
  runbutbar['pause'].visible = True

def setReady( runbutbar ):
  runbutbar['state'] = RunState.Ready
  runbutbar['run'].label = go_lbl
  runbutbar['run'].button_type = enums.ButtonType.success
  runbutbar['pause'].visible = False
  if 'cb' in runbutbar:
    cb = runbutbar.pop( 'cb' )
    curdoc().remove_periodic_callback( cb )

def setPaused( runbutbar ):
  runbutbar['state'] = RunState.Pause
  runbutbar['pause'].label = resume_lbl

def setResumed( runbutbar ):
  runbutbar['state'] = RunState.Running
  runbutbar['pause'].label = pause_lbl

def setTabFromButton( cb, panelnm, tabnm ):
  setActiveTab( panelnm, tabnm )

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
