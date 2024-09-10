#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Apr 2019
#
# _________________________________________________________________________

from enum import Enum
from functools import partial
import re

from dgbpy.bokehcore import *

import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5

but_width = 80
but_height = 32
but_spacer = 5
go_lbl = '▶ Run'
stop_lbl = '◼ Abort'
pause_lbl = '❚❚ Pause'
resume_lbl = '► Resume'
timerkey = 'timerobj'
parent_bar = 'epoch_bar'
child_bar = 'iter_bar'
widget_margin = (4, 0, 0, 0)

RunState = Enum( 'RunState', 'Ready Running Pause', module=__name__ )

def getButton(nm,type=enums.ButtonType.default,callback_fn=None):
  ret = Button(label=nm,button_type=type,width=but_width,height=but_height)
  if callback_fn != None:
    ret.on_click(partial(callback_fn,cb=ret))
  return ret

def getStopTrainingCheckBox():
  return CheckboxGroup(labels=['Stop training after current epoch'], visible=True)

def getRunStopButton(callback_fn=None):
  return getButton(go_lbl,type=enums.ButtonType.success,callback_fn=callback_fn)

def getPauseResumeButton(callback_fn=None):
  return getButton(pause_lbl,type=enums.ButtonType.primary,callback_fn=callback_fn)

class TrainStatusUI():
  def __init__(self):
    self.message = Div(text="", visible=False)

  def set_status(self, status):
    self.message.visible = True
    if status==TrainStatus.Success:
      self.message.text = '✅ Training Successful'
      self.message.styles = { "color":"#139D41", "background-color": "white", "font-size": "12px",
                            "border": "4px solid #0CB61E", "border-radius": "8px",
                            "box-shadow": "2px 2px 5px rgba(0, 0, 0, 0.5)", "padding": "5px 10px" }
    elif status==TrainStatus.Failed:
      self.message.text = '🚫 Training Failed'
      self.message.styles = { "color": "red", "background-color": "white", "font-size": "12px",
                            "border": "4px solid red", "border-radius": "8px", 
                            "box-shadow": "2px 2px 5px rgba(0, 0, 0, 0.5)", "padding": "5px 10px" } 

  def visible(self, bool):
    if bool:
      self.message.visible = True
    else:
      self.message.visible = False

class S3ProgressUI():
  """Show model upload progress as text in the UI"""
  def __init__(self):
    self.message = Div(text="", visible=False)

  def set_progress(self, msgstr):
    self.message.visible = True
    self.message.text = msgstr
    self.message.styles = { "background-color": "white", "font-size": "12px", "padding": "5px 10px" }

  def visible(self, bool):
    if bool:
      self.message.visible = True
    else:
      self.message.visible = False

def getPbar():
  chunk = Div(text = "Chunk 0 of 0")
  fold = Div(text = "Fold 0 of 0")
  parent = ProgBar(setProgValue(type=parent_bar))
  child = ProgBar(setProgValue(type=child_bar), parent=parent)
  status = TrainStatusUI()
  s3progress = S3ProgressUI()

  chunk.visible = False
  fold.visible = False
  parent.visible(False)
  child.visible(False)
  #status.visible(False)

  progressfld = column(chunk, fold, parent.panel(), child.panel(), s3progress.message, status.message)
  ret = {
    'chunk': chunk,
    dgbkeys.foldstr: fold,
    parent_bar: parent,
    child_bar : child,  
    'status' : status,
    'storage_msg': s3progress
    }
  return ret, progressfld

def getRunButtonsBar(progress,runact,abortact,pauseact,resumeact,progressact,timercb, stopaftercurrentepochact):
  runstopbut = getRunStopButton()
  pauseresumebut = getPauseResumeButton()
  stoptrainingcheckbox = getStopTrainingCheckBox()
  pauseresumebut.visible = False
  stoptrainingcheckbox.visible = False
  buttonsgrp = row(pauseresumebut,Spacer(width=but_spacer),runstopbut,width_policy='min')
  buttonsfld = column(row(Spacer(width=but_spacer), buttonsgrp, sizing_mode='stretch_width', align='end'), stoptrainingcheckbox)
  ret = {
    'run': runstopbut,
    'pause': pauseresumebut,
    'state': RunState.Ready,
    'progress': progress,
    'stopaftercurrentepoch': stoptrainingcheckbox,
    timerkey: None
  }
  progressact = partial(progressact, progress['chunk'], progress[dgbkeys.foldstr], progress[parent_bar], 
                        progress[child_bar], progress['status'], progress['storage_msg'])
  runstopbut.on_click(partial(startStopCB,cb=ret,run_fn=runact,abort_fn=abortact,progress_fn=progressact,
                              timer_fn=timercb, stopaftercurrentepoch_fn=stopaftercurrentepochact) )
  pauseresumebut.on_click(partial(pauseResumeCB,cb=ret,pause_fn=pauseact,resume_fn=resumeact))
  return buttonsfld

def startStopCB( cb, run_fn, abort_fn, progress_fn, timer_fn, stopaftercurrentepoch_fn, repeat=2000 ):
  stoptrainingcheckbox = cb['stopaftercurrentepoch']
  if isReady( cb ):
    canrun = run_fn( cb[timerkey] )
    if not canrun:
      return
    setRunning( cb , start_fn=partial(startBarUpdateCB, progress_fn))
    cb.update({
      'cb': curdoc().add_periodic_callback(partial(timerCB,cb=cb,timer_fn=timer_fn),repeat)
    })
    stoptrainingcheckbox.visible = True
    stopaftercurrentepoch_fn(stoptrainingcheckbox)
  else:
    setReady( cb )
    cb[timerkey] = abort_fn( cb[timerkey] )
    isAborted(cb)
    stoptrainingcheckbox.visible = False
    stopaftercurrentepoch_fn(stoptrainingcheckbox)

def isAborted(runbutbar):
  progress = runbutbar['progress']
  progress['status'].set_status(TrainStatus.Failed)


def pauseResumeCB( cb, pause_fn, resume_fn ):
  if isRunning( cb ):
    setPaused( cb )
    cb[timerkey] = pause_fn( cb[timerkey] )
  else:
    setResumed( cb )
    cb[timerkey] = resume_fn( cb[timerkey] )

def timerCB( cb, timer_fn ):
  (docontinue,cb) = timer_fn( cb )
  if not docontinue:
    setReady( cb )

def isReady( runbutbar ):
  return runbutbar['state'] == RunState.Ready

def isRunning( runbutbar ):
  return runbutbar['state'] == RunState.Running

def setRunning( runbutbar, start_fn=None):
  runbutbar['state'] = RunState.Running
  runbutbar['run'].label = stop_lbl
  runbutbar['run'].button_type = enums.ButtonType.danger
  runbutbar['pause'].visible = True
  if start_fn: start_fn(runbutbar['progress'])

def startBarUpdateCB(cb, ret):
  ret['status'].visible(False)
  ret['storage_msg'].visible(False)
  if 'cb' not in ret:
    ret['cb'] = curdoc().add_periodic_callback(cb, 100)

def endBarUpdateCB(ret):
  if 'cb' in ret:
    cb = ret.pop('cb')
    curdoc().remove_periodic_callback(cb)
  ret['chunk'].visible = False
  ret[dgbkeys.foldstr].visible = False
  ret[child_bar].visible(False)
  ret[child_bar].reset()
  ret[parent_bar].visible(False)
  ret[parent_bar].reset()

def setReady( runbutbar ):
  runbutbar['state'] = RunState.Ready
  runbutbar['run'].label = go_lbl
  runbutbar['run'].button_type = enums.ButtonType.success
  runbutbar['pause'].visible = False
  if 'cb' in runbutbar:
    cb = runbutbar.pop( 'cb' )
    curdoc().remove_periodic_callback( cb )
  endBarUpdateCB(runbutbar['progress'])


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

def getGroup( grpnm, options, groups, keywd ):
  for opt,group in zip(options,groups):
    if opt == grpnm:
      if keywd in group:
        return group[keywd]
      return group
  return None

def getAllUiFlds( objects ):
  ret = list()
  if 'uiobjects' in objects:
    for uiobj in objects['uiobjects']:
      ret.append( objects['uiobjects'][uiobj] )
  else:
    try:
      for obj in objects:
        ret.extend( getAllUiFlds(obj) )
    except TypeError:
      pass
  return ret

class ProgBar():
  def __init__(self, value, parent=None, **kw):
    """
      Creates a progress bar widget with methods to control its state
    """
    self.success_ = True
    self.current_value_ = 0
    self.current_step_ = 0
    self.parent = parent
    self.source_ = ColumnDataSource(data={'x_values': [0]})
    self.fig_ = figure(height=20, **kw)
    self.fig_.x_range = Range1d(0, 100, bounds=(0, 100))
    self.fig_.y_range = Range1d(0, 1, bounds=(0, 1))
    self.bar = self.fig_.hbar(y=0.5, right='x_values', height=2, left=0, color='#00779B', source=self.source_, level='underlay')
    self.fig_.xgrid.grid_line_color = None
    self.fig_.ygrid.grid_line_color = None
    self.fig_.toolbar_location = None
    self.fig_.yaxis.visible = False
    self.fig_.xaxis.visible = False
    self.div = Div(text=value)

  def first_init(self, value):
    self.div.text = value

  def reset(self):
    self.current_value_ = 0
    self.current_step_ = 0
    self.bar.glyph.line_color = "#00779B"
    self.bar.glyph.fill_color = "#00779B"
    self.source_.data['x_values'] = [self.current_value_]

  def visible(self, bool):
    self.fig_.visible=bool
    self.div.visible=bool

  def set(self, current, total):
    if not self.parent:
      self.div.text = setProgValue(type=parent_bar, current=current, total=total)
    else:
      self.div.text = setProgValue(type=child_bar, current=current, total=total)
    self.current_value_ = percentage(current, total)
    self.current_step_ = current
    self.source_.data['x_values'] = [self.current_value_]
    if self.current_value_ == 100:
      self.bar.glyph.line_color = "#009B77"
      self.bar.glyph.fill_color = "#009B77"

  def get(self):
    return self.current_value_

  def fail(self):
    self.success_ = False
    self.bar.glyph.line_color = "#9B3333"
    self.bar.glyph.fill_color = "#9B3333"
    if self.current_value_ < 10:
        self.current_value_ = 50
        self.source_.data['x_values'] = [self.current_value_]
    return self.success_

  def panel(self):
    return column(self.div, self.fig_)


def setProgValue(type=None, current=0, total=0):
  """
  Create text value for a progress div widget
  """
  if type==parent_bar:
    text = f"<b>Epoch {current}/{total}</b>"
    return text
  if type==child_bar:
    text = f"<b>Iteration {current}/{total}</b>"
    return text
  return f"<b>{type} {current}/{total}</b>"

def percentage(current, total):
  """ Find percentage between current and total"""
  try:
    return int((current/total)*100)
  except ZeroDivisionError:
    return 0

def getProgMsg(msgstr):
  """Returns the current and total iteration from msgstr"""
  current = re.findall(r'\w+', msgstr)
  return int(current[1]), int(current[3])

def getProgValue(text):
  """Returns the current and total iteration from Div widget in ProgBar """
  text = re.findall(r'\w+', text)
  return int(text[-3]), int(text[-2])

class ProgState(Enum):
  Ready = 0
  Running = 1
  Started = 2

class TrainStatus(Enum):
  Default = 0
  Success = 1
  Failed = 2

def set_augment_mthds(info):
  from dgbpy.transforms import hasOpenCV
  labels = ['Flip Left/Right', 'Add Gaussian Noise', 'Flip Polarity']
  if hasOpenCV():
    labels.append('Rotate')
  if dgbhdf5.isSeisClass(info):
    labels.append('Add Empty Edges')
  return labels

augment_ui_map = {
                    'Flip Left/Right': 'Flip',
                    'Add Gaussian Noise': 'GaussianNoise',
                    'Flip Polarity': 'FlipPolarity',
                    'Rotate': 'Rotate',
                    'Add Empty Edges': 'Translate'
                   }