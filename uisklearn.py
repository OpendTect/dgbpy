#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Apr 2019
#
# _________________________________________________________________________

from functools import partial

from bokeh.core import enums
from bokeh.layouts import row, column
from bokeh.models import Spacer
from bokeh.models.widgets import Button, Select, Slider

from odpy.common import log_msg
from dgbpy.dgbscikit import *
from dgbpy import uibokeh

but_width = uibokeh.but_width
but_height = uibokeh.but_height
but_spacer = uibokeh.but_spacer
defaultbut = enums.ButtonType.default

def getPlatformNm( full=False ):
  if full:
    return platform
  return getMLPlatform()

def getLinearGrp():
  lineartyp = Select(title='Model',value='Ordinary Least Squares',options=getUiLinearTypes() )
  return {
    'uiobjects': {
      'lineartyp': lineartyp
    },
    'name': regmltypes[0][1]
  }

def getLogGrp():
  logtyp = Select(title='Model',value='Logistic Regression Classifier',options=getUiLogTypes() )
  solvertyp = Select(title='Solver',value=getDefaultSolver(),
                     options=getUiSolverTypes() )
  return {
    'uiobjects': {
      'logtyp': logtyp,
      'solvertyp': solvertyp
    },
    'name': classmltypes[0][1]
  }

def getEnsembleGrp():
  ensembletyp = Select(title='Model',value = 'Random Forests',options=getUiEnsembleTypes())
  rfgrp = getRFGrp()
  gbgrp = getGBGrp()
  adagrp = getAdaGrp()
  ensemblegrp = (rfgrp,gbgrp,adagrp)
  ret = {
    'uiobjects': {
      'ensembletyp': ensembletyp,
      'depparfldrf': rfgrp['uiobjects']['depparfldrf'],
      'estparfldrf': rfgrp['uiobjects']['estparfldrf'],
      'depparfldgb': gbgrp['uiobjects']['depparfldgb'],
      'estparfldgb': gbgrp['uiobjects']['estparfldgb'], 
      'lrparfldgb': gbgrp['uiobjects']['lrparfldgb'],
      'estparfldada': adagrp['uiobjects']['estparfldada'],
      'lrparfldada': adagrp['uiobjects']['lrparfldada'], 
    },
    'name': regmltypes[1][1],
  }
  if hasXGBoost():
    xggrp = getXGGrp()
    ensemblegrp += (xggrp,)
    xggrpuiobjs = xggrp['uiobjects']
    ret['uiobjects'].update( {
      'depparfldxg': xggrpuiobjs['depparfldxg'],
      'estparfldxg': xggrpuiobjs['estparfldxg'],
      'lrparfldxg': xggrpuiobjs['lrparfldxg']
    })
  ensembletyp.on_change('value',partial(ensembleChgCB,cb=ensembletyp,
                        ensemblegrp=ensemblegrp))
  return ret

def getNNGrp():
  dict = scikit_dict
  nntyp = Select(title='Model',value = 'Multi-Layer Perception',options=getUiNNTypes() )
  itrparfld = Slider(start=10,end=1000,value=dict['nnpars']['maxitr'],title='Max Iteration')
  lay1parfld = Slider(start=1,end=200,value=50,
                      title='Layer 1', callback_policy='mouseup')
  lay2parfld = Slider(start=1,end=50,value=dict['nnpars']['lay2'],
                      title='Layer 2',callback_policy='mouseup')
  lay3parfld = Slider(start=1,end=10,value=dict['nnpars']['lay3'],
                      title='Layer 3',callback_policy='mouseup')
  lay4parfld = Slider(start=1,end=5,value=dict['nnpars']['lay4'],
                      title='Layer 4',callback_policy='mouseup')
  lay5parfld = Slider(start=1,end=3,value=dict['nnpars']['lay5'],
                      title='Layer 5',callback_policy='mouseup')
  lrparfld = Slider(start=1,end=100,value=dict['nnpars']['lr']*1000,
                    title='Initial Learning Rate '+ '('+u'\u2030'+')')
  # just need number, we treat like this in order to simplify further code
  nb = Slider(start=1,end=5,value=3)
  addbutton = Button(label='Add',button_type=defaultbut,width=but_width,height=but_height)
  lessbutton = Button(label='Less',button_type=defaultbut,width=but_width,height=but_height)
  buttonparfld = row(addbutton,Spacer(width = 5),lessbutton,sizing_mode='stretch_width')
  layergrp = [nb,lay1parfld,lay2parfld,lay3parfld,lay4parfld,lay5parfld,
              addbutton,lessbutton]
  lay1parfld.on_change('value',partial(layer1ChgCB,layergrp))
  lay2parfld.on_change('value',partial(layer2ChgCB,layergrp))
  lay3parfld.on_change('value',partial(layer3ChgCB,layergrp))
  lay4parfld.on_change('value',partial(layer4ChgCB,layergrp))
  try:
    lay1parfld.value_throttled = lay1parfld.value
    lay2parfld.value_throttled = lay2parfld.value
    lay3parfld.value_throttled = lay3parfld.value
    lay4parfld.value_throttled = lay4parfld.value
    lay1parfld.on_change('value_throttled',partial(layer1ChgCB,layergrp))
    lay2parfld.on_change('value_throttled',partial(layer2ChgCB,layergrp))
    lay3parfld.on_change('value_throttled',partial(layer3ChgCB,layergrp))
    lay4parfld.on_change('value_throttled',partial(layer4ChgCB,layergrp))
  except AttributeError:
    log_msg( '[WARNING] Bokeh version too old, consider updating it.' )
    pass
  addbutton.on_click(partial(buttonChgCB,addbutton,layergrp))
  lessbutton.on_click(partial(buttonChgCB,lessbutton,layergrp))
  return {
    'uiobjects': {
      'nntyp': nntyp,
      'itrparfld': itrparfld,
      'lay1parfld': lay1parfld,
      'lay2parfld': lay2parfld,
      'lay3parfld': lay3parfld,
      'lay4parfld': lay4parfld,
      'lay5parfld': lay5parfld,
      'lrparfld': lrparfld,
      'addbutton': addbutton,
      'lessbutton': lessbutton,
      'buttonparfld': buttonparfld,
    },
    'name': regmltypes[2][1],
    'nb': nb
  }

def getSVMGrp():
  dict = scikit_dict
  defkernelstr = getDefaultNNKernel()
  svmtyp = Select(title='Model',value='Support Vector Machine',options=getUiSVMTypes() )
  kernel = Select(title='Kernel',value=defkernelstr,
                  options=getUiNNKernelTypes())
  degree = Slider(start=1,end=5,value=dict['svmpars']['degree'],step=1,title='Degree')
  kernel.on_change('value',partial(kernelChgCB,deg=degree))
  kernelChgCB( 'value', defkernelstr, defkernelstr,degree )
  return {
    'uiobjects': {
      'svmtyp': svmtyp,
      'kernel': kernel,
      'degree': degree
    },
    'name': regmltypes[3][1]
  }

#deg visible only when polynomial kernel function
def kernelChgCB( attrnm, old, new,deg):
  if new == 'Polynomial':
    deg.visible = True
  else:
    deg.visible = False

def layer1ChgCB(layergrp,attr,old,new):
  layergrp[2].end = new
  if new <= layergrp[2].value:
    layergrp[2].value = new
    layer2ChgCB(layergrp,attr,layergrp[2].value,new)

def layer2ChgCB(layergrp,attr,old,new):
  layergrp[3].end = new
  if new <= layergrp[3].value:
    layergrp[3].value = new
    layer3ChgCB(layergrp,attr,layergrp[2].value,new)
  
def layer3ChgCB(layergrp,attr,old,new):
  layergrp[4].end = new
  if new <= layergrp[4].value:
    layergrp[4].value = new
    layer4ChgCB(layergrp,attr,layergrp[2].value,new)
  
def layer4ChgCB(layergrp,attr,old,new):
  layergrp[5].end = new
  if new <= layergrp[5].value:
    layergrp[5].value = new

def buttonChgCB(addbutton,layergrp):
  log_msg('Working')
  nb = layergrp[0].value #number of layers
  if addbutton.label == 'Add':
    layergrp[nb+1].visible = True
    if nb == 4:
      layergrp[6].visible = False # Hide add button
    elif nb == 1:
      layergrp[7].visible = True # Show less button
    layergrp[0].value += 1
  elif addbutton.label == 'Less':
    layergrp[nb].visible = False
    if nb == 5:
      layergrp[6].visible = True # show add button
    elif nb == 2:
      layergrp[7].visible = False # Hide less button
    layergrp[0].value -= 1
  else:
    return None

def modelChgCB( attrnm, old, new, cb, modelsgrp ):
  ret = uibokeh.getAllUiFlds( modelsgrp )
  for uifld in  ret:
    uifld.visible = False
  ret = uibokeh.getGroup( new, cb.options, modelsgrp, 'uiobjects' )
  # set default visibility
  if new == 'Ensemble':
    ret['ensembletyp'].visible = True
    ret['depparfldrf'].visible = True
    ret['estparfldrf'].visible = True
  elif new == 'SVM':
    ret['svmtyp'].visible = True
    ret['kernel'].visible = True
  else:
    for uifld in  ret:
      ret[uifld].visible = True
    if new == 'Neural Network':
      ret['lay4parfld'].visible = False
      ret['lay5parfld'].visible = False

def getRFGrp():
  dict = scikit_dict['ensemblepars']['rf']
  estparfldrf = Slider(start=10,end=1000,value=dict['est'],step=10,title='Estimators')
  depparfldrf = Slider(start=10,end=200,value=dict['maxdep'],step=10,title='Max Depth')
  return {
    'uiobjects': {
      'depparfldrf': depparfldrf,
      'estparfldrf': estparfldrf
      }
    }

def getGBGrp():
  dict = scikit_dict['ensemblepars']['gb']
  depparfldgb = Slider(start=1,end=100,value=dict['maxdep'],step=1,title='Max Depth')
  estparfldgb = Slider(start=10,end=500,value=dict['est'],step=10,title='Estimators')
  lrparfldgb = Slider(start=0.1,end=10,value=dict['lr'],step=0.1,title='Learning Rate')
  return {
    'uiobjects': {
      'depparfldgb': depparfldgb,
      'estparfldgb': estparfldgb,
      'lrparfldgb': lrparfldgb
      }
    }

def getAdaGrp():
  dict = scikit_dict['ensemblepars']['ada']
  estparfldada = Slider(start=10,end=500,value=dict['est'],step=10,title='Estimators')
  lrparfldada = Slider(start=0.1,end=10,value=dict['lr'],step=0.1,title='Learning Rate')
  return {
    'uiobjects': {
      'estparfldada': estparfldada,
      'lrparfldada': lrparfldada
      }
    }

def getXGGrp():
  if not hasXGBoost():
    return None
  dict = scikit_dict['ensemblepars']['xg']
  depparfldxg = Slider(start=10,end=200,value=dict['maxdep'],step=10,title='Max Depth')
  estparfldxg = Slider(start=10,end=1000,value=dict['est'],step=10,title='Estimators')
  lrparfldxg = Slider(start=0.1,end=10,value=dict['lr'],step=0.1,title='Learning Rate')
  return {
    'uiobjects': {
      'depparfldxg': depparfldxg,
      'estparfldxg': estparfldxg,
      'lrparfldxg': lrparfldxg
      }
    }

def ensembleChgCB( attrnm, old, new, cb, ensemblegrp ):
  ret = uibokeh.getAllUiFlds( ensemblegrp )
  for uifld in  ret:
    uifld.visible = False
  ret = uibokeh.getGroup( new, cb.options, ensemblegrp, 'uiobjects' )
  for uifld in  ret:
    ret[uifld].visible = True

def getUiPars(isclassification):
  models = getUiModelTypes(isclassification)
  if isclassification:
    modeltyp = Select(title='Type',value = models[0],options=models)
  else:
    modeltyp = Select(title='Type',value = models[1],options=models)
  if isclassification:
    lineargrp = getLogGrp()
    deftype = classmltypes[0][1]
    linearkey = 'loggrp'
  else:
    lineargrp = getLinearGrp()
    deftype = regmltypes[1][1]
    linearkey = 'lineargrp'
  ensemblegrp = getEnsembleGrp()
  nngrp = getNNGrp()
  svmgrp = getSVMGrp()
  modelsgrp = (lineargrp,ensemblegrp,nngrp,svmgrp)
  modeltyp.on_change('value',partial(modelChgCB,cb=modeltyp,modelsgrp=modelsgrp))
  modelChgCB( 'value', deftype, deftype, modeltyp, modelsgrp )
  if isclassification:
    if hasXGBoost():
      parsgrp = column(modeltyp, \
                       lineargrp['uiobjects']['logtyp'], \
                       lineargrp['uiobjects']['solvertyp'], \
                       ensemblegrp['uiobjects']['ensembletyp'], \
                       ensemblegrp['uiobjects']['estparfldrf'], \
                       ensemblegrp['uiobjects']['depparfldrf'], \
                       ensemblegrp['uiobjects']['estparfldgb'], \
                       ensemblegrp['uiobjects']['depparfldgb'], \
                       ensemblegrp['uiobjects']['lrparfldgb'], \
                       ensemblegrp['uiobjects']['estparfldada'], \
                       ensemblegrp['uiobjects']['lrparfldada'], \
                       ensemblegrp['uiobjects']['estparfldxg'], \
                       ensemblegrp['uiobjects']['depparfldxg'], \
                       ensemblegrp['uiobjects']['lrparfldxg'], \
                       nngrp['uiobjects']['nntyp'], \
                       nngrp['uiobjects']['itrparfld'], \
                       nngrp['uiobjects']['lrparfld'], \
                       nngrp['uiobjects']['lay1parfld'], \
                       nngrp['uiobjects']['lay2parfld'], \
                       nngrp['uiobjects']['lay3parfld'], \
                       nngrp['uiobjects']['lay4parfld'], \
                       nngrp['uiobjects']['lay5parfld'], \
                       nngrp['uiobjects']['buttonparfld'], \
                       svmgrp['uiobjects']['svmtyp'], \
                       svmgrp['uiobjects']['kernel'], \
                       svmgrp['uiobjects']['degree']
                       )
    else:
      parsgrp = column(modeltyp, \
                       lineargrp['uiobjects']['logtyp'], \
                       lineargrp['uiobjects']['solvertyp'], \
                       ensemblegrp['uiobjects']['ensembletyp'], \
                       ensemblegrp['uiobjects']['estparfldrf'], \
                       ensemblegrp['uiobjects']['depparfldrf'], \
                       ensemblegrp['uiobjects']['estparfldgb'], \
                       ensemblegrp['uiobjects']['depparfldgb'], \
                       ensemblegrp['uiobjects']['lrparfldgb'], \
                       ensemblegrp['uiobjects']['estparfldada'], \
                       ensemblegrp['uiobjects']['lrparfldada'], \
                       nngrp['uiobjects']['nntyp'], \
                       nngrp['uiobjects']['itrparfld'], \
                       nngrp['uiobjects']['lrparfld'], \
                       nngrp['uiobjects']['lay1parfld'], \
                       nngrp['uiobjects']['lay2parfld'], \
                       nngrp['uiobjects']['lay3parfld'], \
                       nngrp['uiobjects']['lay4parfld'], \
                       nngrp['uiobjects']['lay5parfld'], \
                       nngrp['uiobjects']['buttonparfld'], \
                       svmgrp['uiobjects']['svmtyp'], \
                       svmgrp['uiobjects']['kernel'], \
                       svmgrp['uiobjects']['degree']
                       )
  else:
    if hasXGBoost():
      parsgrp = column(modeltyp, \
                       lineargrp['uiobjects']['lineartyp'], \
                       ensemblegrp['uiobjects']['ensembletyp'], \
                       ensemblegrp['uiobjects']['estparfldrf'], \
                       ensemblegrp['uiobjects']['depparfldrf'], \
                       ensemblegrp['uiobjects']['estparfldgb'], \
                       ensemblegrp['uiobjects']['depparfldgb'], \
                       ensemblegrp['uiobjects']['lrparfldgb'], \
                       ensemblegrp['uiobjects']['estparfldada'], \
                       ensemblegrp['uiobjects']['lrparfldada'], \
                       ensemblegrp['uiobjects']['estparfldxg'], \
                       ensemblegrp['uiobjects']['depparfldxg'], \
                       ensemblegrp['uiobjects']['lrparfldxg'], \
                       nngrp['uiobjects']['nntyp'], \
                       nngrp['uiobjects']['itrparfld'], \
                       nngrp['uiobjects']['lrparfld'], \
                       nngrp['uiobjects']['lay1parfld'], \
                       nngrp['uiobjects']['lay2parfld'], \
                       nngrp['uiobjects']['lay3parfld'], \
                       nngrp['uiobjects']['lay4parfld'], \
                       nngrp['uiobjects']['lay5parfld'], \
                       nngrp['uiobjects']['buttonparfld'], \
                       svmgrp['uiobjects']['svmtyp'], \
                       svmgrp['uiobjects']['kernel'], \
                       svmgrp['uiobjects']['degree']
                       )
    else:
      parsgrp = column(modeltyp, \
                       lineargrp['uiobjects']['lineartyp'], \
                       ensemblegrp['uiobjects']['ensembletyp'], \
                       ensemblegrp['uiobjects']['estparfldrf'], \
                       ensemblegrp['uiobjects']['depparfldrf'], \
                       ensemblegrp['uiobjects']['estparfldgb'], \
                       ensemblegrp['uiobjects']['depparfldgb'], \
                       ensemblegrp['uiobjects']['lrparfldgb'], \
                       ensemblegrp['uiobjects']['estparfldada'], \
                       ensemblegrp['uiobjects']['lrparfldada'], \
                       nngrp['uiobjects']['nntyp'], \
                       nngrp['uiobjects']['itrparfld'], \
                       nngrp['uiobjects']['lrparfld'], \
                       nngrp['uiobjects']['lay1parfld'], \
                       nngrp['uiobjects']['lay2parfld'], \
                       nngrp['uiobjects']['lay3parfld'], \
                       nngrp['uiobjects']['lay4parfld'], \
                       nngrp['uiobjects']['lay5parfld'], \
                       nngrp['uiobjects']['buttonparfld'], \
                       svmgrp['uiobjects']['svmtyp'], \
                       svmgrp['uiobjects']['kernel'], \
                       svmgrp['uiobjects']['degree']
                       )
  return {
    'grp' : parsgrp,
    'uiobjects': {
      'modeltyp': modeltyp,
      linearkey: lineargrp,
      'ensemblegrp': ensemblegrp,
      'nngrp': nngrp,
      'svmgrp': svmgrp
    }
  }

def getUiParams( sklearnpars ):
  sklearngrp = sklearnpars['uiobjects']
  modeltype = sklearngrp['modeltyp']
  if modeltype.value == 'Linear':
    parmobj = sklearngrp['lineargrp']['uiobjects']
    return getLinearPars( parmobj['lineartyp'].value )
  if modeltype.value == 'Logistic':
    parmobj = sklearngrp['loggrp']['uiobjects']
    return getLogPars( parmobj['logtyp'].value,parmobj['solvertyp'].value)
  if modeltype.value == 'Ensemble':
    parmobj = sklearngrp['ensemblegrp']['uiobjects']
    if parmobj['ensembletyp'].value == 'Random Forests':
      return getEnsembleParsRF( parmobj['ensembletyp'].value,parmobj['depparfldrf'].value,
                                          parmobj['estparfldrf'].value)
    elif parmobj['ensembletyp'].value == 'Gradient Boosting':
      return getEnsembleParsGB( parmobj['ensembletyp'].value,parmobj['depparfldgb'].value,
                                          parmobj['estparfldgb'].value,parmobj['lrparfldgb'].value)
    elif parmobj['ensembletyp'].value == 'Adaboost':
      return getEnsembleParsAda( parmobj['ensembletyp'].value,
                                 parmobj['estparfldada'].value,
                                 parmobj['lrparfldada'].value)
    elif parmobj['ensembletyp'].value == 'XGBoost: (Random Forests)':
      return getEnsembleParsXG( parmobj['ensembletyp'].value,
                                parmobj['depparfldxg'].value,
                                parmobj['estparfldxg'].value,
                                parmobj['lrparfldxg'].value )
  elif modeltype.value == 'Neural Network':
    parmobj = sklearngrp['nngrp']['uiobjects']
    return getNNPars( parmobj['nntyp'].value,
                      parmobj['itrparfld'].value,
                      parmobj['lrparfld'].value/1000,
                      parmobj['lay1parfld'].value,
                      parmobj['lay2parfld'].value,
                      parmobj['lay3parfld'].value,
                      parmobj['lay4parfld'].value,
                      parmobj['lay5parfld'].value,
                      sklearngrp['nngrp']['nb'].value)
  elif modeltype.value == 'SVM':
    parmobj = sklearngrp['svmgrp']['uiobjects']
    return getSVMPars( parmobj['svmtyp'].value,
                       parmobj['kernel'].value,
                       parmobj['degree'].value )
  return None
