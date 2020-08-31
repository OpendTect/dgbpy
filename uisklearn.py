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

def getLinearGrp(uipars=None):
  uiobjs = {}
  if not uipars:
    uiobjs = {'lineartyp': Select(title='Model', options=getUiLinearTypes()),}
    uipars = {'uiobjects': uiobjs, 'name': regmltypes[0][1],}
  else:
    uiobjs = uipars['uiobjects']
    
  uiobjs['lineartyp'].value = 'Ordinary Least Squares'
  return uipars

def getLogGrp(uipars=None):
  uiobjs = {}
  if not uipars:
    uiobjs = {'logtyp': Select(title='Model', options=getUiLogTypes()),
              'solvertyp': Select(title='Solver', options=getUiSolverTypes())
             }
    uipars = {'uiobjects': uiobjs, 'name': classmltypes[0][1],}
  else:
    uiobjs = uipars['uiobjects']
    
  uiobjs['logtyp'].value = 'Logistic Regression Classifier'
  uiobjs['solvertyp'].value = getDefaultSolver()
  return uipars

def getEnsembleGrp(uipars=None):
  uiobjs = {}
  rfgrp = {}
  if not uipars:
    rfgrp = getRFGrp()
    gbgrp = getGBGrp()
    adagrp = getAdaGrp()
    uiobjs = {'ensembletyp': Select(title='Model', options=getUiEnsembleTypes()),
              'depparfldrf': rfgrp['uiobjects']['depparfldrf'],
              'estparfldrf': rfgrp['uiobjects']['estparfldrf'],
              'depparfldgb': gbgrp['uiobjects']['depparfldgb'],
              'estparfldgb': gbgrp['uiobjects']['estparfldgb'], 
              'lrparfldgb': gbgrp['uiobjects']['lrparfldgb'],
              'estparfldada': adagrp['uiobjects']['estparfldada'],
              'lrparfldada': adagrp['uiobjects']['lrparfldada'],
              }
    ensemblegrp = (rfgrp,gbgrp,adagrp)
    if hasXGBoost():
      xggrp = getXGGrp()
      ensemblegrp += (xggrp,)
      xggrpuiobjs = xggrp['uiobjects']
      uiobjs.update({
                      'depparfldxg': xggrpuiobjs['depparfldxg'],
                      'estparfldxg': xggrpuiobjs['estparfldxg'],
                      'lrparfldxg': xggrpuiobjs['lrparfldxg'],
                    })
    uiobjs['ensembletyp'].on_change('value',partial(ensembleChgCB,cb=uiobjs['ensembletyp'],ensemblegrp=ensemblegrp))
    uipars = {'uiobjects': uiobjs, 'name': regmltypes[1][1],}      
  else:
    uiobjs  = uipars['uiobjects']
    rfgrp = {'uiobjects': {'depparfldrf': uiobjs['depparfldrf'], 'estparfldrf': uiobjs['estparfldrf'],},}
    gbgrp = {'uiobjects': {'depparfldgb': uiobjs['depparfldgb'],
                           'estparfldgb': uiobjs['estparfldgb'], 
                           'lrparfldgb': uiobjs['lrparfldgb'],},
            }
    adagrp = {'uiobjects': {'estparfldada': uiobjs['estparfldada'],
                            'lrparfldada': uiobjs['lrparfldada'],}
             }
    getRFGrp(rfgrp)
    getGBGrp(gbgrp)
    getAdaGrp(adagrp)
    if hasXGBoost():
      xggrp = {'uiobjects': {'depparfldxg': uiobjs['depparfldxg'],
                             'estparfldxg': uiobjs['estparfldxg'],
                             'lrparfldxg': uiobjs['lrparfldxg'],},
              }
      getXGGrp(xggrp)
    
  uiobjs['ensembletyp'].value = 'Random Forests'
  return uipars

def getNNGrp(uipars=None):
  dict = scikit_dict
  uiobjs = {}
  if not uipars:
    uiobjs = {'nntyp': Select(title='Model', options=getUiNNTypes()),
              'itrparfld': Slider(start=10,end=1000, title='Max Iteration'),
              'lay1parfld': Slider(start=1,end=200, title='Layer 1'),
              'lay2parfld': Slider(start=1,end=50, title='Layer 2'),
              'lay3parfld': Slider(start=1,end=10, title='Layer 3'),
              'lay4parfld': Slider(start=1,end=5, title='Layer 4'),
              'lay5parfld': Slider(start=1,end=3, title='Layer 5'),
              'lrparfld': Slider(start=1,end=100, title='Initial Learning Rate '+ '('+u'\u2030'+')'),
              'addbutton': Button(label='Add',button_type=defaultbut,width=but_width,height=but_height),
              'lessbutton': Button(label='Less',button_type=defaultbut,width=but_width,height=but_height),
              }
    uiobjs['buttonparfld'] = row(uiobjs['addbutton'],Spacer(width = 5),uiobjs['lessbutton'],sizing_mode='stretch_width')
    nb = Slider(start=1,end=5,value=3)
    layergrp = [nb,uiobjs['lay1parfld'],uiobjs['lay2parfld'],uiobjs['lay3parfld'],
                uiobjs['lay4parfld'],uiobjs['lay5parfld'],uiobjs['addbutton'],uiobjs['lessbutton']]
    try:
      uiobjs['lay1parfld'].value_throttled = uiobjs['lay1parfld'].value
      uiobjs['lay2parfld'].value_throttled = uiobjs['lay2parfld'].value
      uiobjs['lay3parfld'].value_throttled = uiobjs['lay3parfld'].value
      uiobjs['lay4parfld'].value_throttled = uiobjs['lay4parfld'].value
      uiobjs['lay1parfld'].on_change('value_throttled',partial(layer1ChgCB,layergrp))
      uiobjs['lay2parfld'].on_change('value_throttled',partial(layer2ChgCB,layergrp))
      uiobjs['lay3parfld'].on_change('value_throttled',partial(layer3ChgCB,layergrp))
      uiobjs['lay4parfld'].on_change('value_throttled',partial(layer4ChgCB,layergrp))
    except AttributeError:
      log_msg( '[WARNING] Bokeh version too old, consider updating it.' )
      pass
    uiobjs['addbutton'].on_click(partial(buttonChgCB,uiobjs['addbutton'],layergrp))
    uiobjs['lessbutton'].on_click(partial(buttonChgCB,uiobjs['lessbutton'],layergrp))
    uipars = {'uiobjects': uiobjs,     
              'name': regmltypes[2][1],
              'nb': nb
             }
  else:
    uiobjs = uipars['uiobjects']
    
  uiobjs['nntyp'].value = 'Multi-Layer Perceptron'
  uiobjs['itrparfld'].value = dict['nnpars']['maxitr']
  uiobjs['lay1parfld'].value = 50
  uiobjs['lay2parfld'].value = dict['nnpars']['lay2']
  uiobjs['lay3parfld'].value = dict['nnpars']['lay3']
  uiobjs['lay4parfld'].value = dict['nnpars']['lay4']
  uiobjs['lay5parfld'].value = dict['nnpars']['lay5']
  uiobjs['lrparfld'].value = dict['nnpars']['lr']*1000
  uipars['nb'].value = 3
  return uipars

def getSVMGrp(uipars=None):
  dict = scikit_dict
  uiobjs = {}
  defkernelstr = getDefaultNNKernel()
  if not uipars:
    uiobjs = {'svmtyp': Select(title='Model', options=getUiSVMTypes()),
              'kernel': Select(title='Kernel', options=getUiNNKernelTypes()),
              'degree': Slider(start=1,end=5, step=1,title='Degree'),
              }
    uiobjs['kernel'].on_change('value',partial(kernelChgCB,deg=uiobjs['degree']))
    kernelChgCB( 'value', defkernelstr, defkernelstr, uiobjs['degree'])
    uipars = {'uiobjects': uiobjs, 'name': regmltypes[3][1],}
  else:
    uiobjs = uipars['uiobjects']
    
  uiobjs['svmtyp'].value = 'Support Vector Machine'
  uiobjs['kernel'].value = defkernelstr
  uiobjs['degree'].value = dict['svmpars']['degree']
  return uipars

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

def getRFGrp(uipars=None):
  dict = scikit_dict['ensemblepars']['rf']
  uiobjs = {}
  if not uipars:
    uiobjs = {'estparfldrf': Slider(start=10, end=1000, step=10, title='Estimators'),
              'depparfldrf': Slider(start=10, end=200, step=10, title='Max Depth'),
              }
    uipars = {'uiobjects': uiobjs}
  else:
    uiobjs = uipars['uiobjects']
    
  uiobjs['estparfldrf'].value = dict['est']
  uiobjs['depparfldrf'].value = dict['maxdep']
  return uipars

def getGBGrp(uipars=None):
  dict = scikit_dict['ensemblepars']['gb']
  uiobjs = {}
  if not uipars:
    uiobjs = {'depparfldgb': Slider(start=1,end=100,step=1,title='Max Depth'),
              'estparfldgb': Slider(start=10,end=500,step=10,title='Estimators'),
              'lrparfldgb': Slider(start=0.1,end=10,step=0.1,title='Learning Rate'),
              }
    uipars = {'uiobjects': uiobjs,}
  else:
    uiobjs = uipars['uiobjects']
    
  uiobjs['depparfldgb'].value = dict['maxdep']
  uiobjs['estparfldgb'].value = dict['est']
  uiobjs['lrparfldgb'].value = dict['lr']
  return uipars

def getAdaGrp(uipars=None):
  dict = scikit_dict['ensemblepars']['ada']
  uiobjs = {}
  if not uipars:
    uiobjs = {'estparfldada': Slider(start=10,end=500,step=10,title='Estimators'),
              'lrparfldada': Slider(start=0.1,end=10,step=0.1,title='Learning Rate'),
              }
    uipars = {'uiobjects': uiobjs,}
  else:
    uiobjs = uipars['uiobjects']
    
  uiobjs['estparfldada'].value = dict['est']
  uiobjs['lrparfldada'].value = dict['lr']
  return uipars

def getXGGrp(uipars=None):
  if not hasXGBoost():
    return None
  dict = scikit_dict['ensemblepars']['xg']
  uiobjs = {}
  if not uipars:
    uiobjs = {'depparfldxg': Slider(start=10,end=200,step=10,title='Max Depth'),
              'estparfldxg': Slider(start=10,end=1000,step=10,title='Estimators'),
              'lrparfldxg': Slider(start=0.1,end=10,step=0.1,title='Learning Rate'),
              }
    uipars = {'uiobjects': uiobjs,}
  else:
    uiobjs = uipars['uiobjects']
    
  uiobjs['depparfldxg'].value = dict['maxdep']
  uiobjs['estparfldxg'].value = dict['est']
  uiobjs['lrparfldxg'].value = dict['lr']
  return uipars

def ensembleChgCB( attrnm, old, new, cb, ensemblegrp ):
  ret = uibokeh.getAllUiFlds( ensemblegrp )
  for uifld in  ret:
    uifld.visible = False
  ret = uibokeh.getGroup( new, cb.options, ensemblegrp, 'uiobjects' )
  for uifld in  ret:
    ret[uifld].visible = True

def getUiPars(isclassification, uipars=None):
  models = getUiModelTypes(isclassification)
  uiobjs = {}
  parsgrp = None
  linearkey = 'loggrp' if isclassification else 'lineargrp'
  deftype = classmltypes[0][1] if isclassification else regmltypes[1][1]
  
  if not uipars:
    uiobjs = {
      'modeltyp': Select(title='Type',options=models),
      linearkey: getLogGrp() if isclassification else getLinearGrp(),
      'ensemblegrp': getEnsembleGrp(),
      'nngrp': getNNGrp(),
      'svmgrp': getSVMGrp()
    }
    modelsgrp = (uiobjs['lineargrp'], uiobjs['ensemblegrp'], uiobjs['nngrp'], uiobjs['svmgrp'])
    uiobjs['modeltyp'].on_change('value',partial(modelChgCB,cb=uiobjs['modeltyp'],modelsgrp=modelsgrp))
    pars = [uiobjs['modeltyp']]
    ensemblepars = [uiobjs['ensemblegrp']['uiobjects']['ensembletyp'], \
                    uiobjs['ensemblegrp']['uiobjects']['estparfldrf'], \
                    uiobjs['ensemblegrp']['uiobjects']['depparfldrf'], \
                    uiobjs['ensemblegrp']['uiobjects']['estparfldgb'], \
                    uiobjs['ensemblegrp']['uiobjects']['depparfldgb'], \
                    uiobjs['ensemblegrp']['uiobjects']['lrparfldgb'], \
                    uiobjs['ensemblegrp']['uiobjects']['estparfldada'], \
                    uiobjs['ensemblegrp']['uiobjects']['lrparfldada'], \
                   ]
    xgpars = [uiobjs['ensemblegrp']['uiobjects']['estparfldxg'], \
              uiobjs['ensemblegrp']['uiobjects']['depparfldxg'], \
              uiobjs['ensemblegrp']['uiobjects']['lrparfldxg'], \
             ]
    nnpars = [uiobjs['nngrp']['uiobjects']['nntyp'], \
              uiobjs['nngrp']['uiobjects']['itrparfld'], \
              uiobjs['nngrp']['uiobjects']['lrparfld'], \
              uiobjs['nngrp']['uiobjects']['lay1parfld'], \
              uiobjs['nngrp']['uiobjects']['lay2parfld'], \
              uiobjs['nngrp']['uiobjects']['lay3parfld'], \
              uiobjs['nngrp']['uiobjects']['lay4parfld'], \
              uiobjs['nngrp']['uiobjects']['lay5parfld'], \
              uiobjs['nngrp']['uiobjects']['buttonparfld'], \
             ]
    svmpars = [uiobjs['svmgrp']['uiobjects']['svmtyp'], \
               uiobjs['svmgrp']['uiobjects']['kernel'], \
               uiobjs['svmgrp']['uiobjects']['degree'] \
              ]


    if isclassification:
      pars.extend([uiobjs[linearkey]['uiobjects']['logtyp'], uiobjs['lineargrp']['uiobjects']['solvertyp']])
    else:
      pars.extend([uiobjs[linearkey]['uiobjects']['lineartyp']])
    pars.extend(ensemblepars)
    if hasXGBoost():
      pars.extend(xgpars)
    pars.extend(nnpars)
    pars.extend(svmpars)
    parsgrp = column(*pars)
    uipars = {'grp': parsgrp, 'uiobjects': uiobjs}
  else:
    uiobjs = uipars['uiobjects']
    
  if isclassification:
    uiobjs['modeltyp'].value = models[0]
    uiobjs[linearkey] = getLogGrp(uiobjs[linearkey])
  else:
    uiobjs['modeltyp'].value = models[1]
    uiobjs[linearkey] = getLinearGrp(uiobjs[linearkey])
  uiobjs['ensemblegrp'] = getEnsembleGrp(uiobjs['ensemblegrp'])
  uiobjs['nngrp'] = getNNGrp(uiobjs['nngrp'])
  uiobjs['svmgrp'] = getSVMGrp(uiobjs['svmgrp'])

  modelsgrp = (uiobjs[linearkey], uiobjs['ensemblegrp'], uiobjs['nngrp'], uiobjs['svmgrp'])
  modelChgCB( 'value', deftype, deftype, uiobjs['modeltyp'], modelsgrp )
  return uipars

def getUiParams( sklearnpars ):
  sklearngrp = sklearnpars['uiobjects']
  modeltype = sklearngrp['modeltyp']
  if modeltype.value == 'Linear':
    parmobj = sklearngrp['lineargrp']['uiobjects']
    return getLinearPars( modelname=parmobj['lineartyp'].value )
  if modeltype.value == 'Logistic':
    parmobj = sklearngrp['loggrp']['uiobjects']
    return getLogPars( modelname=parmobj['logtyp'].value,
                       solver=parmobj['solvertyp'].value)
  if modeltype.value == 'Ensemble':
    parmobj = sklearngrp['ensemblegrp']['uiobjects']
    if parmobj['ensembletyp'].value == 'Random Forests':
      return getEnsembleParsRF( modelname=parmobj['ensembletyp'].value,
                                maxdep=parmobj['depparfldrf'].value,
                                est=parmobj['estparfldrf'].value)
    elif parmobj['ensembletyp'].value == 'Gradient Boosting':
      return getEnsembleParsGB( modelname=parmobj['ensembletyp'].value,
                                maxdep=parmobj['depparfldgb'].value,
                                est=parmobj['estparfldgb'].value,
                                lr=parmobj['lrparfldgb'].value)
    elif parmobj['ensembletyp'].value == 'Adaboost':
      return getEnsembleParsAda( modelname=parmobj['ensembletyp'].value,
                                 est=parmobj['estparfldada'].value,
                                 lr=parmobj['lrparfldada'].value)
    elif parmobj['ensembletyp'].value == 'XGBoost: (Random Forests)':
      return getEnsembleParsXG( modelname=parmobj['ensembletyp'].value,
                                maxdep=parmobj['depparfldxg'].value,
                                est=parmobj['estparfldxg'].value,
                                lr=parmobj['lrparfldxg'].value )
  elif modeltype.value == 'Neural Network':
    parmobj = sklearngrp['nngrp']['uiobjects']
    return getNNPars( modelname=parmobj['nntyp'].value,
                      maxitr=parmobj['itrparfld'].value,
                      lr=parmobj['lrparfld'].value/1000,
                      lay1=parmobj['lay1parfld'].value,
                      lay2=parmobj['lay2parfld'].value,
                      lay3=parmobj['lay3parfld'].value,
                      lay4=parmobj['lay4parfld'].value,
                      lay5=parmobj['lay5parfld'].value,
                      nb=sklearngrp['nngrp']['nb'].value)
  elif modeltype.value == 'SVM':
    parmobj = sklearngrp['svmgrp']['uiobjects']
    return getSVMPars( modelname=parmobj['svmtyp'].value,
                       kernel=parmobj['kernel'].value,
                       degree=parmobj['degree'].value )
  return None
