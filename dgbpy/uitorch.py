from functools import partial

import numpy as np
from enum import Enum

from bokeh.layouts import column
from dgbpy.bokehcore import *

from odpy.common import log_msg
from dgbpy.transforms import hasOpenCV
import dgbpy.keystr as dgbkeys
from dgbpy import uibokeh
from dgbpy.dgbtorch import *

info = None

def getPlatformNm( full=False ):
  if full:
    return platform
  return getMLPlatform()

def getSizeStr( sizeinbytes ):
  ret = 'Size: '
  try:
    import humanfriendly
    ret += humanfriendly.format_size( sizeinbytes )
  except Exception as e:
    ret += str(int(sizeinbytes)) + ' bytes'
  return ret

def chunkfldCB(sizefld,attr,old,new):
  size = info[dgbkeys.estimatedsizedictstr]
  if sizefld and size:
    sizefld.text = getSizeStr( size/new )

def decimateCB(chunkfld,sizefld,attr,old,new):
  decimate = uibokeh.integerListContains( new, 0 )
  chunkfld.visible = decimate
  size = info[dgbkeys.estimatedsizedictstr]
  if sizefld and size:
    if decimate:
      size /= chunkfld.value
  sizefld.text = getSizeStr( size )

def getUiModelTypes( learntype, classification, ndim ):
  ret = ()
  for model in getModelsByType( learntype, classification, ndim ):
    ret += ((model,),)

  return dgbkeys.getNames( ret )


def getUiPars(uipars=None):
  dict = torch_dict
  learntype = info[dgbkeys.learntypedictstr]
  inpshape = info[dgbkeys.inpshapedictstr]
  if isinstance(inpshape, int):
    ndim = 1
  else:
    ndim = len(inpshape) - inpshape.count(1)
  modeltypes = getUiModelTypes( learntype, info[dgbkeys.classdictstr], ndim )

  if len(modeltypes)==0:
      divfld = Div(text="""No PyTorch models found for this workflow.""")
      parsgrp = column(divfld)
      return {'grp': parsgrp,
              'uiobjects':{
                'divfld': divfld
              }
            }

  defbatchsz = torch_dict['batch']
  defmodel = modeltypes[0]
  estimatedsz = info[dgbkeys.estimatedsizedictstr]
  isCrossVal = dgbhdf5.isCrossValidation(info)
  if isCrossVal:
    validfld = Slider(start=1,end=dgbhdf5.getNrGroupInputs(info),step=1,value=1,
                      title='Number of input for validation', margin=uibokeh.widget_margin)
  else:
    validfld = Slider(start=0.0,end=0.5,step=0.01,value=0.2,
                      title='Validation Percentage Split', margin=uibokeh.widget_margin)
  if tc.TorchUserModel.isImg2Img( defmodel ):
      defbatchsz = 4
  uiobjs = {}
  if not uipars:
    uiobjs = {
      'modeltypfld': Select(title='Type', options=modeltypes, width=300, margin=uibokeh.widget_margin),
      'validfld': validfld,
      'foldfld' : Slider(start=1,end=5,title='Number of fold(s)',visible=isCrossVal, margin=uibokeh.widget_margin),
      'batchfld': Select(title='Batch Size', options=cudacores, width=300, margin=uibokeh.widget_margin),
      'epochfld': Slider(start=1, end=1000, title='Epochs', margin=uibokeh.widget_margin),
      'patiencefld': Slider(start=1, end=100, title='Early Stopping', margin=uibokeh.widget_margin),
      'lrfld': Slider(start=-10,end=-1,step=1, title='Initial Learning Rate (1e)', margin=uibokeh.widget_margin),
      'edfld': Slider(start=1,end=100, title='Epoch drop (%)', step=0.1, margin=uibokeh.widget_margin),
      'sizefld': Div( text='Size: Unknown' , margin=uibokeh.widget_margin),
      'dodecimatefld': CheckboxGroup( labels=['Decimate input'] , margin=uibokeh.widget_margin),
      'chunkfld': Slider( start=1, end=100, title='Number of Chunks' , margin=uibokeh.widget_margin),
      'rundevicefld': CheckboxGroup( labels=['Train on GPU'], visible=can_use_gpu(), margin=uibokeh.widget_margin)
    }
    if estimatedsz:
      uiobjs['sizefld'] = Div( text=getSizeStr( estimatedsz ) )
    uiobjs['dodecimatefld'].on_change('active', partial(decimateCB, uiobjs['chunkfld'], uiobjs['sizefld']))
    try:
      uiobjs['chunkfld'].on_change('value_throttled', partial(chunkfldCB, uiobjs['sizefld']))
    except AttributeError:
      log_msg( '[WARNING] Bokeh version too old, consider updating it.' )
      pass
    parsgrp = column(*list(uiobjs.values()))
    uipars = {'grp': parsgrp, 'uiobjects': uiobjs}
  else:
    uiobjs = uipars['uiobjects']

  uiobjs['modeltypfld'].value = defmodel
  uiobjs['batchfld'].value = str(defbatchsz)
  uiobjs['epochfld'].value = dict['epochs']
  uiobjs['lrfld'].value = np.log10(dict['learnrate'])
  uiobjs['patiencefld'].value = dict['patience']
  uiobjs['edfld'].value = 100*dict['epochdrop']/uiobjs['epochfld'].value
  if estimatedsz:
    uiobjs['sizefld'].text = getSizeStr(estimatedsz)
  uiobjs['foldfld'].value = dict['nbfold']
  uiobjs['dodecimatefld'].active = []
  uiobjs['chunkfld'].value = dict['nbchunk']
  uiobjs['rundevicefld'].active = [0]
  decimateCB( uiobjs['chunkfld'],uiobjs['sizefld'], None,None,uiobjs['dodecimatefld'].active )
  return uipars

def getAdvancedUiPars(uipars=None):
  dict = torch_dict
  uiobjs={}
  if not uipars:
    uiobjs = {
      'tofp16fld': CheckboxGroup(labels=['Use Mixed Precision'], visible=can_use_gpu(), margin=(5, 5, 0, 5)),
      'tensorboardfld': CheckboxGroup(labels=['Enable Tensorboard'], visible=True, margin=(5, 5, 0, 5)),
      'cleartensorboardfld': CheckboxGroup(labels=['Clear Tensorboard log files'], visible=True, margin=(5, 5, 0, 5))
    }
    if not dgbhdf5.isLogInput(info):
      aug_labels = uibokeh.set_augment_mthds(info)
      transformUi = {
        'scalingheadfld' :Div(text="""<strong>Data Scaling</strong>""", height = 10),
        'scalingfld': RadioGroup(labels=[dgbkeys.globalstdtypestr, dgbkeys.localstdtypestr, dgbkeys.normalizetypestr, dgbkeys.minmaxtypestr],
                                active=0, margin = [5, 5, 5, 25]),
        'augmentheadfld': Div(text="""<strong>Data Augmentation</strong>""", height = 10),
        'augmentfld': CheckboxGroup(labels=aug_labels, visible=True, margin=(5, 5, 5, 25)),
      }
      uiobjs = {**transformUi, **uiobjs}

    parsgrp = column(*list(uiobjs.values()))
    uipars = {'grp':parsgrp, 'uiobjects':uiobjs}
  else:
    uiobjs = uipars['uiobjects']

  uiobjs['tofp16fld'].active = [] if not dict['tofp16'] else [0]
  uiobjs['tensorboardfld'].active = [] if not dict['withtensorboard'] else [0]
  uiobjs['cleartensorboardfld'].active = []
  return uipars     

def enableAugmentationCB(args, widget=None):
  widget.disabled =  uibokeh.integerListContains([widget.disabled], 0)

def getUiTransforms(advtorchgrp):
  transforms = []
  if 'augmentfld' in advtorchgrp:
    labels = advtorchgrp['augmentfld'].labels
    selectedkeys = advtorchgrp['augmentfld'].active
    for key in selectedkeys:
      selectedlabel = labels[key]
      transforms.append(uibokeh.augment_ui_map[selectedlabel])
  return transforms

def getUiScaler(advtorchgrp):
  scalers = (dgbkeys.globalstdtypestr, dgbkeys.localstdtypestr, dgbkeys.normalizetypestr, dgbkeys.minmaxtypestr)
  selectedOption = 0
  if 'scalingfld' in advtorchgrp:
    selectedOption = advtorchgrp['scalingfld'].active
  return scalers[selectedOption]

def getUiParams( torchpars, advtorchpars ):
  torchgrp = torchpars['uiobjects']     
  advtorchgrp = advtorchpars['uiobjects']
  nrepochs = torchgrp['epochfld'].value
  epochdroprate = torchgrp['edfld'].value / 100
  epochdrop = int(nrepochs*epochdroprate)
  patience = torchgrp['patiencefld'].value
  validation_split = torchgrp['validfld'].value
  nbfold = torch_dict['nbfold']
  if torchgrp['foldfld'].visible:
    nbfold = torchgrp['foldfld'].value
  if epochdrop < 1:
    epochdrop = 1
  runoncpu = not torchgrp['rundevicefld'].visible or \
             not isSelected( torchgrp['rundevicefld'] )
  scale = getUiScaler(advtorchgrp)
  transform = getUiTransforms(advtorchgrp)
  withtensorboard = True if len(advtorchgrp['tensorboardfld'].active)!=0 else False
  tofp16 = True if len(advtorchgrp['tofp16fld'].active)!=0 else False
  return getParams( dodec=isSelected(torchgrp['dodecimatefld']), \
                             nbchunk = torchgrp['chunkfld'].value, \
                             epochs=torchgrp['epochfld'].value, \
                             batch=int(torchgrp['batchfld'].value), \
                             learnrate= 10**torchgrp['lrfld'].value, \
                             nntype=torchgrp['modeltypfld'].value, \
                             patience=patience, \
                             epochdrop=epochdrop, \
                             validation_split = validation_split, \
                             nbfold= nbfold, \
                             prefercpu = runoncpu,
                             scale = scale,
                             transform=transform,
                             withtensorboard = withtensorboard,
                             tofp16 = tofp16,)

def isSelected( fldwidget, index=0 ):
  return uibokeh.integerListContains( fldwidget.active, index )
