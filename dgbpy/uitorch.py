from functools import partial

import numpy as np

from bokeh.layouts import column
from bokeh.models.widgets import CheckboxGroup, Div, Select, Slider

from odpy.common import log_msg
from dgbpy.dgbtorch import *
import dgbpy.keystr as dgbkeys
from dgbpy import uibokeh


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


def getUiModelTypes( learntype, classification, ndim ):
  ret = ()
  for model in getModelsByType( learntype, classification, ndim ):
    ret += ((model,),)

  return dgbkeys.getNames( ret )


def getUiPars(uipars=None):
  dict = torch_dict
  learntype = info[dgbkeys.learntypedictstr]
  if isinstance(info[dgbkeys.inpshapedictstr], int):
    ndim = 1
  else:
    ndim = len(info[dgbkeys.inpshapedictstr])
  modeltypes = getUiModelTypes( learntype, info[dgbkeys.classdictstr], ndim )

  if len(modeltypes)==0:
      divfld = Div(text="""No PyTorch models found for this workflow.""")
      parsgrp = column(divfld)
      return {'grp': parsgrp,
              'uiobjects':{
                'divfld': divfld
              }
            }

  defbatchsz = torch_dict['batch_size']
  defmodel = modeltypes[0]
  estimatedsz = info[dgbkeys.estimatedsizedictstr]
  if tc.TorchUserModel.isImg2Img( defmodel ):
      defbatchsz = 4
  uiobjs = {}
  if not uipars:
    uiobjs = {
      'modeltypfld': Select(title='Type', options=modeltypes),
      'batchfld': Select(title='Batch Size',options=cudacores),
      'epochfld': Slider(start=1,end=1000, title='Epochs'),
      'epochdrop': Slider(start=1, end=100, title='Early Stopping'),
      'lrfld': Slider(start=-10,end=-1,step=1, title='Initial Learning Rate (1e)'),
    }
    if estimatedsz:
      uiobjs['sizefld'] = Div( text=getSizeStr( estimatedsz ) )
    parsgrp = column(*list(uiobjs.values()))
    uipars = {'grp': parsgrp, 'uiobjects': uiobjs}
  else:
    uiobjs = uipars['uiobjects']

  uiobjs['modeltypfld'].value = defmodel
  uiobjs['batchfld'].value = str(defbatchsz)
  uiobjs['epochfld'].value = dict['epochs']
  uiobjs['lrfld'].value = np.log10(dict['learnrate'])
  uiobjs['epochdrop'].value = dict['epochdrop']
  if estimatedsz:
    uiobjs['sizefld'].text = getSizeStr(estimatedsz)
  return uipars

def getAdvancedUiPars(uipars=None):
  dict = torch_dict
  uiobjs={}
  if not uipars:
    uiobjs = {
      'augmentfld': CheckboxGroup(labels=['Enable Data Augmentation'], visible=True, margin=(5, 5, 0, 5)),
    }
    parsgrp = column(*list(uiobjs.values()))
    uipars = {'grp':parsgrp, 'uiobjects':uiobjs}
  else:
    uiobjs = uipars['uiobjects']
  uiobjs['augmentfld'].active = [] if not dict['transform'] else [0]
  return uipars

def getUiParams( torchpars, advtorchpars ):
  torchgrp = torchpars['uiobjects']
  advtorchgrp = advtorchpars['uiobjects']
  nrepochs = torchgrp['epochfld'].value
  epochdroprate = torchgrp['epochfld'].value / 100
  epochdrop = int(nrepochs*epochdroprate)
  if epochdrop < 1:
    epochdrop = 1
  transform = ['RandomRotation'] if len(advtorchgrp['augmentfld'].active)!=0 else False
  return getParams( epochs=torchgrp['epochfld'].value, \
                             batch=int(torchgrp['batchfld'].value), \
                             learnrate= 10**torchgrp['lrfld'].value, \
                             nntype=torchgrp['modeltypfld'].value, \
                             epochdrop=torchgrp['epochdrop'].value, \
                             transform=transform)

def isSelected( fldwidget, index=0 ):
  return uibokeh.integerListContains( fldwidget.active, index )
