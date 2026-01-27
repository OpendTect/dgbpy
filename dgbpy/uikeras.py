#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Apr 2019
#
# _________________________________________________________________________

from functools import partial

import numpy as np
from enum import Enum

from dgbpy.bokehcore import *

from odpy.common import log_msg
import dgbpy.keystr as dgbkeys
from dgbpy.dgbkeras import *
from dgbpy.transforms import all_scalers, all_scalers_opts, get_scaler_ui_option
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
  plfdict = keras_dict
  learntype = info[dgbkeys.learntypedictstr]
  inpshape = info[dgbkeys.inpshapedictstr]
  if isinstance(inpshape, int):
    ndim = 1
  else:
    ndim = len(inpshape) - inpshape.count(1)
  modeltypes = getUiModelTypes( learntype, info[dgbkeys.classdictstr], ndim )

  if len(modeltypes)==0:
    divfld = Div(text="""No Keras models found for this workflow.""")
    parsgrp = column(divfld)
    return {'grp': parsgrp,
            'uiobjects':{
              'divfld': divfld
            }
           }

  defmodel = modeltypes[0]
  defbatchsz = plfdict['batch']
  estimatedsz = info[dgbkeys.estimatedsizedictstr]
  userandomseed = plfdict[dgbkeys.userandomseeddictstr]
  isCrossVal = dgbhdf5.isCrossValidation(info)
  if isCrossVal:
    validsld = Slider(start=1,end=dgbhdf5.getNrGroupInputs(info),step=1,value=1,
                      title='Number of input for validation', width=220)
  else:
    validsld = Slider(start=0.0,end=0.5,step=0.01,value=0.2,
                      title='Validation Split', width=220)
  if kc.UserModel.isImg2Img( defmodel ):
      defbatchsz = 4
  uiobjs = {}
  if not uipars:
    uiobjs = {
      'modeltypfld': Select(title='Type', options=modeltypes, width=300, margin=uibokeh.widget_margin),
      'validsld': validsld,
      'validinp': TextInput(value=str(plfdict['split']), title='', width=60),
      'foldfld' : Slider(start=1,end=5,title='Number of fold(s)',visible=isCrossVal, margin=uibokeh.widget_margin),
      'batchfld': Select(title='Batch Size', options=cudacores, width=300, margin=uibokeh.widget_margin),
      'epochsld': Slider(start=1, end=1000, title='Epochs', width=220),
      'epochinp': TextInput(value=str(plfdict['epochs']), title='', width=60),
      'patiencesld': Slider(start=1, end=100, title='Early Stopping', width=220),
      'patienceinp': TextInput(value=str(plfdict['patience']), title='', width=60),
      'lrsld': Slider(start=-10,end=-1,step=1, title='Initial Learning Rate (1e)', width=220),
      'lrinp': TextInput(value=str(np.log10(plfdict['learnrate'])), title='', width=60),
      'edsld': Slider(start=1,end=100, title='Epoch drop (%)', step=0.1, width=220),
      'edinp': TextInput(value=str(plfdict['epochdrop']), title='', width=60),
      'sizefld': Div( text='Size: Unknown' , margin=uibokeh.widget_margin),
      'userandomseedfld': TextInput( title='Random seed', value=str(userandomseed), margin=uibokeh.widget_margin ),
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

    def syncFunctions( sliderKey, inputKey ):
      def sliderToInput( attr, old, new ):
          uiobjs[inputKey].value = str(new)

      def inputToSlider( attr, old, new ):
          if new.isdigit():
              val = int(new)
              slider = uiobjs[sliderKey]
              if slider.start <= val <= slider.end:
                  slider.value = val

      return sliderToInput, inputToSlider

    uikeys = [('validsld', 'validinp', 'validfld'),
              ('epochsld', 'epochinp', 'epochfld'),
              ('patiencesld', 'patienceinp', 'patiencefld'),
              ('lrsld', 'lrinp', 'lrfld'),
              ('edsld', 'edinp', 'edfld')]

    for sliderKey, inputKey, fieldKey in uikeys:
      sliderHandler, inputHandler = syncFunctions(sliderKey, inputKey)
      uiobjs[sliderKey].on_change('value', sliderHandler)
      uiobjs[inputKey].on_change('value', inputHandler)
      uiobjs[fieldKey] = row(uiobjs[sliderKey], uiobjs[inputKey], margin=uibokeh.widget_margin)

    parsobj = []
    for key in ['modeltypfld', 'validfld', 'foldfld', 'batchfld', 'epochfld',
                'patiencefld', 'lrfld', 'edfld', 'sizefld', 'userandomseedfld',
                'dodecimatefld', 'chunkfld', 'rundevicefld']:
      parsobj.append(uiobjs[key])
    parsgrp = column(*parsobj)
    uipars = {'grp': parsgrp, 'uiobjects': uiobjs}
  else:
    uiobjs = uipars['uiobjects']

  uiobjs['modeltypfld'].value = defmodel
  uiobjs['batchfld'].value = str(defbatchsz)
  uiobjs['epochsld'].value = plfdict['epochs']
  uiobjs['epochinp'].value = str(uiobjs['epochsld'].value)
  uiobjs['lrsld'].value = np.log10(plfdict['learnrate'])
  uiobjs['lrinp'].value = str(uiobjs['lrsld'].value)
  uiobjs['patiencesld'].value = plfdict['patience']
  uiobjs['patienceinp'].value = str(uiobjs['epochsld'].value)
  uiobjs['edsld'].value = 100*plfdict['epochdrop']/uiobjs['epochsld'].value
  uiobjs['edinp'].value = str(uiobjs['edsld'].value)
  if estimatedsz:
    uiobjs['sizefld'].text = getSizeStr(estimatedsz)
  uiobjs['foldfld'].value = plfdict['nbfold']
  uiobjs['dodecimatefld'].active = []
  uiobjs['chunkfld'].value = plfdict['nbchunk']
  uiobjs['rundevicefld'].active = [0]
  decimateCB( uiobjs['chunkfld'],uiobjs['sizefld'], None, None, uiobjs['dodecimatefld'].active )
  return uipars

def setup_scaler_ui(info):
  scaler = dgbhdf5.getScalerStr(info)
  if 'disable_scaling' in info and info['disable_scaling']:
    return True, scaler
  return False, scaler

def createAdvancedUiLeftPane():
  disable_scaling, scaler = setup_scaler_ui(info)
  uiobjs = {
    'tofp16fld': CheckboxGroup(labels=['Use Mixed Precision'], visible=can_use_gpu(), \
                                disabled= not is_mixed_precision_compatible(), margin=(5, 5, 0, 5)),
    'tensorboardheadfld': Div(text="""<strong>Tensorboard Options</strong>""", height = 10),
    'tensorboardfld': CheckboxGroup(labels=['Enable Tensorboard'], visible=True, margin=(5, 5, 0, 5)),
    'cleartensorboardfld': CheckboxGroup(labels=['Clear Tensorboard log files'], visible=True, margin=(5, 5, 0, 5))
  }

  tblogdir = keras_dict['tblogdir']
  if tblogdir != None:
    tbbuttonsgrp, tbobjs = uibokeh.getTBButtonBar( tblogdir )

    def getTensorboardButton(attr, old, new):
      is_checked = len(uiobjs['tensorboardfld'].active) > 0
      tbobjs['start'].visible = is_checked

    uiobjs['tensorboardfld'].on_change('active', getTensorboardButton)
    uiobjs['tensorboardrow'] = tbbuttonsgrp
  else:
    uiobjs['tensorboardrow'] = Div(text="TensorBoard is disabled.", visible=False)

  if not dgbhdf5.isLogInput(info):
    aug_labels = uibokeh.set_augment_mthds(info)
    transformUi = {
      'scalingheadfld' :Div(text="""<strong>Data Scaling</strong>""", height = 10),
      'scalingfld': RadioGroup(labels=list(all_scalers_opts), active=get_scaler_ui_option(scaler),
                               margin = [5, 5, 5, 25], disabled=disable_scaling),
      'augmentheadfld': Div(text="""<strong>Data Augmentation</strong>""", height = 10),
      'augmentfld': CheckboxGroup(labels=aug_labels, visible=True, margin=(5, 5, 5, 25)),
    }
    uiobjs = {**transformUi, **uiobjs}

  parsgrp = column(
    *[uiobjs[key] for key in uiobjs if key not in ['tofp16fld', 'tensorboardheadfld', 'tensorboardfld', 'tensorboardrow', 'cleartensorboardfld']],
    uiobjs['tofp16fld'],
    uiobjs['tensorboardheadfld'],
    uiobjs['tensorboardfld'],
    uiobjs['tensorboardrow'],           
    uiobjs['cleartensorboardfld']
  )
  return parsgrp, uiobjs

def getSaveTypes( exclude=[None] ):
  return [ _type.name for _type in SaveType if _type.value not in exclude]

def createAdvancedUiRightPane():
  uiobjs = {
    'savetypehead': Div(text="""<strong>Save Format</strong>""", height = 10),
    'savetypefld': RadioGroup(labels=getSaveTypes(), active=0, margin = [5, 5, 5, 25]),
  }
  parsgrp = column(*list(uiobjs.values()))
  return parsgrp, uiobjs

def getAdvancedUiPars(uipars=None):
  plfdict = keras_dict
  uiobjs={}
  if not uipars:
    parsgrp, uiobjs = createAdvancedUiLeftPane()
    parsgrp2, uiobjs2 = createAdvancedUiRightPane()
    parsgrp = row(parsgrp, parsgrp2)
    uiobjs = {**uiobjs, **uiobjs2}
    uipars = {'grp': parsgrp, 'uiobjects': uiobjs}
  else:
    uiobjs = uipars['uiobjects']

  uiobjs['tofp16fld'].active = [] if not plfdict['tofp16'] else [0]
  uiobjs['tensorboardfld'].active = [] if not plfdict['withtensorboard'] else [0]
  uiobjs['cleartensorboardfld'].active = []
  uiobjs['savetypefld'].active = list(SaveType).index( SaveType(defsavetype) )
  return uipars

def getUiTransforms(advkerasgrp):
  transforms = []
  if 'augmentfld' in advkerasgrp:
    labels = advkerasgrp['augmentfld'].labels
    selectedkeys = advkerasgrp['augmentfld'].active
    for key in selectedkeys:
      selectedlabel = labels[key]
      transforms.append(uibokeh.augment_ui_map[selectedlabel])
  return transforms

def getUiScaler(advkerasgrp):
  selectedOption = 0
  if 'scalingfld' in advkerasgrp:
    selectedOption = advkerasgrp['scalingfld'].active
  return all_scalers[selectedOption]

def getUiParams( keraspars, advkeraspars ):
  kerasgrp = keraspars['uiobjects']
  advkerasgrp = advkeraspars['uiobjects']
  nrepochs = kerasgrp['epochsld'].value
  epochdroprate = kerasgrp['edsld'].value / 100
  epochdrop = int(nrepochs*epochdroprate)
  patience = kerasgrp['patiencesld'].value
  validation_split = kerasgrp['validsld'].value
  savetype = list(SaveType)[advkerasgrp['savetypefld'].active].value
  nbfold = keras_dict['nbfold']
  if kerasgrp['foldfld'].visible:
    nbfold = kerasgrp['foldfld'].value
  if epochdrop < 1:
    epochdrop = 1
  runoncpu = not kerasgrp['rundevicefld'].visible or \
             not isSelected( kerasgrp['rundevicefld'] )
  try:
    seed = int(kerasgrp['userandomseedfld'].value)
  except:
    seed = keras_dict[dgbkeys.userandomseeddictstr]
  scale = getUiScaler(advkerasgrp)
  transform = getUiTransforms(advkerasgrp)
  withtensorboard = True if len(advkerasgrp['tensorboardfld'].active)!=0 else False
  tofp16 = bool(advkerasgrp['tofp16fld'].active) and not advkerasgrp['tofp16fld'].disabled
  return getParams( dodec=isSelected(kerasgrp['dodecimatefld']),
                             nbchunk=kerasgrp['chunkfld'].value,
                             epochs=kerasgrp['epochsld'].value,
                             batch=int(kerasgrp['batchfld'].value),
                             patience=patience,
                             learnrate= 10 ** kerasgrp['lrsld'].value,
                             epochdrop=epochdrop,
                             validation_split = validation_split,
                             savetype=savetype,
                             nbfold=nbfold,
                             nntype=kerasgrp['modeltypfld'].value,
                             prefercpu=runoncpu,
                             scale=scale,
                             transform=transform,
                             withtensorboard=withtensorboard,
                             tblogdir=None,
                             tofp16=tofp16,
                             userandomseed=seed,
                             stopaftercurrentepoch = False)

def isSelected( fldwidget, index=0 ):
  return uibokeh.integerListContains( fldwidget.active, index )
