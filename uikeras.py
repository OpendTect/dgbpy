#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Apr 2019
#
# _________________________________________________________________________

from functools import partial

from bokeh.layouts import column
from bokeh.models.widgets import CheckboxGroup, Div, Select, Slider

from odpy.common import log_msg
from dgbpy.dgbkeras import *
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

def getUiPars():
  dict = keras_dict
  learntype = info[dgbkeys.learntypedictstr]
  modeltypes = getUiModelTypes( learntype )
  defmodel = modeltypes[0]
  modeltypfld = Select(title='Type',value=defmodel,
                       options=modeltypes )
  epochfld = Slider(start=1,end=1000,value=dict['epoch'],
              title='Epochs')
  defbatchsz = keras_dict['batch']
  if isUnet( defmodel ):
    defbatchsz = 4
  batchfld = Select(title='Batch Size',value=str(defbatchsz),options=cudacores)
  lrfld = Slider(start=1,end=100,value=dict['learnrate']*1000,
                 title='Initial Learning Rate '+ '('+u'\u2030'+')')
  edfld = Slider(start=1,end=100,value=100*dict['epochdrop']/epochfld.value,
                 title='Epoch drop (%)', step=0.1)
  patiencefld = Slider(start=1,end=100,value=dict['patience'],
                title='Patience')
  dodecimatefld = CheckboxGroup( labels=['Decimate input'], active=[] )
  chunkfld = Slider(start=1,end=100,value=dict['nbchunk'],
                    title='Number of Chunks')
  sizefld = None
  estimatedsz = info[dgbkeys.estimatedsizedictstr]
  if estimatedsz != None:
    sizefld = Div( text=getSizeStr( estimatedsz ) )
  decimateCB( dodecimatefld.active,chunkfld,sizefld )
  dodecimatefld.on_click(partial(decimateCB,chunkfld=chunkfld,sizefld=sizefld))
  try:
    chunkfld.value_throttled = chunkfld.value
    chunkfld.on_change('value_throttled',partial(chunkfldCB, sizefld))
  except AttributeError:
    log_msg( '[WARNING] Bokeh version too old, consider updating it.' )
    pass
  parsgrp = column(modeltypfld, \
                   batchfld,epochfld,patiencefld,lrfld,edfld,sizefld,dodecimatefld, \
                   chunkfld)
  return {
    'grp' : parsgrp,
    'uiobjects': {
      'modeltypfld': modeltypfld,
      'dodecimatefld': dodecimatefld,
      'sizefld': sizefld,
      'chunkfld': chunkfld,
      'epochfld': epochfld,
      'batchfld': batchfld,
      'patiencefld': patiencefld,
      'lrfld': lrfld,
      'edfld': edfld
    }
  }

def chunkfldCB(sizefld,attr,old,new):
  if sizefld == None:
    return
  sizefld.text = getSizeStr( info[dgbkeys.estimatedsizedictstr]/new )

def getUiParams( keraspars ):
  kerasgrp = keraspars['uiobjects']
  nrepochs = kerasgrp['epochfld'].value
  epochdroprate = kerasgrp['edfld'].value / 100
  epochdrop = int(nrepochs*epochdroprate)
  if epochdrop < 1:
    epochdrop = 1
  return getParams( dodec=doDecimate(kerasgrp['dodecimatefld']), \
                             nbchunk=kerasgrp['chunkfld'].value, \
                             epochs=kerasgrp['epochfld'].value, \
                             batch=int(kerasgrp['batchfld'].value), \
                             patience=kerasgrp['patiencefld'].value, \
                             learnrate=kerasgrp['lrfld'].value/1000, \
                             epochdrop=epochdrop, \
                             nntype=kerasgrp['modeltypfld'].value )

def doDecimate( fldwidget, index=0 ):
  return uibokeh.integerListContains( fldwidget.active, index )

def decimateCB( widgetactivelist,chunkfld,sizefld ):
  decimate = uibokeh.integerListContains( widgetactivelist, 0 )
  chunkfld.visible = decimate
  if sizefld == None:
    return
  size = info[dgbkeys.estimatedsizedictstr]
  if decimate:
    size /= chunkfld.value
  sizefld.text = getSizeStr( size )
