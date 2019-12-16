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

from dgbpy.dgbkeras import *
from dgbpy import uibokeh

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

def getUiPars(learntype,estimatedszgb=None):
  dict = keras_dict
  modeltypes = getUiModelTypes( learntype )
  modeltypfld = Select(title='Type',value=modeltypes[0],
                       options=modeltypes )
  epochfld = Slider(start=1,end=1000,value=dict['epoch'],
              title='Epochs')
  batchfld = Select(title='Batch Size',value=cudacores[2],
                    options=cudacores)
  lrfld = Slider(start=1,end=100,value=dict['learnrate']*1000,
                 title='Initial Learning Rate '+ '('+u'\u2030'+')')
  edfld = Slider(start=1,end=100,value=100*dict['epochdrop']/epochfld.value,
                 title='Epoch drop (%)', step=0.1)
  patiencefld = Slider(start=1,end=100,value=dict['patience'],
                title='Patience')
  dodecimatefld = CheckboxGroup( labels=['Decimate input'], active=[] )
  chunkfld = Slider(start=1,end=100,value=dict['nbchunk'],value_throttled=dict['nbchunk'],
                    title='Number of Chunks',callback_policy='mouseup')
  sizefld = None
  if estimatedszgb != None:
    sizefld = Div( text=getSizeStr(estimatedszgb) )
  decimateCB( dodecimatefld.active,chunkfld,sizefld, estimatedszgb )
  dodecimatefld.on_click(partial(decimateCB,chunkfld=chunkfld,sizefld=sizefld,
                                 estimatedszgb=estimatedszgb))
  chunkfld.on_change('value_throttled',partial(chunkfldCB, sizefld, estimatedszgb))
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

def chunkfldCB(sizefld,datasize,attr,old,new):
  if sizefld == None:
    return
  sizefld.text = getSizeStr( datasize/new )

def getUiParams( keraspars ):
  kerasgrp = keraspars['uiobjects']
  nrepochs = kerasgrp['epochfld'].value
  epochdroprate = kerasgrp['edfld'].value / 100
  epochdrop = int(nrepochs*epochdroprate)
  if epochdrop < 1:
    epochdrop = 1
  return getParams( doDecimate(kerasgrp['dodecimatefld']), \
                             kerasgrp['chunkfld'].value, \
                             kerasgrp['epochfld'].value, \
                             int(kerasgrp['batchfld'].value), \
                             kerasgrp['patiencefld'].value, \
                             kerasgrp['lrfld'].value/1000, \
                             epochdrop, \
                             kerasgrp['modeltypfld'].value )

def doDecimate( fldwidget, index=0 ):
  return uibokeh.integerListContains( fldwidget.active, index )

def decimateCB( widgetactivelist,chunkfld,sizefld,estimatedszgb ):
  decimate = uibokeh.integerListContains( widgetactivelist, 0 )
  chunkfld.visible = decimate
  if sizefld == None:
    return
  size = estimatedszgb
  if decimate:
    size /= chunkfld.value
  sizefld.text = getSizeStr( size )
