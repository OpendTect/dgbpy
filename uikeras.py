#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Apr 2019
#
# _________________________________________________________________________

from functools import partial

from bokeh.layouts import column
from bokeh.models.widgets import CheckboxGroup, Select, Slider

from dgbpy.dgbkeras import *
from dgbpy import uibokeh

def getPlatformNm( full=False ):
  if full:
    return platform
  return getMLPlatform()

def getUiPars():
  dict = keras_dict
  modeltypfld = Select(title='Type',value=getUiModelTypes()[0],
                       options=getUiModelTypes() )
  epochfld = Slider(start=1,end=1000,value=dict['epoch'],
              title='Epochs')
  batchfld = Select(title='Number of Batch',value=cudacores[2],
                    options=cudacores)
  lrfld = Slider(start=1,end=100,value=dict['learnrate']*1000,
                 title='Initial Learning Rate '+ '('+u'\u2030'+')')
  edfld = Slider(start=1,end=100,value=100*dict['epochdrop']/epochfld.value,
                 title='Epoch drop (%)', step=0.1)
  patiencefld = Slider(start=1,end=100,value=dict['patience'],
                title='Patience')
  dodecimatefld = CheckboxGroup( labels=['Decimate input'], active=[] )
  decimatefld = Slider(start=0.1,end=99.9,value=dict['dec']*100, step=0.1,
                title='Decimation (%)')
  iterfld = Slider(start=1,end=100,value=dict['iters'],
              title='Iterations')
  decimateCB( dodecimatefld.active, decimatefld, iterfld )
  dodecimatefld.on_click(partial(decimateCB,decimatefld=decimatefld,iterfld=iterfld))
  parsgrp = column(modeltypfld, \
                   batchfld,epochfld,patiencefld,lrfld,edfld,dodecimatefld, \
                   decimatefld,iterfld)
  return {
    'grp' : parsgrp,
    'uiobjects': {
      'modeltypfld': modeltypfld,
      'dodecimatefld': dodecimatefld,
      'decimatefld': decimatefld,
      'iterfld': iterfld,
      'epochfld': epochfld,
      'batchfld': batchfld,
      'patiencefld': patiencefld,
      'lrfld': lrfld,
      'edfld': edfld
    }
  }

def getUiParams( keraspars ):
  kerasgrp = keraspars['uiobjects']
  nrepochs = kerasgrp['epochfld'].value
  epochdroprate = kerasgrp['edfld'].value / 100
  epochdrop = int(nrepochs*epochdroprate)
  if epochdrop < 1:
    epochdrop = 1
  return getParams( doDecimate(kerasgrp['dodecimatefld']), \
                             kerasgrp['decimatefld'].value/100, \
                             kerasgrp['iterfld'].value, \
                             kerasgrp['epochfld'].value, \
                             int(kerasgrp['batchfld'].value), \
                             kerasgrp['patiencefld'].value, \
                             kerasgrp['lrfld'].value/1000, \
                             epochdrop, \
                             kerasgrp['modeltypfld'].value )

def doDecimate( fldwidget, index=0 ):
  return uibokeh.integerListContains( fldwidget.active, index )

def decimateCB( widgetactivelist, decimatefld, iterfld ):
  decimate = uibokeh.integerListContains( widgetactivelist, 0 )
  decimatefld.visible = decimate
  iterfld.visible = decimate
