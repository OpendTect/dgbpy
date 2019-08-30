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
  modeltypfld = Select(title='Type',options=getUiModelTypes() )
  epochfld = Slider(start=1,end=1000,value=dict['epoch'],step=1,
              title='Epochs')
  batchfld = Slider(start=1,end=100,value=dict['batch'],step=1,
            title='Number of Batch')
  patiencefld = Slider(start=1,end=100,value=dict['patience'],step=1,
                title='Patience')
  dodecimatefld = CheckboxGroup( labels=['Decimate input'], active=[] )
  decimatefld = Slider(start=0.1,end=99.9,value=dict['dec']*100, step=0.1,
                title='Decimation (%)')
  iterfld = Slider(start=1,end=100,value=dict['iters'],step=1,
              title='Iterations')
  decimateCB( dodecimatefld.active, decimatefld, iterfld )
  dodecimatefld.on_click(partial(decimateCB,decimatefld=decimatefld,iterfld=iterfld))
  parsgrp = column(modeltypfld, \
                   epochfld,batchfld,patiencefld,dodecimatefld, \
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
      'patiencefld': patiencefld
    }
  }

def getUiParams( keraspars ):
  kerasgrp = keraspars['uiobjects']
  return getParams( doDecimate(kerasgrp['dodecimatefld']), \
                             kerasgrp['decimatefld'].value/100, \
                             kerasgrp['iterfld'].value, \
                             kerasgrp['epochfld'].value, \
                             kerasgrp['batchfld'].value, \
                             kerasgrp['patiencefld'].value,
                             kerasgrp['modeltypfld'].value )

def doDecimate( fldwidget, index=0 ):
  return uibokeh.integerListContains( fldwidget.active, index )

def decimateCB( widgetactivelist, decimatefld, iterfld ):
  decimate = uibokeh.integerListContains( widgetactivelist, 0 )
  decimatefld.visible = decimate
  iterfld.visible = decimate
