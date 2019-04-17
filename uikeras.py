#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Apr 2019
#
# _________________________________________________________________________

from functools import partial

from bokeh.layouts import column
from bokeh.models.widgets import CheckboxGroup, Slider
from bokeh.plotting import curdoc

import dgbpy.keystr as dgbkeys
from dgbpy import dgbkeras
from dgbpy import uibokeh

def getPlatformNm( full=False ):
  if full:
    return dgbkeras.platform
  return dgbkeras.getMLPlatform()

def getUiPars():
  dict = dgbkeras.keras_dict
  epochfld = Slider(start=1,end=100,value=dict['epoch'],step=1,
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
  return {
    'grp' : column(epochfld,batchfld,patiencefld,dodecimatefld,decimatefld,iterfld),
    'uiobjects': {
      'dodecimatefld': dodecimatefld,
      'decimatefld': decimatefld,
      'iterfld': iterfld,
      'epochfld': epochfld,
      'batchfld': batchfld,
      'patiencefld': patiencefld
    }
  }

def getParams( keraspars ):
  kerasgrp = keraspars['uiobjects']
  return dgbkeras.getParams( doDecimate(kerasgrp['dodecimatefld']), \
                             kerasgrp['decimatefld'].value/100, \
                             kerasgrp['iterfld'].value, \
                             kerasgrp['epochfld'].value, \
                             kerasgrp['batchfld'].value, \
                             kerasgrp['patiencefld'].value )

def doDecimate( fldwidget, index=0 ):
  return uibokeh.integerListContains( fldwidget.active, index )

def decimateCB( widgetactivelist, decimatefld, iterfld ):
  decimate = uibokeh.integerListContains( widgetactivelist, 0 )
  decimatefld.visible = decimate
  iterfld.visible = decimate
