#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Apr 2019
#
# _________________________________________________________________________

from functools import partial

from bokeh.layouts import column
from bokeh.models.widgets import Select, Slider

from dgbpy.dgbscikit import *
from dgbpy import uibokeh

def getPlatformNm( full=False ):
  if full:
    return platform
  return getMLPlatform()

def getLinearPars():
  lineartyp = Select(title='Model',options=getUiLinearTypes() )
  nbparfld = Slider(start=1,end=100,value=3,step=1,title='Number')
  #nbparfld = Slider(start=1,end=100,value=dict['nb'],step=1,title='Number')
  return {
    'grp': column(lineartyp,nbparfld),
    'uiobjects': {
      'lineartyp': lineartyp,
      'nbparfld': nbparfld
    }
  }

def getEnsemblePars():
  ensembletyp = Select(title='Model',options=getUiEnsembleTypes() )
  return {
    'grp': column(ensembletyp),
    'uiobjects': {
      'ensembletyp': ensembletyp
    }
  }

def getNNPars():
  nntyp = Select(title='Model',options=getUiNNTypes() )
  return {
    'grp': column(nntyp),
    'uiobjects': {
      'nntyp': nntyp
    }
  }

def modelChgCB( attrnm, old, new, cb, modelsgrp ):
  ret = uibokeh.getAllUiFlds( modelsgrp )
  for uifld in  ret:
    uifld.visible = False
  ret = uibokeh.getGroup( new, cb.options, modelsgrp, 'uiobjects' )
  for uifld in  ret:
    ret[uifld].visible = True

def getUiPars():
  dict = scikit_dict
  modeltyp = Select(title='Type',options=getUiModelTypes() )
  lineargrp = getLinearPars()
  ensemblegrp = getEnsemblePars()
  nngrp = getNNPars()
  modelsgrp = (lineargrp,ensemblegrp,nngrp)
  modeltyp.on_change('value',partial(modelChgCB,cb=modeltyp,modelsgrp=modelsgrp))
  modelChgCB( 'value', mltypes[0][1], mltypes[0][1], modeltyp, modelsgrp )
  allflds = uibokeh.getAllUiFlds( modelsgrp )
  parsgrp = column(modeltyp, \
                   lineargrp['uiobjects']['lineartyp'], \
                   lineargrp['uiobjects']['nbparfld'], \
                   ensemblegrp['uiobjects']['ensembletyp'], \
                   nngrp['uiobjects']['nntyp'] \
                  )
  return {
    'grp' : parsgrp,
    'uiobjects': {
      'modeltyp': modeltyp,
      'lineargrp': lineargrp,
      'ensemblegrp': ensemblegrp,
      'nngrp': nngrp,
    }
  }

def getUiParams( sklearnpars ):
  sklearngrp = sklearnpars['uiobjects']
  return None
  #return dgbscikit.getParams( sklearngrp['nbparfld'].value )
