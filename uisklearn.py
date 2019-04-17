#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Apr 2019
#
# _________________________________________________________________________

from bokeh.layouts import column
from bokeh.models.widgets import Slider

from dgbpy import dgbscikit

def getPlatformNm( full=False ):
  if full:
    return dgbscikit.platform
  return dgbscikit.getMLPlatform()

def getUiPars():
  dict = dgbscikit.scikit_dict
  nbparfld = Slider(start=1,end=100,value=dict['nb'],step=1,title='Number')
  return {
    'grp' : column(nbparfld),
    'uiobjects': {
      'nbparfld': nbparfld
    }
  }

def getParams( sklearnpars ):
  sklearngrp = sklearnpars['uiobjects']
  return dgbscikit.getParams( sklearngrp['nbparfld'].value )
