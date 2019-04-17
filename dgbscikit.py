#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Jan 2019
#
# _________________________________________________________________________
# various tools machine learning using Sci-kit platform
#

import dgbpy.keystr as dgbkeys


platform = (dgbkeys.scikitplfnm,'Scikit-learn')
mltypes = ['Linear','Regressor','MLP']

def getMLPlatform():
  return platform[0]

def getUIMLPlatform():
  return platform[1]

scikit_dict = {
  'nb': 3
}

def getParams( nb=scikit_dict['nb'] ):
  return {
    'decimation': False,
    'number': nb
  }

