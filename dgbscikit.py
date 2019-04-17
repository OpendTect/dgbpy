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
mltypes = [('linear','Linear'),('ensemble','Ensemble'),('neuralnet','Neural Network')]
lineartypes = [('oslq','Ordinary Least Squares')]
ensembletypes = [('randfor','Random Forests')]
nntypes = [('mlp','Multi-Layer Perceptron')]

def getMLPlatform():
  return platform[0]

def getUIMLPlatform():
  return platform[1]

def getUiModelTypes():
  return dgbkeys.getNames( mltypes )

def getUiLinearTypes():
  return dgbkeys.getNames( lineartypes )

def getUiEnsembleTypes():
  return dgbkeys.getNames( ensembletypes )

def getUiNNTypes():
  return dgbkeys.getNames( nntypes )

scikit_dict = {
  'nb': 3
}

def getParams( nb=scikit_dict['nb'] ):
  return {
    'decimation': False,
    'number': nb
  }

