#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Jan 2019
#
# _________________________________________________________________________
# various tools machine learning using Sci-kit platform
#

from os.path import splitext
import h5py
import json
import joblib
import numpy as np
import pickle

import sklearn
from sklearn.preprocessing import StandardScaler

from odpy.common import log_msg
import odpy.hdf5 as odhdf5
import dgbpy.keystr as dgbkeys
from dgbpy import hdf5 as dgbhdf5

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

def save( model, inpfnm, outfnm, save_type='joblib', scaler=None ):
  log_msg( 'Saving model.' )
  h5file = h5py.File( outfnm, 'w' )
  odhdf5.setAttr( h5file, 'backend', 'scikit-learn' )
  odhdf5.setAttr( h5file, 'sklearn_version', sklearn.__version__ )
  odhdf5.setAttr( h5file, 'type', 'RandomForestRegressor' )
  odhdf5.setAttr( h5file, 'model_config', json.dumps(model.get_params()) )
  modelgrp = h5file.create_group( 'model' )
  odhdf5.setAttr( modelgrp, 'type', save_type )
  if save_type == 'pickle':
    exported_modelstr = pickle.dumps(model)
    exported_model = np.frombuffer( exported_modelstr, dtype='S1', count=len(exported_modelstr) )
    modelgrp.create_dataset('object',data=exported_model)
  elif save_type == 'joblib':
    joutfnm = splitext( outfnm )[0] + '.joblib'
    joblib.dump( model, joutfnm )
    odhdf5.setAttr( modelgrp, 'path', joutfnm )
  h5file.close()
  if scaler != None:
    dgbhdf5.addScaler( outfnm, scaler )
  dgbhdf5.addInfo( inpfnm, getMLPlatform(), outfnm )
  log_msg( 'Model saved.' )

def load( modelfnm ):
  model = None
  h5file = h5py.File( modelfnm, 'r' )

  modelpars = json.loads( odhdf5.getAttr(h5file,'model_config') )
  modeltype = odhdf5.getText( h5file, 'type' )
  info = odhdf5.getInfoDataSet( h5file )
  scaler = dgbhdf5.getScaler( modelfnm )

  modelgrp = h5file['model']
  savetype = odhdf5.getText( modelgrp, 'type' )
  if savetype == 'joblib':
    modfnm = odhdf5.getText( modelgrp, 'path' )
    model = joblib.load( modfnm )
  elif savetype == 'pickle':
    modeldata = modelgrp['object']
    model = pickle.loads( modeldata[:].tostring() )

  h5file.close()
  return (model,scaler)
 
def apply( model, samples, scaler, isclassification, withpred, withprobs, withconfidence, doprobabilities ):
  samples = np.reshape( samples, (len(samples),-1) )
  if scaler != None:
    samples = scaler.transform( samples )

  ret = {}
  res = None
  if withpred:
    if isclassification:
      res = None  #TODO
    else:
      res = model.predict( samples )
    ret.update({dgbkeys.preddictstr: res})

  if isclassification and (doprobabilities or withconfidence):
    ret.update({dgbkeys.probadictstr: None}) #TODO

  return ret
