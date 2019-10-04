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
import xgboost
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
#from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from odpy.common import log_msg
import odpy.hdf5 as odhdf5
import dgbpy.keystr as dgbkeys
from dgbpy import hdf5 as dgbhdf5

platform = (dgbkeys.scikitplfnm,'Scikit-learn')
mltypes = (\
            ('linear','Linear'),\
            ('ensemble','Ensemble'),\
            ('neuralnet','Neural Network'),\
            ('svr','SVR')
          )
lineartypes = [ ('oslq','Ordinary Least Squares') ]
ensembletypes = [\
                  ('randfor','Random Forests'),\
                  ('gbc','Gradient Boosting'),\
                  ('ada','Adaboost'),\
                  ('xgb','XGBoost: (Random Forests)')\
                ]
nntypes = [ ('mlp','Multi-Layer Perceptron') ]
svrtypes = [ ('svr','Support Vector Regression') ]

savetypes = ( 'joblib', 'pickle' )
defsavetype = savetypes[0]

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

def getUiSVRTypes():
  return dgbkeys.getNames( svrtypes)

scikit_dict = {
  'ensemblepars': {
    'rf': {
      'maxdep': 50,
      'est': 100
      },
    'gb': {
      'maxdep': 3,
      'est': 100,
      'lr': 0.1
      },
    'ada': {
      #'maxdep': 3,
      'est': 50,
      'lr': 1
      },
    'xg': {
      'maxdep': 50,
      'est': 200,
      'lr': 1
      }
    },
  'nnpars': {
    'maxitr': 200,
    'lr': 0.001,
#    'laysizes': (50,25,5),
    'lay1': 50,
    'lay2': 25,
    'lay3': 5,
    'lay4': 3,
    'lay5': 3,
    'nb': 3
    },
  'svrpars': {
    'kernel': 'Radial Basis Function',
    'degree': 3
    }
}

def getLinearPars( modelname='Ordinary Least Squares'):
  return {
    'decimation': False,
    'modelname': 'Ordinary Least Squares'
    }

def getEnsembleParsRF( modelname='Random Forests',
                       maxdep=scikit_dict['ensemblepars']['rf']['maxdep'],
                       est=scikit_dict['ensemblepars']['rf']['est'] ):
  return {
    'decimation': False,
    'maxdep' : maxdep,
    'est': est,
    'modelname': 'Random Forests'
    }

def getEnsembleParsGB( modelname='Gradient Boosting',
                       maxdep=scikit_dict['ensemblepars']['gb']['maxdep'],
                       est=scikit_dict['ensemblepars']['gb']['est'],
                       lr=scikit_dict['ensemblepars']['gb']['lr'] ):
  return {
    'decimation': False,
    'maxdep' : maxdep,
    'est': est,
    'lr': lr,
    'modelname': 'Gradient Boosting'
    }

def getEnsembleParsAda( modelname='Adaboost',
                        est=scikit_dict['ensemblepars']['ada']['est'],
                        lr=scikit_dict['ensemblepars']['ada']['lr'] ):
  return {
    'decimation': False,
    #'maxdep' : maxdep,
    'est': est,
    'lr': lr,
    'modelname': 'Adaboost'
    }

def getEnsembleParsXG( modelname='XGBoost: (Random Forests)',
                       maxdep=scikit_dict['ensemblepars']['xg']['maxdep'],
                       est=scikit_dict['ensemblepars']['xg']['est'],
                       lr=scikit_dict['ensemblepars']['xg']['lr'] ):
  return {
    'decimation': False,
    'maxdep' : maxdep,
    'est': est,
    'lr': lr,
    'modelname': 'XGBoost: (Random Forests)'
    }

def getNNPars( modelname='Multi-Layer Perceptron',
               maxitr = scikit_dict['nnpars']['maxitr'],
               lr = scikit_dict['nnpars']['lr'],
               lay1=scikit_dict['nnpars']['lay1'],
               lay2=scikit_dict['nnpars']['lay2'],
               lay3=scikit_dict['nnpars']['lay3'],
               lay4=scikit_dict['nnpars']['lay4'],
               lay5=scikit_dict['nnpars']['lay5'],
               nb = scikit_dict['nnpars']['nb']):
  return {
    'decimation': False,
    'maxitr': maxitr,
    'lr': lr,
    'lay1': lay1,
    'lay2': lay2,
    'lay3': lay3,
    'lay4': lay4,
    'lay5': lay5,
    'nb': nb,
    'modelname': 'Multi-Layer Perceptron'
    }

def getSVRPars( modelname='Support Vector Regression',
                kernel = scikit_dict['svrpars']['kernel'],
                degree = scikit_dict['svrpars']['degree'] ):
  return {
    'decimation': False,
    'kernel': kernel,
    'degree': degree,
    'modelname': 'Support Vector Regression'
    }


def getNewScaler( mean, scale ):
  scaler = StandardScaler()
  scaler.mean_ = np.array( mean )
  scaler.scale_ = np.array( scale )
  scaler.var_ = np.square( scaler.scale_ )
  scaler.n_samples_seen_ = len(mean)
  return scaler

def getScaler( x_train, byattrib ):
  nrattribs = x_train.shape[1]
  mean = list()
  var = list()
  if byattrib:
    for a in range(nrattribs):
      inp = x_train[:,a,:,:,:]
      mean.append( np.mean(inp,dtype=np.float64) )
      var.append( np.var(inp,dtype=np.float64) )
  else:
    mean.append( np.mean(x_train,dtype=np.float64) )
    var.append( np.var(x_train,dtype=np.float64) )
  scaler = StandardScaler()
  scaler.mean_ = np.array( mean )
  scaler.var_ = np.array( var )
  scaler.scale_ = np.sqrt( scaler.var_ )
  scaler.n_samples_seen_ = len(mean)
  return scaler

def transform( samples, mean, stddev ):
  samples -= mean
  samples /= stddev
  return samples

def scale( samples, scaler ):
  if scaler == None:
    return samples
  if scaler.n_samples_seen_ == 1:
    samples = transform( samples, scaler.mean_[0], scaler.scale_[0] )
  else:
    for i in range(scaler.n_samples_seen_):
      samples[:,i] = transform( samples[:,i], scaler.mean_[i], scaler.scale_[i] )
  
  return samples

def train( training, params=scikit_dict ):
  #learning rate need to multiply by 0.0001(Neural Network)
  #Please check sklearn documentation for more application of each methods
  #
  #TODO: add URL ?
  modelname = params['modelname']
  x_train = training[dgbkeys.xtraindictstr]
  y_train = training[dgbkeys.ytraindictstr]
  try:
    #TODO: remove: flattening can be done with getScaledTrainingData
    x_train = np.reshape( x_train, (len(x_train),-1) )
    if modelname == 'Ordinary Least Squares':
      model = LinearRegression(fit_intercept=True, normalize=False,
                               copy_X=True, n_jobs=None)
    elif modelname == 'Random Forests':
      max_depth = params['maxdep']
      n_estimators = params['est']
      model = RandomForestRegressor(n_estimators, 'mse', max_depth,
                                    min_samples_split=2, min_samples_leaf=1,
                                    min_weight_fraction_leaf=0.0,
                                    max_features='auto', max_leaf_nodes=None,
                                    min_impurity_decrease=0.0,
                                    min_impurity_split=None,
                                    bootstrap=True, oob_score=False,
                                    n_jobs=None, random_state=None, verbose=0,
                                    warm_start=False)
    elif modelname == 'Gradient Boosting':
      max_depth = params['maxdep']
      n_estimators = params['est']
      learning_rate = params['lr']
      model = GradientBoostingRegressor('ls', learning_rate, n_estimators, 1.0,
                                        'friedman_mse', 2, 1, 0.0, max_depth,
                                        min_impurity_decrease=0.0,
                                        min_impurity_split=None, init=None,
                                        random_state=None, max_features=None,
                                        alpha=0.9, verbose=0,
                                        max_leaf_nodes=None, warm_start=False,
                                        presort='auto',
                                        validation_fraction=0.1,
                                        n_iter_no_change=None, tol=0.0001)
    elif modelname == 'Adaboost':
      #we use default maxdepth here 3
      #max_depth = params['maxdep']
      n_estimators = params['est']
      learning_rate = params['lr']
      model = AdaBoostRegressor(None, n_estimators, learning_rate,
                                loss='linear', random_state=None)
      #model =  AdaBoostRegressor(DecisionTreeRegressor('mse', 'best', max_depth), n_estimators,learning_rate,loss='linear', random_state=None)
    elif modelname == 'XGBoost: (Random Forests)':
      max_depth = params['maxdep']
      n_estimators = params['est']
      learning_rate = params['lr']
      model = xgboost.XGBRFRegressor(max_depth,learning_rate,n_estimators,
                          verbosity=1,silent=None,objective='reg:squarederror',
                          n_jobs=1,nthread=None,gamma=0,min_child_weight=1,
                          max_delta_step=0,subsample=0.8,colsample_bytree=1,
                          colsample_bylevel=1, colsample_bynode=0.8,
                          reg_alpha=0, reg_lambda=1e-05, scale_pos_weight=1,
                          base_score=0.5, random_state=0, seed=None,
                          missing=None)
    elif modelname == 'Multi-Layer Perception':
      lay1 = params['lay1']
      lay2 = params['lay2']
      lay3 = params['lay3']
      lay4 = params['lay4']
      lay5 = params['lay5']
      nb = params['nb']
      hidden_layer = (lay1,lay2,lay3,lay4,lay5)
      hidden_layer = hidden_layer[0:nb]
      max_iter = params['maxitr']
      learning_rate_init = params['lr']
      model = MLPRegressor(hidden_layer,'relu','adam',0.0001,'auto',
                            'constant',learning_rate_init,0.5, max_iter,
                            shuffle=True, random_state=None, tol=0.0001,
                            verbose=False,warm_start=False,momentum=0.9,
                            nesterovs_momentum=True,early_stopping=False,
                            validation_fraction=0.1, beta_1=0.9,
                            beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
    elif modelname == 'Support Vector Regression':
      name = params['kernel']
      #transform UI name to actual kernel function name
      if name == 'Radial Basis Function':
        kernel = 'rbf'
      elif name == 'Linear':
        kernel = 'linear'
      elif name == 'Polynomial':
        kernel = 'poly'
      elif name == 'Sigmoid':
        kernel = 'sigmoid'
      degree = params['degree']
      model = SVR(kernel, degree, gamma='auto_deprecated', coef0=0.0,
                  tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200,
                  verbose=False, max_iter=-1)
    model = model.fit(x_train,y_train)
  except Exception as e:
    log_msg( 'Exception:', e )
    raise e
  return model

def save( model, inpfnm, outfnm, save_type=defsavetype ):
  h5file = h5py.File( outfnm, 'w' )
  odhdf5.setAttr( h5file, 'backend', 'scikit-learn' )
  odhdf5.setAttr( h5file, 'sklearn_version', sklearn.__version__ )
  odhdf5.setAttr( h5file, 'type', 'RandomForestRegressor' )
  odhdf5.setAttr( h5file, 'model_config', json.dumps(model.get_params()) )
  modelgrp = h5file.create_group( 'model' )
  odhdf5.setAttr( modelgrp, 'type', save_type )
  if save_type == savetypes[0]:
    joutfnm = splitext( outfnm )[0] + '.joblib'
    joblib.dump( model, joutfnm )
    odhdf5.setAttr( modelgrp, 'path', joutfnm )
  elif save_type == savetypes[1]:
    exported_modelstr = pickle.dumps(model)
    exported_model = np.frombuffer( exported_modelstr, dtype='S1', count=len(exported_modelstr) )
    modelgrp.create_dataset('object',data=exported_model)
  h5file.close()
  dgbhdf5.addInfo( inpfnm, getMLPlatform(), outfnm )

def load( modelfnm ):
  model = None
  h5file = h5py.File( modelfnm, 'r' )

  modelpars = json.loads( odhdf5.getAttr(h5file,'model_config') )
  modeltype = odhdf5.getText( h5file, 'type' )
  info = odhdf5.getInfoDataSet( h5file )

  modelgrp = h5file['model']
  savetype = odhdf5.getText( modelgrp, 'type' )
  if savetype == savetypes[0]:
    modfnm = odhdf5.getText( modelgrp, 'path' )
    model = joblib.load( modfnm )
  elif savetype == savetypes[1]:
    modeldata = modelgrp['object']
    model = pickle.loads( modeldata[:].tostring() )

  h5file.close()
  return model
 
def apply( model, samples, scaler, isclassification, withpred, withprobs, withconfidence, doprobabilities ):
  model.verbose = 0
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
