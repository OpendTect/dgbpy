#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Jan 2019
#
# _________________________________________________________________________
# various tools machine learning using Sci-kit platform
#

import os.path
import h5py
import json
import joblib
import numpy as np
import pickle

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
#from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from odpy.common import log_msg
import odpy.hdf5 as odhdf5
import dgbpy.keystr as dgbkeys
from dgbpy import hdf5 as dgbhdf5

platform = (dgbkeys.scikitplfnm,'Scikit-learn')
regmltypes = (\
            ('linear','Linear'),\
            ('ensemble','Ensemble'),\
            ('neuralnet','Neural Network'),\
            ('svm','SVM')
          )
classmltypes = (\
            ('logistic','Logistic'),\
            ('ensemble','Ensemble'),\
            ('neuralnet','Neural Network'),\
            ('svm','SVM')
          )
lineartypes = [ ('oslq','Ordinary Least Squares') ]
logistictypes = [ ('log','Logistic Regression Classifier') ]
ensembletypes = [\
                  ('randfor','Random Forests'),\
                  ('gbc','Gradient Boosting'),\
                  ('ada','Adaboost'),\
                ]
try:
  import xgboost
  ensembletypes.append( ('xgb','XGBoost: (Random Forests)') )
except Exception as e:
  pass

nntypes = [ ('mlp','Multi-Layer Perceptron') ]
svmtypes = [ ('svm','Support Vector Machine') ]

savetypes = ( 'joblib', 'pickle' )
defsavetype = savetypes[0]

def getMLPlatform():
  return platform[0]

def getUIMLPlatform():
  return platform[1]

def getUiModelTypes(isclassification):
  if isclassification:
    return dgbkeys.getNames( classmltypes )
  return dgbkeys.getNames( regmltypes )

def getUiLinearTypes():
  return dgbkeys.getNames( lineartypes )

def getUiLogTypes():
  return dgbkeys.getNames( logistictypes )

def getUiEnsembleTypes():
  return dgbkeys.getNames( ensembletypes )

def getUiNNTypes():
  return dgbkeys.getNames( nntypes )

def getUiSVMTypes():
  return dgbkeys.getNames( svmtypes)

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
  'svmpars': {
    'kernel': 'Radial Basis Function',
    'degree': 3
    }
}

def getLinearPars( modelname='Ordinary Least Squares'):
  return {
    'decimation': False,
    'modelname': modelname
    }

def getLogPars( modelname='Logistic Regression Classifier',solver='Liblinear'):
  return {
    'decimation': False,
    'modelname': modelname,
    'solver': solver
    }

def getEnsembleParsRF( modelname='Random Forests',
                       maxdep=scikit_dict['ensemblepars']['rf']['maxdep'],
                       est=scikit_dict['ensemblepars']['rf']['est'] ):
  return {
    'decimation': False,
    'maxdep' : maxdep,
    'est': est,
    'modelname': modelname
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
    'modelname': modelname
    }

def getEnsembleParsAda( modelname='Adaboost',
                        est=scikit_dict['ensemblepars']['ada']['est'],
                        lr=scikit_dict['ensemblepars']['ada']['lr'] ):
  return {
    'decimation': False,
    #'maxdep' : maxdep,
    'est': est,
    'lr': lr,
    'modelname': modelname
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
    'modelname': modelname
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
    'modelname': modelname
    }

def getSVMPars( modelname='Support Vector Machine',
                kernel = scikit_dict['svmpars']['kernel'],
                degree = scikit_dict['svmpars']['degree'] ):
  return {
    'decimation': False,
    'kernel': kernel,
    'degree': degree,
    'modelname': modelname
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

def transformBack( samples, mean, stddev ):
  samples *= stddev
  samples += mean
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

def unscale( samples, scaler ):
  if scaler == None:
    return samples
  if scaler.n_samples_seen_ == 1:
    samples = transformBack( samples, scaler.mean_[0], scaler.scale_[0] )
  else:
    for i in range(scaler.n_samples_seen_):
      samples[:,i] = transformBack( samples[:,i], scaler.mean_[i], scaler.scale_[i] )
  
  return samples

def getDefaultModel( setup, params=scikit_dict ):
  modelname = params['modelname']
  isclassification = setup[dgbhdf5.classdictstr]
  try:
    if modelname == 'Ordinary Least Squares':
      model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
    elif modelname == 'Logistic Regression Classifier':
      solvername = params['solver']
      #transform UI name to actual kernel function name
      if solvername == 'Newton-CG':
        solver = 'newton-cg'
      elif solvername == 'Lbfgs':
        solver = 'lbfgs'
      elif solvername == 'Liblinear':
        solver = 'liblinear'
      elif solvername == 'Sag':
        solver = 'sag'
      elif solvername == 'Saga':
        solver = 'saga'
      model = LogisticRegression('l2',False,0.0001,1.0,True,1,None,None,solver)
    elif modelname == 'Random Forests':
      max_depth = params['maxdep']
      n_estimators = params['est']
      if isclassification:
        model = RandomForestClassifier(n_estimators,'gini', max_depth)
      else:
        model = RandomForestRegressor(n_estimators,'mse', max_depth)     
    elif modelname == 'Gradient Boosting':
      max_depth = params['maxdep']
      n_estimators = params['est']
      learning_rate = params['lr']
      if isclassification:
        model = GradientBoostingClassifier('deviance', learning_rate, n_estimators, 1.0, 'friedman_mse',
                                        2, 1, 0.0, max_depth)
      else:
        model = GradientBoostingRegressor('ls', learning_rate, n_estimators, 1.0, 'friedman_mse',
                                        2, 1, 0.0, max_depth)
    elif modelname == 'Adaboost':
      n_estimators = params['est']
      learning_rate = params['lr']
      if isclassification:
        model = AdaBoostClassifier(None,n_estimators,learning_rate)
      else:
        model = AdaBoostRegressor(None, n_estimators, learning_rate)
    elif modelname == 'XGBoost: (Random Forests)':
      import xgboost
      max_depth = params['maxdep']
      n_estimators = params['est']
      learning_rate = params['lr']
      if isclassification:
        model = xgboost.XGBRFClassifier(max_depth,learning_rate,n_estimators)
      else:
        model = xgboost.XGBRFRegressor(max_depth,learning_rate,n_estimators)
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
      if isclassification:
        model = MLPClassifier(hidden_layer,'relu','adam',0.0001,'auto',
                            'constant',learning_rate_init,0.5, max_iter)
      else:
        model = MLPRegressor(hidden_layer,'relu','adam',0.0001,'auto',
                            'constant',learning_rate_init,0.5, max_iter)
    elif modelname == 'Support Vector Machine':
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
      if isclassification:
        model = SVC(1.0, kernel, degree)
      else:
        model = SVR(kernel, degree)
  except Exception as e:
    log_msg( 'Exception:', e )
    raise e
  return model

def train(model, trainingdp):
  x_train = trainingdp[dgbkeys.xtraindictstr]
  y_train = trainingdp[dgbkeys.ytraindictstr]
  return model.fit(x_train,y_train)

def save( model, inpfnm, outfnm, save_type=defsavetype ):
  h5file = h5py.File( outfnm, 'w' )
  odhdf5.setAttr( h5file, 'backend', 'scikit-learn' )
  odhdf5.setAttr( h5file, 'sklearn_version', sklearn.__version__ )
  odhdf5.setAttr( h5file, 'type', 'RandomForestRegressor' )
  odhdf5.setAttr( h5file, 'model_config', json.dumps(model.get_params()) )
  modelgrp = h5file.create_group( 'model' )
  odhdf5.setAttr( modelgrp, 'type', save_type )
  if save_type == savetypes[0]:
    joutfnm = os.path.splitext( outfnm )[0] + '.joblib'
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
    modbasefnm = os.path.basename( modfnm )
    moddir = os.path.dirname( modelfnm )
    modlocfnm = os.path.join( moddir, modbasefnm )
    if os.path.exists( modlocfnm ):
      modfnm = modlocfnm
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
    res = np.transpose( model.predict( samples ) )
    ret.update({dgbkeys.preddictstr: res})

  if isclassification and (doprobabilities or withconfidence):
    res = np.transpose( model.predict_proba( samples ) )
    ret.update({dgbkeys.probadictstr: res})

  return ret
