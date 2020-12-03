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
import sys
import h5py
import json
import joblib
import numpy as np
import pickle
from pathlib import PurePosixPath, PureWindowsPath

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
from sklearn.svm import SVC, LinearSVR

from odpy.common import log_msg, redirect_stdout, restore_stdout, isWin
from odpy.oscommand import printProcessTime
import odpy.hdf5 as odhdf5
import dgbpy.keystr as dgbkeys
from dgbpy import hdf5 as dgbhdf5
from multiprocessing import cpu_count

tot_cpu = cpu_count()
n_cpu = tot_cpu

def hasXGBoost():
  try:
    import xgboost
  except ModuleNotFoundError:
    return False
  return True

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
ensembletypes = []
if hasXGBoost():
  ensembletypes.append( ('xgbdt','XGBoost: (Decision Tree)') )
  ensembletypes.append( ('xgbrf','XGBoost: (Random Forests)') )
ensembletypes.append( ('randfor','Random Forests') )
ensembletypes.append( ('gbc','Gradient Boosting') )
ensembletypes.append( ('ada','Adaboost') )

nntypes = [ ('mlp','Multi-Layer Perceptron') ]
svmtypes = [ ('svm','Support Vector Machine') ]

solvertypes = [\
                ('newton-cg','Newton-CG'),\
                ('lbfgs','Lbfgs'),\
                ('liblinear','Liblinear'),\
                ('sag','Sag'),\
                ('saga','Saga'),\
              ]
kerneltypes = [\
                ('linear','Linear'),\
                ('poly','Polynomial'),\
                ('rbf','Radial Basis Function'),\
                ('sigmoid','Sigmoid'),\
              ]

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

def getUiSolverTypes():
  return dgbkeys.getNames( solvertypes )

def getUiNNKernelTypes():
  return dgbkeys.getNames( kerneltypes )

def getDefaultSolver( uiname=True ):
  solverstr = LogisticRegression().solver
  return dgbkeys.getNameFromList( solvertypes, solverstr, uiname )

def getDefaultNNKernel( uiname=True ):
  kernelstr = SVC().kernel
  return dgbkeys.getNameFromList( kerneltypes, kernelstr, uiname )


scikit_dict = {
  'ensemblepars': {
    'xgdt': {
      'lr': 1,
      'maxdep': 1,
      'est': 1
      },
    'xgrf': {
      'lr': 1,
      'maxdep': 1,
      'est': 1
      },
    'rf': {
      'maxdep': 50, #default: None, but we prefer less
      'est': RandomForestRegressor().n_estimators
      },
    'gb': {
      'maxdep': GradientBoostingRegressor().max_depth,
      'est': GradientBoostingRegressor().n_estimators,
      'lr': GradientBoostingRegressor().learning_rate
      },
    'ada': {
      'est': AdaBoostRegressor().n_estimators,
      'lr': AdaBoostRegressor().learning_rate
      },
    },
  'nnpars': {
    'maxitr': MLPRegressor().max_iter,
    'lr': MLPRegressor().learning_rate_init,
#    'laysizes': (50,25,5),
    'lay1': 50,
    'lay2': 25,
    'lay3': 5,
    'lay4': 3,
    'lay5': 3,
    'nb': 3
    },
  'svmpars': {
    'kernel': getDefaultNNKernel(False),
    'degree': SVC().degree
    }
}
if hasXGBoost():
  from xgboost import XGBRegressor, XGBRFRegressor
  defdtregressor = XGBRegressor()
  xgdtpars = {
    'lr': scikit_dict['ensemblepars']['xgdt']['lr'],
    'maxdep': scikit_dict['ensemblepars']['xgdt']['maxdep'],
    'est': defdtregressor.n_estimators,
  }
  if defdtregressor.learning_rate != None:
    xgdtpars.update({'lr': defdtregressor.learning_rate})
  if defdtregressor.max_depth != None:
    xgdtpars.update({'maxdep': defdtregressor.max_depth})
  scikit_dict['ensemblepars'].update({'xgdt': xgdtpars})

  defrfregressor = XGBRFRegressor()
  xgrfpars = {
      'lr': defrfregressor.learning_rate,
      'maxdep': scikit_dict['ensemblepars']['xgrf']['maxdep'],
      'est': defrfregressor.n_estimators,
  }
  if defrfregressor.max_depth != None:
    xgrfpars.update({'maxdep': defrfregressor.max_depth})
  scikit_dict['ensemblepars'].update({'xgrf': xgrfpars})

def getLinearPars( modelname='Ordinary Least Squares'):
  return {
    'decimation': False,
    'modelname': modelname
    }

def getLogPars( modelname='Logistic Regression Classifier',solver=None):
  if solver == None:
    solver = getDefaultSolver()
  return {
    'decimation': False,
    'modelname': modelname,
    'solver': solver
    }

def getEnsembleParsXGDT( modelname='XGBoost: (Decision Tree)',
                       maxdep=scikit_dict['ensemblepars']['xgdt']['maxdep'],
                       est=scikit_dict['ensemblepars']['xgdt']['est'],
                       lr=scikit_dict['ensemblepars']['xgdt']['lr'] ):
  return {
    'decimation': False,
    'maxdep' : maxdep,
    'est': est,
    'lr': lr,
    'modelname': modelname
    }

def getEnsembleParsXGRF( modelname='XGBoost: (Random Forests)',
                       maxdep=scikit_dict['ensemblepars']['xgrf']['maxdep'],
                       est=scikit_dict['ensemblepars']['xgrf']['est'],
                       lr=scikit_dict['ensemblepars']['xgrf']['lr'] ):
  return {
    'decimation': False,
    'maxdep' : maxdep,
    'est': est,
    'lr': lr,
    'modelname': modelname
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
      model = LinearRegression(n_jobs=-1)
    elif modelname == 'Logistic Regression Classifier':
      solvernm = dgbkeys.getNameFromUiName( solvertypes, params['solver'] )
      model = LogisticRegression(solver=solvernm,n_jobs=-1)
    elif modelname == 'XGBoost: (Decision Tree)':
      from xgboost import XGBClassifier, XGBRegressor
      learning_rate = params['lr']
      max_depth = params['maxdep']
      n_estimators = params['est']
      if isclassification:
        model = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,n_jobs=n_cpu)
      else:
        model = XGBRegressor(objective='reg:squarederror',n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,n_jobs=n_cpu)
    elif modelname == 'XGBoost: (Random Forests)':
      from xgboost import XGBRFClassifier, XGBRFRegressor
      learning_rate = params['lr']
      max_depth = params['maxdep']
      n_estimators = params['est']
      if isclassification:
        model = XGBRFClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,n_jobs=n_cpu)
      else:
        model = XGBRFRegressor(objective='reg:squarederror',n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,n_jobs=n_cpu)
    elif modelname == 'Random Forests':
      n_estimators = params['est']
      max_depth = params['maxdep']
      if isclassification:
        model = RandomForestClassifier(n_estimators=n_estimators,criterion='gini',max_depth=max_depth,n_jobs=-1)
      else:
        model = RandomForestRegressor(n_estimators=n_estimators,criterion='mse',max_depth=max_depth,n_jobs=-1)
    elif modelname == 'Gradient Boosting':
      n_estimators = params['est']
      learning_rate = params['lr']
      max_depth = params['maxdep']
      if isclassification:
        model = GradientBoostingClassifier(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth)
      else:
        model = GradientBoostingRegressor(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth)
    elif modelname == 'Adaboost':
      n_estimators = params['est']
      learning_rate = params['lr']
      if isclassification:
        model = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
      else:
        model = AdaBoostRegressor(n_estimators=n_estimators,learning_rate=learning_rate)
    elif modelname == 'Multi-Layer Perceptron':
      lay1 = params['lay1']
      lay2 = params['lay2']
      lay3 = params['lay3']
      lay4 = params['lay4']
      lay5 = params['lay5']
      nb = params['nb']
      hidden_layer = (lay1,lay2,lay3,lay4,lay5)
      hidden_layer = hidden_layer[0:nb]
      learning_rate = params['lr']
      max_iter = params['maxitr']
      if isclassification:
        model = MLPClassifier(hidden_layer_sizes=hidden_layer,learning_rate_init=learning_rate,max_iter=max_iter)
      else:
        model = MLPRegressor(hidden_layer_sizes=hidden_layer,learning_rate_init=learning_rate,max_iter=max_iter)
    elif modelname == 'Support Vector Machine':
      kernel = dgbkeys.getNameFromUiName( kerneltypes, params['kernel'] )
      degree = params['degree']
      if isclassification:
        model = SVC(kernel=kernel,degree=degree)
      else:
        model = LinearSVR(kernel=kernel,degree=degree)
  except Exception as e:
    log_msg( 'Exception:', e )
    raise e
  return model

def train(model, trainingdp):
  x_train = trainingdp[dgbkeys.xtraindictstr]
  y_train = trainingdp[dgbkeys.ytraindictstr].ravel()
  printProcessTime( 'Training with scikit-learn', True, print_fn=log_msg )
  log_msg( '\nTraining on', len(y_train), 'samples' )
  log_msg( 'Validate on', len(trainingdp[dgbkeys.yvaliddictstr]), 'samples\n' )
  redirect_stdout()
  model.verbose = 51
  ret = model.fit(x_train,y_train)
  restore_stdout()
  printProcessTime( 'Training with scikit-learn', False, print_fn=log_msg, withprocline=False )
  assessQuality( model, trainingdp )
  return ret

def assessQuality( model, trainingdp ):
  if not dgbkeys.yvaliddictstr in trainingdp:
    return
  try:
    x_validate = trainingdp[dgbkeys.xvaliddictstr]
    y_validate = trainingdp[dgbkeys.yvaliddictstr].ravel()
    y_predicted = model.predict(x_validate)
    if trainingdp[dgbkeys.infodictstr][dgbkeys.classdictstr]:
      cc = np.sum( y_predicted==y_validate) / len(y_predicted)
    else:
      cc = np.corrcoef( y_predicted, y_validate )[0,1]
    log_msg( '\nCorrelation coefficient with validation data: ', "%.4f" % cc, '\n' )
  except Exception as e:
    log_msg( '\nCannot compute model quality:' )
    log_msg( repr(e) )

def save( model, outfnm, save_type=defsavetype ):
  h5file = odhdf5.openFile( outfnm, 'w' )
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

def translateFnm( modfnm, modelfnm ):
  posidxh5fp = PurePosixPath( modelfnm )
  winh5fp = PureWindowsPath( modelfnm )
  posixmodfp = PurePosixPath( modfnm )
  winmodfp = PureWindowsPath( modfnm )
  if isWin():
    moddir = winh5fp.parent
    modbasefnm = winmodfp.name
    modlocfnm = PureWindowsPath( moddir ).joinpath( PureWindowsPath(modbasefnm))
    if os.path.exists(modlocfnm):
      modfnm = modlocfnm
    else:
      moddir = posidxh5fp.parent
      modbasefnm = posixmodfp.name
      modlocfnm = PurePosixPath( moddir ).joinpath( PurePosixPath(modbasefnm) )
      if os.path.exists(modlocfnm):
        modfnm = modlocfnm
  else:
    moddir = posidxh5fp.parent
    modbasefnm = posixmodfp.name
    modlocfnm = PurePosixPath( moddir ).joinpath( PurePosixPath(modbasefnm) )
    if os.path.exists(modlocfnm):
      modfnm = modlocfnm
    else:
      moddir = winh5fp.parent
      modbasefnm = winmodfp.name
      modlocfnm = PureWindowsPath( moddir).joinpath(PureWindowsPath(modbasefnm))
      modlocfnm = modlocfnm.as_posix()
      if os.path.exists(modlocfnm):
        modfnm = modlocfnm
  return modfnm

def load( modelfnm ):
  model = None
  h5file = odhdf5.openFile( modelfnm, 'r' )

  modelpars = json.loads( odhdf5.getAttr(h5file,'model_config') )
  modeltype = odhdf5.getText( h5file, 'type' )
  info = odhdf5.getInfoDataSet( h5file )

  modelgrp = h5file['model']
  savetype = odhdf5.getText( modelgrp, 'type' )
  if savetype == savetypes[0]:
    modfnm = odhdf5.getText( modelgrp, 'path' )
    modfnm = translateFnm( modfnm, modelfnm )
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
