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
import json
import joblib
import numpy as np
import pickle

try:
  import sklearn
  from sklearn.preprocessing import MinMaxScaler, StandardScaler
  from sklearn.linear_model import LinearRegression, LogisticRegression
  from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
  from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
  from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
  from sklearn.neural_network import MLPClassifier, MLPRegressor
  from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
  from sklearn.multioutput import MultiOutputRegressor
  from sklearn.cluster import KMeans, MeanShift, SpectralClustering
except ModuleNotFoundError:
  pass
from odpy.common import log_msg, redirect_stdout, restore_stdout
from odpy.oscommand import printProcessTime
import odpy.hdf5 as odhdf5
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
from dgbpy.mlio import announceTrainingSuccess, announceTrainingFailure
from multiprocessing import cpu_count

tot_cpu = cpu_count()
n_cpu = tot_cpu

def hasScikit():
  try:
    import sklearn
  except ModuleNotFoundError:
    return False 
  return True

def isVersionAtLeast(version):
  try:
    import sklearn
  except ModuleNotFoundError:
    return False
  return sklearn.__version__ >= version

def hasXGBoost():
  try:
    import xgboost
  except ModuleNotFoundError:
    return False
  return True

platform = (dgbkeys.scikitplfnm,'Scikit-learn')
mse_criterion = 'squared_error' if isVersionAtLeast('1.0') else 'mse'

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
clustertypes = [ ('cluster','Clustering') ]
ensembletypes = []
if hasXGBoost():
  ensembletypes.append( ('xgbdt','XGBoost: (Decision Tree)') )
  ensembletypes.append( ('xgbrf','XGBoost: (Random Forests)') )
ensembletypes.append( ('randfor','Random Forests') )
ensembletypes.append( ('gbc','Gradient Boosting') )
ensembletypes.append( ('ada','Adaboost') )

nntypes = [ ('mlp','Multi-Layer Perceptron') ]
svmtypes = [ ('svm','Support Vector Machine') ]
clustermethods = [ ('kmeans','K-Means'),\
                   ('meanshift','Mean Shift'),\
                   ('spec','Spectral Clustering') ]

solvertypes = [\
                ('newton-cg','Newton-CG'),\
                ('lbfgs','Lbfgs'),\
                ('liblinear','Liblinear'),\
                ('sag','Sag'),\
                ('saga','Saga'),\
              ]
linkernel = 'linear'
kerneltypes = [\
                (linkernel,'Linear'),\
                ('poly','Polynomial'),\
                ('rbf','Radial Basis Function'),\
                ('sigmoid','Sigmoid'),\
              ]

savetypes = ( 'onnx', 'joblib', 'pickle' )
defsavetype = savetypes[0]
xgboostjson = 'xgboostjson'

defstoragetype = dgbhdf5.StorageType.LOCAL.value

scikit_dict = {
  'storagetype': defstoragetype,
  's3_bucket': None,
  'savetype': defsavetype,
  'scaler': None,
}

def getMLPlatform():
  return platform[0]

def getUIMLPlatform():
  return platform[1]

def getUiModelTypes(isclassification,ismultiregression, issegmentation):
  if issegmentation:
    return getUiClusterTypes()
  if isclassification:
    return dgbkeys.getNames( classmltypes )
  if ismultiregression:
    return dgbkeys.getNames( (regmltypes[1],) )
  return dgbkeys.getNames( regmltypes )

def getUiLinearTypes():
  return dgbkeys.getNames( lineartypes )

def getUiLogTypes():
  return dgbkeys.getNames( logistictypes )

def getUiClusterTypes():
  return dgbkeys.getNames( clustertypes )

def getUiClusterMethods():
  return dgbkeys.getNames( clustermethods )

def getUiEnsembleTypes(ismultiregression):
  if ismultiregression:
    return dgbkeys.getNames( (ensembletypes[2],) )
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

def getDefaultNNKernel( isclass, uiname=True ):
  if isclass:
    kernelstr = SVC().kernel
  else:
    kernelstr = linkernel
  return dgbkeys.getNameFromList( kerneltypes, kernelstr, uiname )

if hasScikit():
  scikit_dict.update({
    'ensemblepars': {
      'xgdt': {
        'lr': 1,
        'maxdep': 5,
        'est': 1
        },
      'xgrf': {
        'lr': 1,
        'maxdep': 5,
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
      'kernel': getDefaultNNKernel(False,uiname=False),
      'degree': SVC().degree
      },
    'clusterpars': {
      'kmeans': {
        'n_clusters': KMeans().n_clusters,
        'n_init': KMeans().n_init,
        'max_iter': KMeans().max_iter
        },
      'meanshift': {
        'max_iter': MeanShift().max_iter
        },
      'spectral': {
        'n_clusters': SpectralClustering().n_clusters,
        'n_init': SpectralClustering().n_init
        }
      }
  })
else:
  scikit_dict.update({
  'ensemblepars': {
    'xgdt': {
      'lr': 1,
      'maxdep': 5,
      'est': 1
      },
    'xgrf': {
      'lr': 1,
      'maxdep': 5,
      'est': 1
      },
    'rf': {
      'maxdep': 50, #default: None, but we prefer less
      'est': 1
      },
    'gb': {
      'maxdep': 1,
      'est': 1,
      'lr': 1
      },
    'ada': {
      'est': 1,
      'lr': 1
      },
    },
  'nnpars': {
    'maxitr': 1,
    'lr': 1,
#    'laysizes': (50,25,5),
    'lay1': 50,
    'lay2': 25,
    'lay3': 5,
    'lay4': 3,
    'lay5': 3,
    'nb': 3
    },
  'svmpars': {
    'kernel': None,
    'degree': 1
    },
  'clusterpars': {
    'kmeans': {
      'n_clusters': 1,
      'n_init': 1,
      'max_iter': 1
      },
    'meanshift': {
      'max_iter': 1
      },
    'spectral': {
      'n_clusters': 1,
      'n_init': 1
      }
    }
  })
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


def getClusterParsKMeans( methodname, nclust, ninit, maxiter ):
  return {
    'modelname': 'Clustering',
    'methodname': methodname,
    'n_clusters': nclust,
    'n_init': ninit,
    'max_iter': maxiter
  }


def getClusterParsMeanShift( methodname, maxiter ):
  return {
    'modelname': 'Clustering',
    'methodname': methodname,
    'max_iter': maxiter
  }


def getClusterParsSpectral( methodname, nclust, ninit ):
  return {
    'modelname': 'Clustering',
    'methodname': methodname,
    'n_clusters': nclust,
    'n_init': ninit
  }


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
  """ Gets new scaler object for standardization 

  Parameters:
    * mean (ndarray of shape (n_features,) or None): mean value to be used for scaling
    * scale ndarray of shape (n_features,) or None: Per feature relative scaling of the
      data to achieve zero mean and unit variance (from sklearn docs)

  Returns:
    * object: scaler (an instance of sklearn.preprocessing.StandardScaler())
  """

  scaler = StandardScaler()
  scaler.mean_ = np.array( mean )
  scaler.scale_ = np.array( scale )
  scaler.var_ = np.square( scaler.scale_ )
  scaler.n_samples_seen_ = len(mean)
  return scaler

def getNewMinMaxScaler( data, minout=0, maxout=1 ):
  """ Gets new scaler object for normalization

  Parameters:
    * data ndarray: data used to fit the MinMaxScaler object
    * minout int: desired minimum value of transformed data
    * maxout int: desired maximum value of transformed data

  Returns:
    * object: scaler (an instance of sklearn.preprocessing.MinMaxScaler())

  """

  scaler = MinMaxScaler( feature_range=(minout, maxout), copy=False )
  scaler.fit( data.reshape((np.prod(data.shape),1)) )
  return scaler

def getScaler( x_train, byattrib ):
  """ Extract scaler for standardization of features.
  The scaler is such that when it is applied to the samples they get
  a mean of 0 and standard deviation of 1, globally or per channel

  Parameters:
    * x_train ndarray: data used to fit the StandardScaler object
    * byattrib Boolean: sets a per channel scaler if True

    Returns:
    * object: scaler (an instance of sklearn.preprocessing.StandardScaler())


  """

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
  if stddev:
    samples /= stddev
  return samples

def transformBack( samples, mean, stddev ):
  samples *= stddev
  samples += mean
  return samples

def scale( samples, scaler ):
  """ Applies a scaler transformation to an array of features
  If the scaler is a StandardScaler, the returned samples have 
  a mean and standard deviation according to the value set in the scaler.
  If the scaler is a MinMaxScaler, the returned samples have
  a min/max value according to the range set in that scaler
  Scaling is applied on the input array directly.

  Parameters:
    * samples ndarray: input/output values to be scaled
    * scaler sklearn.preprocessing scaler object (see sklearn docs)

  """

  if scaler == None:
    return samples
  if isinstance(scaler,MinMaxScaler):
    shape = samples.shape
    samples = scaler.transform( samples.reshape((np.prod(shape),1)) )
    samples = samples.reshape( shape )
  elif scaler.n_samples_seen_ == 1:
    samples = transform( samples, scaler.mean_[0], scaler.scale_[0] )
  else:
    mean = 0.0
    scale = 1.0
    for i in range(samples.shape[1]):
      if i<scaler.n_samples_seen_:
        mean = scaler.mean_[i]
        scale = scaler.scale_[i]
      samples[:,i] = transform( samples[:,i], mean, scale )

  return samples

def unscale( samples, scaler ):
  """ Applies an inverse scaler transformation to an array of features
  Scaling is applied on the input array directly.

  Parameters:
    * samples ndarray: input/output values to be unscaled
    * scaler sklearn.preprocessing scaler object (see sklearn docs)

  """

  if scaler == None:
    return samples
  if isinstance(scaler,MinMaxScaler):
    shape = samples.shape
    samples = scaler.inverse_transform( samples.reshape((np.prod(shape),1)) )
    samples = samples.reshape( shape )
  elif scaler.n_samples_seen_ == 1:
    samples = transformBack( samples, scaler.mean_[0], scaler.scale_[0] )
  else:
    mean = 0.0
    scale = 1.0
    for i in range(samples.shape[1]):
      if i<scaler.n_samples_seen_:
        mean = scaler.mean_[i]
        scale = scaler.scale_[i]
      samples[:,i] = transformBack( samples[:,i], mean, scale )

  return samples

def getDefaultModel( setup, params=scikit_dict ):
  modelname = params['modelname']
  isclassification = setup[dgbhdf5.classdictstr]
  ismultilabelregression = dgbhdf5.isMultiLabelRegression(setup)
  if ismultilabelregression and modelname != 'Random Forests':
    log_msg('Multilabel prediction is only supported for Random Forest in Scikit')   
    return None
  try:
    if modelname =='Clustering':
      method = params['methodname']
      if method == clustermethods[0][1]:
        model = KMeans( params['n_clusters'] )
        model.n_init = params['n_init']
        model.max_iter = params['max_iter']
      elif method == clustermethods[1][1]:
        model = MeanShift()
      else:
        model = SpectralClustering( params['n_clusters'] )
    elif modelname == 'Ordinary Least Squares':
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
      elif ismultilabelregression:
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,criterion=mse_criterion,max_depth=max_depth,n_jobs=-1))
      else:
        model = RandomForestRegressor(n_estimators=n_estimators,criterion=mse_criterion,max_depth=max_depth,n_jobs=-1)
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
      kernel = dgbkeys.getNameFromList( kerneltypes, params['kernel'], uiname=False )
      degree = params['degree']
      if isclassification:
        if kernel == linkernel:
          model = LinearSVC()
        else:
          model = SVC(kernel=kernel,degree=degree)
      else:
        if kernel == linkernel:
          model = LinearSVR()
        else:
          model = SVR(kernel=kernel,degree=degree)
  except Exception as e:
    log_msg( 'Exception:', e )
    announceTrainingFailure
    raise e
  return model

def train(model, trainingdp):
  try:
    x_train = trainingdp[dgbkeys.xtraindictstr]
    if dgbhdf5.isMultiLabelRegression(trainingdp[dgbkeys.infodictstr]):
      y_train = trainingdp[dgbkeys.ytraindictstr]
    else:
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
    announceTrainingSuccess()
    return ret
  except Exception as e:
    announceTrainingFailure()
    raise e

def assessQuality( model, trainingdp ):
  if not dgbkeys.yvaliddictstr in trainingdp:
    return
  try:
    x_validate = trainingdp[dgbkeys.xvaliddictstr]
    if dgbhdf5.isMultiLabelRegression(trainingdp[dgbkeys.infodictstr]):
      y_validate = trainingdp[dgbkeys.yvaliddictstr]
    else:
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
    announceTrainingFailure()
    
def onnx_from_sklearn(model):
  try:
    nattribs = model.n_features_in_
  except AttributeError:
    return None
  from skl2onnx import convert_sklearn
  from skl2onnx.common.data_types import FloatTensorType
  initial_type = [('float_input', FloatTensorType([None,nattribs]))]
  options = None
  if getattr(model,'multi_class',None) or \
     getattr(model,'predict_proba',None):
    options = {id(model): {'zipmap': False}}
    if isinstance(model,MLPClassifier):
      model.classes_ = model.classes_.astype( np.int64 )
  return convert_sklearn(model, initial_types=initial_type, options=options)

def save( model, outfnm, save_type=defsavetype ):
  h5file = odhdf5.openFile( outfnm, 'w' )
  odhdf5.setAttr( h5file, 'backend', 'scikit-learn' )
  odhdf5.setAttr( h5file, 'sklearn_version', sklearn.__version__ )
  odhdf5.setAttr( h5file, 'type', model.__class__.__name__ )
  if isinstance(model, MultiOutputRegressor):
    params = {key:value for key,value in model.get_params().items() if key != "estimator"}
    odhdf5.setAttr( h5file, 'model_config', json.dumps(params) )
  else:
    odhdf5.setAttr( h5file, 'model_config', json.dumps(model.get_params()) )
  modelgrp = h5file.create_group( 'model' )
  if hasXGBoost():
    from xgboost import XGBClassifier, XGBRegressor, \
                        XGBRFClassifier, XGBRFRegressor
    if isinstance( model, XGBClassifier ) or isinstance( model, XGBRegressor ) or \
       isinstance( model, XGBRFClassifier ) or isinstance( model, XGBRFRegressor ):
      import xgboost
      if float(xgboost.__version__[:3]) < 1.0 :
        raise Exception("Xgboost requires a version > 1.0 to save model")
      odhdf5.setAttr( h5file, 'xgboost_version', xgboost.__version__ )
      save_type = xgboostjson
  odhdf5.setAttr( modelgrp, 'type', save_type )
  if save_type == savetypes[0]:
    joutfnm = os.path.splitext( outfnm )[0] + '.onnx'
    onx = onnx_from_sklearn(model)
    with open(joutfnm, 'wb') as f:
      f.write(onx.SerializeToString())
    odhdf5.setAttr( modelgrp, 'path', joutfnm )
  elif save_type == savetypes[1]:
    joutfnm = os.path.splitext( outfnm )[0] + '.joblib'
    joblib.dump( model, joutfnm )
    odhdf5.setAttr( modelgrp, 'path', joutfnm )
  elif save_type == savetypes[2]:
    exported_modelstr = pickle.dumps(model)
    exported_model = np.frombuffer( exported_modelstr, dtype='S1', count=len(exported_modelstr) )
    modelgrp.create_dataset('object',data=exported_model)
  elif save_type == xgboostjson:
    joutfnm = os.path.splitext( outfnm )[0] + '.json'
    model.save_model( joutfnm )
    odhdf5.setAttr( modelgrp, 'path', joutfnm )
  h5file.close()

def load( modelfnm ):
  model = None
  h5file = odhdf5.openFile( modelfnm, 'r' )
  modelgrp = h5file['model']
  savetype = odhdf5.getText( modelgrp, 'type' )
  if savetype == savetypes[0]:
    modfnm = odhdf5.getText( modelgrp, 'path' )
    modfnm = dgbhdf5.translateFnm( modfnm, modelfnm )
    from dgbpy.sklearn_classes import OnnxScikitModel
    model = OnnxScikitModel( str(modfnm) )
  if savetype == savetypes[1]:
    modfnm = odhdf5.getText( modelgrp, 'path' )
    modfnm = dgbhdf5.translateFnm( modfnm, modelfnm )
    model = joblib.load( modfnm )
  elif savetype == savetypes[2]:
    modeldata = modelgrp['object']
    model = pickle.loads( modeldata[:].tostring() )
  elif savetype == xgboostjson and hasXGBoost():
    from xgboost import XGBClassifier, XGBRegressor, \
                        XGBRFClassifier, XGBRFRegressor
    try:
      infods = odhdf5.ensureHasDataset( h5file )
      xgbtyp = odhdf5.getText( infods, 'Model.Class' )
    except Exception as e:
      log_msg('Cannot determine type of model')
      raise e
      
    if xgbtyp == 'XGBClassifier':
      model = XGBClassifier()
    elif xgbtyp == 'XGBRFClassifier':
      model = XGBRFClassifier()
    elif xgbtyp == 'XGBRegressor':
      model = XGBRegressor()
    elif xgbtyp == 'XGBRFRegressor':
      model = XGBRFRegressor()
    if model != None:
      modfnm = odhdf5.getText( modelgrp, 'path' )
      modfnm = dgbhdf5.translateFnm( modfnm, modelfnm )
      model.load_model( modfnm )
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
