#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Nov 2018
#
# _________________________________________________________________________
# various tools machine learning using Keras platform
#


import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical

import dgbpy.keystr as dgbkeys
from dgbpy import hdf5 as dgbhdf5
from dgbpy import dgbkeras
from dgbpy import keras_fix

class TrainingSequence(Sequence):
  def __init__(self,trainbatch,forvalidation,model,exfilenm=None,batch_size=1,with_augmentation=True):
      self._trainbatch = trainbatch
      self._forvalid = forvalidation
      self._model = model
      self._batch_size = batch_size
      self._augmentation = with_augmentation
      self._channels_format = dgbkeras.get_data_format(model)
      self._infos = self._trainbatch[dgbkeys.infodictstr]
      self._data_IDs = []
      if exfilenm == None:
        self._exfilenm = self._infos[dgbkeys.filedictstr]
      else:
        self._exfilenm = exfilenm
      self._nrclasses = 0
      if self._infos[dgbkeys.classdictstr] and not dgbhdf5.isImg2Img(self._infos):
        self._nrclasses = dgbhdf5.getNrClasses( self._infos )

  def __len__(self):
      return int(np.floor(len(self._data_IDs)/float(self._batch_size)))

  def set_chunk(self,ichunk):
      infos = self._infos
      nbchunks = len(infos[dgbkeys.trainseldicstr])
      if nbchunks > 1:
          from dgbpy import mlapply as dgbmlapply
          trainbatch = dgbmlapply.getScaledTrainingDataByInfo( infos, flatten=False,
                                                 scale=True, ichunk=ichunk )
      else:
          trainbatch = self._trainbatch
          
      if self._forvalid:
          if not dgbkeys.xvaliddictstr in trainbatch or \
             not dgbkeys.yvaliddictstr in trainbatch:
              return False
          x_data = trainbatch[dgbkeys.xvaliddictstr]
          y_data = trainbatch[dgbkeys.yvaliddictstr]
      else:
          if not dgbkeys.xtraindictstr in trainbatch or \
             not dgbkeys.ytraindictstr in trainbatch:
              return False
          x_data = trainbatch[dgbkeys.xtraindictstr]
          y_data = trainbatch[dgbkeys.ytraindictstr]
      model = self._model
      self._x_data = dgbkeras.adaptToModel( model, x_data )
      if len(y_data.shape) > 2:
          self._y_data = dgbkeras.adaptToModel( model, y_data )
      else:
          self._y_data = y_data
      inp_shape = self._x_data.shape[1:]
      if self._augmentation and len(inp_shape) == 4:
          if self._channels_format == 'channels_first':
              self._rotdims = (2,3)
              cubesz = self._x_data.shape[2:4]
          else:
              self._rotdims = (1,2)
              cubesz = self._x_data.shape[1:3]
          if cubesz[0] == cubesz[1]:
              self._rot = range(4)
              self._rotidx = self._rot
          else:
              self._rot = range(2)
              self._rotidx = range(0,4,2)
      elif self._augmentation and len(inp_shape) == 3:
          self._rot = range(2)
          self._rotidx = range(0,4,2)
          if self._channels_format == 'channels_first':
              self._rotdims = 2
          else:
              self._rotdims = 1
      else:
          self._rot = range(1)
      self._data_IDs = range(len(self._x_data)*len(self._rot))
      self.on_epoch_end()
      return True

  def on_epoch_end(self):
      self._indexes = np.arange(len(self._data_IDs))
      np.random.shuffle(self._indexes)
      
  def __getitem__(self, index):
      islast = index==(len(self)-1)
      bsize = self._batch_size
      if islast:
        indexes = self._indexes[index*bsize:]
      else:
        indexes = self._indexes[index*bsize:(index+1)*bsize]
      data_IDs_temp = [self._data_IDs[k] for k in indexes]
      return self.__data_generation(data_IDs_temp)
    
  def __data_generation(self, data_IDs_temp):
      x_data = self._x_data
      y_data = self._y_data
      inp_shape = x_data.shape[1:]
      out_shape = y_data.shape[1:]
      nrpts = len(data_IDs_temp)
      X = np.empty( (nrpts,*inp_shape), dtype=x_data.dtype )
      Y = np.empty( (nrpts,*out_shape), dtype=y_data.dtype )
      nrrot = len(self._rot)
      if nrrot == 1:
          X = x_data[data_IDs_temp]
          Y = y_data[data_IDs_temp]
      else:
          iindex,frem = np.divmod(data_IDs_temp,nrrot)
          n = 0
          rotdims = self._rotdims
          flip2d = not isinstance( rotdims, tuple )
          for j,k in zip(self._rot,self._rotidx):
              rotidx = iindex[frem==j]
              m = len(rotidx)
              if m > 0:
                  wrrg = range(n,n+m)
                  if flip2d:
                      if k == 0:
                          X[wrrg] = x_data[rotidx]
                      else:
                          X[wrrg] = np.fliplr(x_data[rotidx])
                  else:
                      X[wrrg] = np.rot90(x_data[rotidx],k,rotdims)
                  if len(y_data.shape) > 2:
                      if flip2d:
                          if k == 0:
                              Y[wrrg] = y_data[rotidx]
                          else:
                              Y[wrrg] = np.fliplr(y_data[rotidx])
                      else:
                          Y[wrrg] = np.rot90(y_data[rotidx],k,rotdims)
                  else:
                      Y[wrrg] = y_data[rotidx]
                  n = wrrg.stop
      if self._nrclasses > 0:
          Y = to_categorical(Y,self._nrclasses)
      return (X, Y, [None])

import importlib
import pkgutil
import inspect

from abc import ABC, abstractmethod
from keras import backend
from pathlib import Path


class UserModel(ABC):
  """Abstract base class for user defined Keras machine learning models
  
  This module provides support for users to add their own machine learning
  models to OpendTect.

  It defines an abstract base class. Users derive there own model classes from this base
  class and implement the _make_model static method to define the structure of the keras model.
  The users model definition should be saved in a file name with "mlmodel_" as a prefix and be 
  at the top level of the module search path so it can be discovered.
  
  The "mlmodel_" class should also define some class variables describing the class:
  uiname : str - this is the name that will appear in the user interface
  uidescription : str - this is a short description which may be displayed to help the user
  modtype : str - type of model (must be one of modeltypes)
  dims : str - describes number of input dimensions supported by model, one of '1', '2', '3' or 'any'

  Examples
  --------
    from dgbpy.keras_classes import UserModel
  
    class myModel(UserModel):
      uiname = 'mymodel'
      uidescription = 'short description of model'
      modtype = classifiertypestr
      dims = '3'
      
      def _make_model(self, input_shape, nroutputs, learnrate, data_format):
        inputs = Input(input_shape)
        conv1 = Conv3D(2, (3,3,3), activation='relu', padding='same')(inputs)
        conv1 = Conv3D(2, (3,3,3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling3D(pool,size=(2,2,2))(conv1)
        ...
        conv8 = Conv3D(1, (1,1,1,), activation='sigmoid')(conv7)
      
        model = Model(inputs=[inputs], outputs=[conv8])
        model.compile(optimizer = Adam(lr = 1e-4), loss = cross_entropy_balanced, metrics = ['accuracy'])
        return model
      
    
  """
  mlmodels = []
  
  classifiertypestr = 'classifier'
  regressortypestr = 'regressor'
  img2imgtypestr = 'img2img'
  modeltypes = (classifiertypestr,regressortypestr,img2imgtypestr)
  
  def __init__(self, ):
    self._learnrate = None
    self._nroutputs = None
    self._data_format = None
    self._model = None

  @staticmethod
  def findModels():
    """Static method that searches the PYTHONPATH for modules containing user
    defined Keras machine learning models (UserModels).
    
    The module name must be prefixed by "mlmodel_". All subclasses of the
    UserModel base class is each found module will be added to the mlmodels
    class variable.
    """

    mlm = []

    for _, name, ispkg in pkgutil.iter_modules(path=[Path(__file__).parent.absolute()]):
      if name.startswith("mlmodel_"):
        module = importlib.import_module('.'.join(['dgbpy',name]))
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for (_, c) in clsmembers:
          if issubclass(c, UserModel) & (c is not UserModel):
            mlm.append(c())
        
    for _, name, ispkg in pkgutil.iter_modules():
      if name.startswith('mlmodel_'):
        module = importlib.import_module(name)
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for (_, c) in clsmembers:
          if issubclass(c, UserModel) & (c is not UserModel):
            mlm.append(c())
    return mlm
  
  @staticmethod
  def findName(modname):
    """Static method that searches the found UserModel's for a match with the
    uiname class variable
    
    Parameters
    ----------
    modname : str
    Name (i.e. uiname) of the UserModel to search for.
    
    Returns
    -------
    an instance of the class with the first matching name in the mlmodels
    list or None if no match is found
    
    """
    return next((model for model in UserModel.mlmodels if model.uiname == modname), None)
  
  @staticmethod
  def getModelsByType(model_type, dims):
    """Static method that returns a list of the UserModels filtered by the given
    model type and dimensions
    
    Parameters
    ----------
    modeltype: str
    The type of model to filter by either 'classifier' or 'other'
    dims: str
    The dimensions that the model must support
    
    Returns
    -------
    a list of matching model or None if no match found
    
    """
    if model_type in UserModel.modeltypes:
      return [model for model in UserModel.mlmodels \
                  if model.modtype == model_type and\
                    (model.dims == dims or model.dims == 'any')]
    return None

  @staticmethod
  def getNamesByType(model_type=classifiertypestr, dims='any'):
      models = UserModel.getModelsByType(model_type,dims)
      return [model.uiname for model in models]
  
  @staticmethod
  def isModelType( modelnm, modtype ):
      models = UserModel.getModelsByType( modtype, 'any' )
      for mod in models:
          if mod.uiname == modelnm:
              return True
      return False
  
  @staticmethod
  def isClassifier( modelnm ):
      return UserModel.isModelType( modelnm, UserModel.classifiertypestr )
  
  @staticmethod
  def isRegressor( modelnm ):
      return UserModel.isModelType( modelnm, UserModel.regressortypestr )  
  
  @staticmethod
  def isImg2Img( modelnm ):
      return UserModel.isModelType( modelnm, UserModel.img2imgtypestr )
  
  @abstractmethod
  def _make_model(self, input_shape, nroutputs, learnrate, data_format):
    """Abstract static method that defines a machine learning model.
    
    Must be implemented in the user's derived class
    
    Parameters
    ----------
    input_shape : tuple
    Defines input data shape in the Keras default format for the current backend
    nroutputs : int
    Number of outputs
    learnrate : float
    The step size applied at each iteration to move toward a minimum of the loss function
    
    Returns
    -------
    a compiled keras model
    
    """
    pass

  def model(self, input_shape, nroutputs, learnrate, data_format=backend.image_data_format()):
    """Creates/returns a compiled keras model instance
    
    Parameters
    ----------
    input_shape : tuple
    Defines input data shape (by convention in the machine learning plugin this
    will be in the 'channels_first' format by default. 
    nroutputs : int
    Number of outputs
    learnrate : float
    The step size applied at each iteration to move toward a minimum of the loss function
    data_format: str
    The data format used by the machine learning plugin, will generally be 'channels_first'
    
    Returns
    -------
    a compiled keras model
    
    """
      
    newmodel = self._model is None or input_shape != self._model.input_shape or \
                nroutputs != self._nroutputs or learnrate != self._learnrate or \
                data_format != self._data_format
    if  newmodel:
      self._nroutputs = nroutputs
      self._learnrate = learnrate
      self._model = self._make_model(input_shape,nroutputs,learnrate,data_format)
      self._data_format = dgbkeras.get_data_format( self._model )
    return self._model
  
UserModel.mlmodels = UserModel.findModels()
