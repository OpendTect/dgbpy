#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Nov 2018
#
# _________________________________________________________________________
# various tools machine learning using Keras platform
#

from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical

import dgbpy.keystr as dgbkeys
from dgbpy import hdf5 as dgbhdf5
from dgbpy import dgbkeras
from dgbpy import keras_fix

class TrainingSequence(Sequence):
  def __init__(self,trainbatch,forvalidation,model,exfilenm=None,batch_size=1,\
               with_augmentation=True,tempnm=None):
      self._trainbatch = trainbatch
      self._forvalid = forvalidation
      self._model = model
      self._nrdone = -1
      self._tempnm = tempnm
      self._lastsaved = datetime.now()
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
      if self._infos[dgbkeys.classdictstr]:
        self._nrclasses = dgbhdf5.getNrClasses( self._infos )
        if dgbhdf5.isImg2Img(self._infos) and self._nrclasses <= 2:
          self._nrclasses = 0

  def __len__(self):
      return int(np.floor(len(self._data_IDs)/float(self._batch_size)))

  def set_chunk(self,ichunk):
      infos = self._infos
      nbchunks = len(infos[dgbkeys.trainseldicstr])
      if nbchunks > 1:
          from dgbpy import mlapply as dgbmlapply
          trainbatch = dgbmlapply.getScaledTrainingDataByInfo( infos,
                                                 flatten=False,
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
      self._nrdone = self._nrdone+1
      if self._tempnm != None and self._nrdone > 0:
          now = datetime.now()
          if now - self._lastsaved > timedelta(minutes=10):
              dgbkeras.save( self._model, self._tempnm )
              self._lastsaved = now
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
from enum import Enum

class DataPredType(Enum):
  Continuous = 'Continuous Data'
  Classification = 'Classification Data'
  Segmentation = 'Segmentation'
  Any = 'Any'

class OutputType(Enum):
  Pixel = 1
  Image = 2
  Any = 3
    
class DimType(Enum):
  D1 = 1
  D2 = 2
  D3 = 3
  Any = 4

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
  predtype : DataPredType enum - type of prediction (must be member of DataPredType enum)
  outtype: OutputType enum - output shape type (OutputType.Pixel or OutputType.Image)
  dimtype : DimType enum - the input dimensions supported by model (must be member of DimType enum)

  Examples
  --------
    from dgbpy.keras_classes import UserModel, DataPredType, OutputType, DimType
  
    class myModel(UserModel):
      uiname = 'mymodel'
      uidescription = 'short description of model'
      predtype = DataPredType.Classification
      outtype = OutputType.Pixel
      dimtype = DimType.D3
      
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
  def getModelsByType(pred_type, out_type, dim_type):
    """Static method that returns a list of the UserModels filtered by the given
    prediction, output and dimension types
    
    Parameters
    ----------
    pred_type: DataPredType enum
    The prediction type of the model to filter by
    out_type: OutputType enum
    The output shape type of the model to filter by
    dim_type: DimType enum
    The dimensions that the model must support
    
    Returns
    -------
    a list of matching model or None if no match found
    
    """
    if isinstance(pred_type, DataPredType) and isinstance(out_type, OutputType) and\
       isinstance(dim_type, DimType) :
      return [model for model in UserModel.mlmodels \
          if (model.predtype == pred_type or pred_type == DataPredType.Any) and\
	     (model.outtype == out_type or out_type == OutputType.Any) and\
             (model.dimtype == dim_type or model.dimtype == DimType.Any)]
    return None

  @staticmethod
  def getNamesByType(pred_type, out_type, dim_type):
      models = UserModel.getModelsByType(pred_type, out_type, dim_type)
      return [model.uiname for model in models]
  
  @staticmethod
  def isPredType( modelnm, pred_type ):
      models = UserModel.getModelsByType( pred_type, OutputType.Any, DimType.Any )
      for mod in models:
          if mod.uiname == modelnm:
              return True
      return False
  
  @staticmethod
  def isOutType( modelnm, out_type ):
      models = UserModel.getModelsByType( DataPredType.Any, out_type, DimType.Any )
      for mod in models:
          if mod.uiname == modelnm:
              return True
      return False
  
  @staticmethod
  def isClassifier( modelnm ):
      return UserModel.isPredType( modelnm, DataPredType.Classification )
  
  @staticmethod
  def isRegressor( modelnm ):
      return UserModel.isPredType( modelnm, DataPredType.Continuous )  
  
  @staticmethod
  def isImg2Img( modelnm ):
      return UserModel.isOutType( modelnm, OutputType.Image )
  
  @abstractmethod
  def _make_model(self, input_shape, nroutputs, learnrate, data_format):
    """Abstract static method that defines a machine learning model.
    
    Must be implemented in the user's derived class
    
    Parameters
    ----------
    input_shape : tuple
    Defines input data shape in the Keras default data_format for the current backend.
    For the TensorFlow backend the default data_format is 'channels_last'
    nroutputs : int (number of discrete classes for a classification)
    Number of outputs
    learnrate : float
    The step size applied at each iteration to move toward a minimum of the loss function
    
    Returns
    -------
    a compiled keras model
    
    """
    pass

  def model(self, input_shape, nroutputs, learnrate, data_format='channels_first'):
    """Creates/returns a compiled keras model instance
    
    Parameters
    ----------
    input_shape : tuple
    Defines input data shape arranged as per the data_format setting. 
    nroutputs : int (number of discrete classes for a classification)
    Number of outputs
    learnrate : float
    The step size applied at each iteration to move toward a minimum of the loss function
    data_format: str
    The data format used. The machine learning plugin uses 'channels_first' data_format.
    
    Returns
    -------
    a compiled keras model
    
    """
    modshape = input_shape
    if data_format=='channels_first' and tf.keras.backend.image_data_format()=='channels_last':
      modshape = (*input_shape[1:], input_shape[0])
    elif data_format=='channels_last' and tf.keras.backend.image_data_format()=='channels_first':
      modshape = (input_shape[-1], *input_shape[0:-1])
      
    newmodel = self._model is None or modshape != self._model.input_shape or \
                nroutputs != self._nroutputs or learnrate != self._learnrate
    if  newmodel:
      self._nroutputs = nroutputs
      self._learnrate = learnrate
      self._model = self._make_model(modshape,nroutputs,learnrate)
      self._data_format = dgbkeras.get_data_format( self._model )
    return self._model
  
UserModel.mlmodels = UserModel.findModels()
