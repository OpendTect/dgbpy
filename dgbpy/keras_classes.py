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
from typing import Iterable
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical

import dgbpy.keystr as dgbkeys
from dgbpy import hdf5 as dgbhdf5

def model_info( modelfnm ):
  from dgbpy.dgbkeras import load
  model = load( modelfnm, False )
  mi = model_info_dict( model )
  return (mi['input_shape'], mi['output_shape'], mi['data_format'])

def model_info_dict( keras_model ):
  minfo = {}
  minfo['input_shape'] = keras_model.input_shape
  minfo['output_shape'] = keras_model.output_shape
  minfo['data_format'] = next((layer.data_format if hasattr(layer, 'data_format') else 'channels_last' for layer in keras_model.layers))
  return minfo

class TrainingSequence(Sequence):
  def __init__(self,trainbatch,forvalidation,model,exfilenm=None,batch_size=1,\
               scale=None,transform=list(),transform_copy=True,tempnm=None):
      from dgbpy.dgbkeras import get_data_format
      self._trainbatch = trainbatch
      self._forvalid = forvalidation
      self._model = model
      self._nrdone = -1
      self._doshuffle = True
      self._tempnm = tempnm
      self._lastsaved = datetime.now()
      self._batch_size = batch_size
      self._channels_format = get_data_format(model)
      self._infos = self._trainbatch[dgbkeys.infodictstr]
      self._data_IDs = []
      self.ndims = self._getDims(self._infos)
      self.transform = []
      self.transform_seed = dgbhdf5.getSeed(self._infos)
      self.transform_copy = transform_copy
      if exfilenm == None:
        self._exfilenm = self._infos[dgbkeys.filedictstr]
      else:
        self._exfilenm = exfilenm
      self._nrclasses = 0
      if self._infos[dgbkeys.classdictstr]:
        self._nrclasses = dgbhdf5.getNrClasses( self._infos )
        if dgbhdf5.isImg2Img(self._infos) and self._nrclasses <= 2:
          self._nrclasses = 0
      self.transform_init(scale, transform, transform_copy=transform_copy)

  def transform_init(self,scale,transform,transform_copy=True):
      from dgbpy import transforms as T
      scale, self.isDefScaler = dgbhdf5.isDefaultScaler(scale, self._infos)
      if not self.isDefScaler:
        if self._forvalid: transform = [scale]
        else: transform.append(scale)
      self.transform = T.TransformCompose(transform, self._infos, self.ndims, create_copy=transform_copy)
      self.transform_multiplier = self.transform.multiplier

  def _getDims(self, info):
      from dgbpy.dgbkeras import get_model_shape, getModelDims
      attribs = dgbhdf5.getNrAttribs(info)
      model_shape = get_model_shape(info[dgbkeys.inpshapedictstr], attribs, True)
      ndims = getModelDims(model_shape, True)
      return ndims

  def __len__(self):
      return int(np.floor(len(self._data_IDs)/float(self._batch_size)))

  def enable_shuffling(self, yn):
    self._doshuffle = yn

  def set_transform_seed(self):
    if not isinstance(self.transform, list) and self.transform_seed and not self._forvalid:
      self.transform_seed+=1
      self.transform.set_uniform_generator_seed(self.transform_seed, len(self._data_IDs))

  def set_chunk(self,ichunk):
      infos = self._infos
      nbchunks = len(infos[dgbkeys.trainseldicstr])
      if nbchunks > 1 or dgbhdf5.isCrossValidation(infos):
        return self.set_fold(ichunk, 1) #set first fold initially for each chunk
      else:
          trainbatch = self._trainbatch
      return self.get_data(trainbatch)

  def set_fold(self,ichunk,ifold):
    infos = self._infos
    from dgbpy import mlapply as dgbmlapply
    trainbatch = dgbmlapply.getScaledTrainingDataByInfo( infos,
                                              flatten=False,
                                              scale=self.isDefScaler, ichunk=ichunk, ifold=ifold )
    return self.get_data(trainbatch)

  def get_data(self, trainbatch):
    if self._forvalid:
          if not dgbkeys.xvaliddictstr in trainbatch or \
             not dgbkeys.yvaliddictstr in trainbatch:
              return False
          self._x_data = trainbatch[dgbkeys.xvaliddictstr]
          self._y_data = trainbatch[dgbkeys.yvaliddictstr]
    else:
        if not dgbkeys.xtraindictstr in trainbatch or \
            not dgbkeys.ytraindictstr in trainbatch:
            return False
        self._x_data = trainbatch[dgbkeys.xtraindictstr]
        self._y_data = trainbatch[dgbkeys.ytraindictstr]
    self._data_IDs = range((len(self._x_data)*len(self.transform_multiplier)))
    self.on_epoch_end()
    return True

  def on_epoch_end(self):
      self._nrdone = self._nrdone+1
      if self._tempnm != None and self._nrdone > 0:
          now = datetime.now()
          if now - self._lastsaved > timedelta(minutes=10):
              from dgbpy import dgbkeras
              dgbkeras.save( self._model, self._tempnm )
              self._lastsaved = now
      self._indexes = np.arange(len(self._data_IDs))
      if self._doshuffle and not self._forvalid:
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
      from dgbpy import dgbkeras
      x_data = self._x_data
      y_data = self._y_data
      inp_shape = x_data.shape[1:]
      out_shape = y_data.shape[1:]
      nrpts = len(data_IDs_temp)
      X = np.empty( (nrpts,*inp_shape), dtype=x_data.dtype )
      Y = np.empty( (nrpts,*out_shape), dtype=y_data.dtype )
      idx, rem = np.divmod(data_IDs_temp, len(self.transform_multiplier))
      for i, (ID, _ID) in enumerate(zip(idx, data_IDs_temp)):
        if self.transform:
            X[i,], Y[i,] = self.transform(x_data[ID], y_data[ID], _ID, transform_idx=rem[i])
        else:
            X[i,], Y[i,] = x_data[ID], y_data[ID]
      dictinpshape = self._infos[dgbkeys.inpshapedictstr]
      dictinpshape = tuple( dictinpshape ) if not isinstance(dictinpshape, int) else (dictinpshape,)
      X = dgbkeras.adaptToModel( self._model, X, dictinpshape )
      if len(Y.shape) > 2:
          Y = dgbkeras.adaptToModel( self._model, Y, dictinpshape )
      if self._nrclasses > 0:
          Y = to_categorical(Y,self._nrclasses)
      return (X, Y)

import importlib
import pkgutil
import inspect
import os,re
import odpy.common as odcommon

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

  @classmethod
  def is_valid(cls, dim_type):
    if not isinstance(dim_type, Iterable):
      return isinstance(dim_type, cls)
    return all(isinstance(dim_type, cls) for dim_type in dim_type)

class UserModel(ABC):
  """Abstract base class for user defined Keras machine learning models

  This module provides support for users to add their own machine learning
  models to OpendTect.

  It defines an abstract base class. Users derive there own model classes from this base
  class and implement the _make_model static method to define the structure of the keras model.
  The users model definition should be saved in a file name with "mlmodel_keras" as a prefix and be
  at the top level of the module search path so it can be discovered.

  The "mlmodel_keras" class should also define some class variables describing the class:
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

    The module name must be prefixed by "mlmodel_keras". All subclasses of the
    UserModel base class is each found module will be added to the mlmodels
    class variable.
    """

    mlm = []

    dgbpypath = Path(__file__).parent.absolute()
    dgbpypathstr = os.fsdecode( dgbpypath )
    for _, name, ispkg in pkgutil.iter_modules(path=[dgbpypathstr]):
      if name.startswith("mlmodel_keras"):
        module = importlib.import_module('.'.join(['dgbpy',name]))
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for (_, c) in clsmembers:
          if issubclass(c, UserModel) & (c is not UserModel):
            mlm.append(c())

    try:
      py_settings_path = odcommon.get_settings_filename( 'settings_python' )
      pattern = r'^PythonPath\.\d+: (.+)$'
      py_paths = []
      with open(py_settings_path, 'r') as f:
        for line in f.readlines():
          match = re.match(pattern, line)
          if match and os.path.exists(match.group(1)): py_paths.append(match.group(1))
    except FileNotFoundError: pass

    for path in py_paths:
      for root, _, files in os.walk(path):
        for file in files:
          if file.startswith('mlmodel_keras') and file.endswith('.py'):
            relpath = os.path.relpath(root, path)
            if relpath != '.': name = '.'.join([relpath, file[:-3]]).replace(os.path.sep, '.')  
            else: name = file[:-3]
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
       DimType.is_valid(dim_type):
      return [model for model in UserModel.mlmodels \
          if (model.predtype == pred_type or pred_type == DataPredType.Any) and\
	           (model.outtype == out_type or out_type == OutputType.Any) and\
             (model.dimtype == dim_type or model.dimtype == DimType.Any or\
                (isinstance(model.dimtype, Iterable) and dim_type in model.dimtype))]
    return None

  @staticmethod
  def getNamesByType(pred_type, out_type, dim_type):
      models = UserModel.getModelsByType(pred_type, out_type, dim_type)
      model_names = []
      for model in models:
        if model.uiname not in model_names: model_names.append(model.uiname)
      return model_names

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
      from dgbpy import dgbkeras
      self._nroutputs = nroutputs
      self._learnrate = learnrate
      self._model = self._make_model(modshape,nroutputs,learnrate)
      self._data_format = dgbkeras.get_data_format( self._model )
    return self._model

UserModel.mlmodels = UserModel.findModels()
