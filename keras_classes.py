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
import keras
from keras.utils import Sequence

import dgbpy.keystr as dgbkeys
from dgbpy import hdf5 as dgbhdf5
from dgbpy import dgbkeras


class TrainingSequence(Sequence):
  def __init__(self,trainbatch,forvalidation,model,exfilenm=None,batch_size=1,with_augmentation=True):
      self._trainbatch = trainbatch
      self._forvalid = forvalidation
      self._model = model
      self._batch_size = batch_size
      self._augmentation = with_augmentation
      self._infos = self._trainbatch[dgbkeys.infodictstr]
      if exfilenm == None:
        self._exfilenm = self._infos[dgbkeys.filedictstr]
      else:
        self._exfilenm = exfilenm
      self._nrclasses = 0
      if self._infos[dgbkeys.classdictstr] and not dgbhdf5.isImg2Img(self._infos):
        self._nrclasses = len(dgbhdf5.getNrClasses( self._infos ))

  def __len__(self):
      return int(np.floor(len(self._x_data)/self._batch_size))

  def __getitem__(self, index):
      bsize = self._batch_size
      indexes = self._indexes[index*bsize:(index+1)*bsize]
      data_IDs = range(len(self._x_data))
      data_IDs_temp = [data_IDs[k] for k in indexes]
      return self.__data_generation(data_IDs_temp)
    
  def set_chunk(self,ichunk):
      infos = self._infos
      nbchunks = len(infos[dgbkeys.trainseldicstr])
      if nbchunks > 1:
          trainbatch = dgbmlapply.getScaledTrainingDataByInfo( infos, flatten=False,
                                                 scale=True, ichunk=ichunk )
      else:
          trainbatch = self._trainbatch
          
      if not dgbkeys.xtraindictstr in trainbatch:
          log_msg('No data to train the model')
          return False
      if self._forvalid:
          self._x_data = trainbatch[dgbkeys.xvaliddictstr]
          self._y_data = trainbatch[dgbkeys.yvaliddictstr]
      else:
          self._x_data = trainbatch[dgbkeys.xtraindictstr]
          self._y_data = trainbatch[dgbkeys.ytraindictstr]
      model = self._model
      self._x_data = dgbkeras.adaptToModel( model, self._x_data )
      if len(self._y_data.shape) > 2:
          self._y_data = dgbkeras.adaptToModel( model, self._y_data )
      if self._nrclasses > 0:
          self._y_data = keras.utils.to_categorical(self._y_data,self._nrclasses)
      inp_shape = self._x_data.shape[1:]
      if self._augmentation and len(inp_shape) == 4:
        self._nrrot = 4
      elif self._augmentation and len(inp_shape) == 3:
        self._nrrot = 2
      else:
        self._nrrot = 1
      self.on_epoch_end()
      return True

  def on_epoch_end(self):
      self._indexes = np.arange(len(self._x_data))
      np.random.shuffle(self._indexes)
      
  def __data_generation(self, data_IDs_temp):
      idx = data_IDs_temp[0]
      x_data = self._x_data
      y_data = self._y_data
      inp_shape = x_data.shape[1:]
      out_shape = y_data.shape[1:]
      nrrot = self._nrrot
      X = np.empty( (nrrot,*inp_shape), dtype=x_data.dtype )
      Y = np.empty( (nrrot,*out_shape), dtype=y_data.dtype )
      if nrrot == 4:
        for i in range(nrrot):
          X[i] = np.reshape(np.rot90(x_data[idx],i,(0,1)), inp_shape)
          Y[i] = np.reshape(np.rot90(y_data[idx],i,(0,1)), inp_shape)
      else:
        X[0] = x_data[idx]
        Y[0] = y_data[idx]
        if nrrot == 2:
          X[1] = np.fliplr(x_data[idx])
          Y[1] = np.fliplr(y_data[idx])
      return X, Y

