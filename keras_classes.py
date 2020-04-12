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
      self._channels_format = dgbkeras.getDataFormat(model)
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
      if self._channels_format == 'channels_last':
          lastaxis = len(x_data.shape)-1
          x_data = np.swapaxes(x_data,1,lastaxis)
          if len(y_data.shape) > 2:
              y_data = np.swapaxes(y_data,1,lastaxis)
      self._x_data = dgbkeras.adaptToModel( model, x_data, sample_data_format=self._channels_format)
      if len(y_data.shape) > 2:
          self._y_data = dgbkeras.adaptToModel( model, y_data, sample_data_format=self._channels_format )
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
          Y = keras.utils.to_categorical(Y,self._nrclasses)
      return X, Y

