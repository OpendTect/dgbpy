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
          
      if not dgbkeys.xtraindictstr in trainbatch:
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
      inp_shape = self._x_data.shape[1:]
      if self._augmentation and len(inp_shape) == 4:
        self._nrrot = 4
      elif self._augmentation and len(inp_shape) == 3:
        self._nrrot = 2
      else:
        self._nrrot = 1
      self._nrrot = 1
      self._data_IDs = range(len(self._x_data)*self._nrrot)
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
      nrrot = self._nrrot
      iindex,frem = np.divmod(data_IDs_temp,nrrot)
      if nrrot == 4:
          if len(y_data.shape) > 2:
              for i in range(1,nrrot):
                  for k,idx in zip(range(i*nrpts,(i+1)*nrpts),data_IDs_temp):
                      X[k] = np.reshape(np.rot90(x_data[idx],i,(0,1)), inp_shape)
                      Y[k] = np.reshape(np.rot90(y_data[idx],i,(0,1)), out_shape)
          else:
              for i in range(1,nrrot):
                  for k,idx in zip(range(i*nrpts,(i+1)*nrpts),data_IDs_temp):
                      X[k] = np.reshape(np.rot90(x_data[idx],i,(0,1)), inp_shape)
              Y[nrpts:2*nrpts] = y_data[data_IDs_temp]
              Y[2*nrpts:3*nrpts] = y_data[data_IDs_temp]
              Y[3*nrpts:] = y_data[data_IDs_temp]
      elif nrrot == 2:
          norotidx = iindex[frem==0]
          n = len(norotidx)
          if n > 0:
              X[:n] = x_data[norotidx]
              Y[:n] = y_data[norotidx]
          rotidx = iindex[frem==1]
          n = len(rotidx)
          if n > 0:
              X[n:] = np.flip(x_data[rotidx])
              if len(y_data.shape) > 2:
                  Y[n:] = np.flip(y_data[rotidx])
              else:
                  Y[n:] = y_data[rotidx]
      else:
          X = x_data[iindex]
          Y = y_data[iindex]
      if self._nrclasses > 0:
          Y = keras.utils.to_categorical(Y,self._nrclasses)
      return X, Y

