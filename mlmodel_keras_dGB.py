#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Wayne Mogg
# DATE     : June 2020
#
# dGB Keras machine learning models in UserModel format
#

from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, RMSprop
from keras import backend as kb

from dgbpy.keras_classes import UserModel, DataPredType, OutputType, DimType
import dgbpy.dgbkeras as dgbkeras

def _to_tensor(x, dtype):
  from keras.optimizers import tf
  x = tf.convert_to_tensor(value=x)
  if x.dtype != dtype:
    x = tf.cast(x, dtype)
  return x

def root_mean_squared_error(y_true, y_pred):
  return kb.sqrt(kb.mean(kb.square(y_pred - y_true)))

def cross_entropy_balanced(y_true, y_pred):
  from keras.models import K
  from keras.optimizers import tf
  _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
  y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
  y_pred   = tf.math.log(y_pred/ (1 - y_pred))

  y_true = tf.cast(y_true, tf.float32)
  count_neg = tf.reduce_sum(input_tensor=1. - y_true)
  count_pos = tf.reduce_sum(input_tensor=y_true)
  beta = count_neg / (count_neg + count_pos)
  pos_weight = beta / (1 - beta)
  cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
  cost = tf.reduce_mean(input_tensor=cost * (1 - beta))
  return tf.compat.v1.where(tf.equal(count_pos, 0.0), 0.0, cost)

def dGBUNet(model_shape, nroutputs, predtype):
    input_shape = model_shape
    data_format = 'channels_last'
      
    ndim = dgbkeras.getModelDims( model_shape, data_format )
    
    conv = pool = upsamp = None
    if ndim==3:
      conv = Conv3D
      pool = MaxPooling3D
      upsamp = UpSampling3D
    elif ndim==2:
      conv = Conv2D
      pool = MaxPooling2D
      upsamp = UpSampling2D
    elif ndim==1:
      conv = Conv1D
      pool = MaxPooling1D
      upsamp = UpSampling1D
      
    unet_smallsz = (2,64)
    unet_mediumsz = (16,512)
    unet_largesz = (32,512)

    unetnszs= unet_mediumsz
    poolsz1 = 2
    poolsz2 = 2
    poolsz3 = 2
    upscalesz = 2
    filtersz1 = unetnszs[0]
    filtersz2 = filtersz1 * poolsz2
    filtersz3 = filtersz2 * poolsz3
    filtersz4 = unetnszs[1]
    axis = -1

    params = dict(kernel_size=3, activation='relu', padding='same', data_format=data_format)

    inputs = Input(input_shape)
    conv1 = conv(filtersz1, **params)(inputs)
    conv1 = conv(filtersz1, **params)(conv1)
    
    pool1 = pool(pool_size=poolsz1, data_format=data_format)(conv1)
    
    conv2 = conv(filtersz2, **params)(pool1)
    conv2 = conv(filtersz2, **params)(conv2)
    pool2 = pool(pool_size=poolsz2, data_format=data_format)(conv2)
    
    conv3 = conv(filtersz3, **params)(pool2)
    conv3 = conv(filtersz3, **params)(conv3)
    pool3 = pool(pool_size=poolsz3, data_format=data_format)(conv3)
    
    conv4 = conv(filtersz4, **params)(pool3)
    conv4 = conv(filtersz4, **params)(conv4)
    
    up5 = concatenate([upsamp(size=upscalesz,data_format=data_format)(conv4), conv3], axis=axis)
    conv5 = conv(filtersz3, **params)(up5)
    conv5 = conv(filtersz3, **params)(conv5)
    
    up6 = concatenate([upsamp(size=poolsz2,data_format=data_format)(conv5), conv2], axis=axis)
    conv6 = conv(filtersz2, **params)(up6)
    conv6 = conv(filtersz2, **params)(conv6)
    
    up7 = concatenate([upsamp(size=poolsz1,data_format=data_format)(conv6), conv1], axis=axis)
    conv7 = conv(filtersz1, **params)(up7)
    conv7 = conv(filtersz1, **params)(conv7)
    
    if isinstance(predtype, DataPredType) and predtype==DataPredType.Continuous:
      nrout = nroutputs
      activation = 'linear'
    else:
      if nroutputs == 2:
        nrout = 1
        activation = 'sigmoid'
      else:
        nrout = nroutputs
        activation = 'softmax'
    conv8 = conv(nrout, 1, activation=activation, data_format=data_format)(conv7)
      
    model = Model(inputs=[inputs], outputs=[conv8])
    return model

class dGB_UnetSeg(UserModel):
  uiname = 'dGB UNet Segmentation'
  uidescription = 'dGBs Unet image segmentation'
  predtype = DataPredType.Classification
  outtype = OutputType.Image
  dimtype = DimType.Any
  
  def _make_model(self, model_shape, nroutputs, learnrate):
    model = dGBUNet(model_shape, nroutputs, self.predtype)
    if nroutputs<=2:
      loss = cross_entropy_balanced
    else:
      loss = 'categorical_crossentropy'

    model.compile(optimizer = Adam(lr = learnrate), loss = loss, metrics = ['accuracy'])

    return model
  
class dGB_UnetReg(UserModel):
  uiname = 'dGB UNet Regression'
  uidescription = 'dGBs Unet image regression'
  predtype = DataPredType.Continuous
  outtype = OutputType.Image
  dimtype = DimType.Any
  
  def _make_model(self, model_shape, nroutputs, learnrate):
    model = dGBUNet(model_shape, nroutputs, self.predtype)
    model.compile(optimizer = Adam(lr = learnrate), loss = 'mse', metrics = ['mae'])
    return model
  
def dGBLeNet(model_shape, nroutputs, predtype):
    
  input_shape = model_shape
  data_format = 'channels_last'
  bnaxis = -1
  if not dgbkeras.need_channels_last():
  #Prefer channels_first data_format for efficiency if supported
  # CPU training requires channels_last format due to TensorFlow bug
      data_format = 'channels_first'
      input_shape = (model_shape[-1], *model_shape[0:-1])
      bnaxis = 1
      
  ndim = dgbkeras.getModelDims(model_shape, data_format)

  conv = None
  if ndim==3:
    conv = Conv3D
  elif ndim==2:
    conv = Conv2D
  elif ndim==1 or ndim==0:
    conv = Conv1D
    
  filtersz = 50
  densesz = 10  
  kernel_sz1 = 5
  kernel_sz2 = 3
  stride_sz1 = 4
  stride_sz2 = 2
  dropout = 0.2

  inputs = Input(input_shape)
  conv1 = conv(filtersz, kernel_sz1, strides=stride_sz1, padding='same', data_format=data_format)(inputs)
  conv1 = BatchNormalization(axis=bnaxis)(conv1)
  conv1 = Activation('relu')(conv1)
  conv2 = conv(filtersz, kernel_sz2, strides=stride_sz2, padding='same', data_format=data_format)(conv1)
  conv2 = Dropout(dropout)(conv2)
  conv2 = BatchNormalization(axis=bnaxis)(conv2)
  conv2 = Activation('relu')(conv2)
  conv3 = conv(filtersz, kernel_sz2, strides=stride_sz2, padding='same', data_format=data_format)(conv2)
  conv3 = Dropout(dropout)(conv3)
  conv3 = BatchNormalization(axis=bnaxis)(conv3)
  conv3 = Activation('relu')(conv3)
  conv4 = conv(filtersz, kernel_sz2, strides=stride_sz2, padding='same', data_format=data_format)(conv3)
  conv4 = Dropout(dropout)(conv4)
  conv4 = BatchNormalization(axis=bnaxis)(conv4)
  conv4 = Activation('relu')(conv4)
  conv5 = conv(filtersz, kernel_sz2, strides=stride_sz2, padding='same', data_format=data_format)(conv4)
  dense1 = Flatten()(conv5)
  dense1 = Dense(filtersz)(dense1)
  dense1 = BatchNormalization(axis=bnaxis)(dense1)
  dense1 = Activation('relu')(dense1)
  dense2 = Dense(densesz)(dense1)
  dense2 = BatchNormalization(axis=bnaxis)(dense2)
  dense2 = Activation('relu')(dense2)
  dense3 = None
  if isinstance(predtype, DataPredType) and predtype==DataPredType.Continuous:
    dense3 = Dense(nroutputs, activation='linear')(dense2)
  else:
    dense3 = Dense(nroutputs)(dense2)
    dense3 = BatchNormalization(axis=bnaxis)(dense3)
    dense3 = Activation('softmax')(dense3)
    
  model = Model(inputs=[inputs], outputs=[dense3])
  return model
        
class dGB_LeNet_Classifier(UserModel):
  uiname = 'dGB LeNet classifier'
  uidescription = 'dGBs LeNet classifier Keras model in UserModel form'
  predtype = DataPredType.Classification
  outtype = OutputType.Pixel
  dimtype = DimType.Any
  
  def _make_model(self, input_shape, nroutputs, learnrate):
    model = dGBLeNet(input_shape, nroutputs, self.predtype)
    
    loss = 'binary_crossentropy'
    if nroutputs>2:
      loss = 'categorical_crossentropy'
    
    model.compile(optimizer=Adam(lr=learnrate), loss=loss, metrics=['accuracy'])
    
    return model

class dGB_LeNet_Regressor(UserModel):
  uiname = 'dGB LeNet regressor'
  uidescription = 'dGBs LeNet regressor Keras model in UserModel form'
  predtype = DataPredType.Continuous
  outtype = OutputType.Pixel
  dimtype = DimType.Any
  
  def _make_model(self, input_shape, nroutputs, learnrate):
    model = dGBLeNet(input_shape, nroutputs, self.predtype)
      
    model.compile(optimizer=Adam(lr=learnrate), loss='mse', metrics=['mae'])
    
    return model

