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

class dGB_Unet(UserModel):
  uiname = 'dGB UNet'
  uidescription = 'dGBs Unet img2img Keras model in UserModel form'
  predtype = DataPredType.Classification
  outtype = OutputType.Image
  dimtype = DimType.Any
  
  unet_smallsz = (2,64)
  unet_mediumsz = (16,512)
  unet_largesz = (32,512)
  
  def _make_model(self, model_shape, nroutputs, learnrate, data_format):
    
    input_shape = model_shape
    if data_format == 'channels_first':
      #Tensorflow bug; always bad accuracy, no training
      data_format = 'channels_last'
      dims = model_shape[1:]
      input_shape = ( *dims, model_shape[0] )
      
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
      
    unetnszs= self.unet_mediumsz
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
    
    conv8 = conv(nroutputs, 1, activation='sigmoid', data_format=data_format)(conv7)
    
    model = Model(inputs=[inputs], outputs=[conv8])
    
    model.compile(optimizer = Adam(lr = learnrate), loss = cross_entropy_balanced, metrics = ['accuracy'])

    return model
  
def dGBLeNet(model_shape, nroutputs, data_format):
    
  input_shape = model_shape
  if dgbkeras.need_channels_last() and data_format == 'channels_first':
      #Tensorflow bug; cannot use channel_first on CPU: crash or no accuracy
      data_format = 'channels_last'
      dims = model_shape[1:]
      input_shape = ( *dims, model_shape[0] )
      
  ndim = dgbkeras.getModelDims(model_shape, data_format)
  axis = -1
  if data_format == 'channels_first':
    axis = 1
  
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
  conv1 = BatchNormalization(axis=axis)(conv1)
  conv1 = Activation('relu')(conv1)
  conv2 = conv(filtersz, kernel_sz2, strides=stride_sz2, padding='same', data_format=data_format)(conv1)
  conv2 = Dropout(dropout)(conv2)
  conv2 = BatchNormalization(axis=axis)(conv2)
  conv2 = Activation('relu')(conv2)
  conv3 = conv(filtersz, kernel_sz2, strides=stride_sz2, padding='same', data_format=data_format)(conv2)
  conv3 = Dropout(dropout)(conv3)
  conv3 = BatchNormalization(axis=axis)(conv3)
  conv3 = Activation('relu')(conv3)
  conv4 = conv(filtersz, kernel_sz2, strides=stride_sz2, padding='same', data_format=data_format)(conv3)
  conv4 = Dropout(dropout)(conv4)
  conv4 = BatchNormalization(axis=axis)(conv4)
  conv4 = Activation('relu')(conv4)
  conv5 = conv(filtersz, kernel_sz2, strides=stride_sz2, padding='same', data_format=data_format)(conv4)
  dense1 = Flatten()(conv5)
  dense1 = Dense(filtersz)(dense1)
  dense1 = BatchNormalization(axis=axis)(dense1)
  dense1 = Activation('relu')(dense1)
  dense2 = Dense(densesz)(dense1)
  dense2 = BatchNormalization(axis=axis)(dense2)
  dense2 = Activation('relu')(dense2)
  dense3 = Dense(nroutputs)(dense2)
  dense3 = BatchNormalization(axis=axis)(dense3)
  dense3 = Activation('softmax')(dense3)
  model = Model(inputs=[inputs], outputs=[dense3])
  return model
        
class dGB_LeNet_Classifier(UserModel):
  uiname = 'dGB LeNet classifier'
  uidescription = 'dGBs LeNet classifier Keras model in UserModel form'
  predtype = DataPredType.Classification
  outtype = OutputType.Pixel
  dimtype = DimType.Any
  
  def _make_model(self, input_shape, nroutputs, learnrate, data_format):
    model = dGBLeNet(input_shape, nroutputs, data_format)
    
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
  
  def _make_model(self, input_shape, nroutputs, learnrate, data_format):
    model = dGBLeNet(input_shape, nroutputs, data_format)
      
    model.compile(optimizer=RMSprop(lr=learnrate), loss=root_mean_squared_error, metrics=[root_mean_squared_error])
    
    return model
    
    
    
    
      
