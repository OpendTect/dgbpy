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

from dgbpy.keras_classes import UserModel
import dgbpy.dgbkeras as dgbkeras

def root_mean_squared_error(y_true, y_pred):
  return kb.sqrt(kb.mean(kb.square(y_pred - y_true)))

class dGB_Unet(UserModel):
  uiname = 'dGB UNet'
  uidescription = 'dGBs Unet img2img Keras model in UserModel form'
  modtype = UserModel.img2imgtypestr
  dims = 'any'
  
  def _make_model(self, input_shape, nroutputs, learnrate):
    ndim = len(input_shape)-1
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
      
    unetnszs=(16,512)
    poolsz1 = 2
    poolsz2 = 2
    poolsz3 = 2
    upscalesz = 2
    filtersz1 = unetnszs[0]
    filtersz2 = filtersz1 * poolsz2
    filtersz3 = filtersz2 * poolsz3
    filtersz4 = unetnszs[1]
    axis = -1

    params = dict(kernel_size=3, activation='relu', padding='same')

    inputs = Input(input_shape)
    conv1 = conv(filtersz1, **params)(inputs)
    conv1 = conv(filtersz1, **params)(conv1)
    
    pool1 = pool(pool_size=poolsz1)(conv1)
    
    conv2 = conv(filtersz2, **params)(pool1)
    conv2 = conv(filtersz2, **params)(conv2)
    pool2 = pool(pool_size=poolsz2)(conv2)
    
    conv3 = conv(filtersz3, **params)(pool2)
    conv3 = conv(filtersz3, **params)(conv3)
    pool3 = pool(pool_size=poolsz3)(conv3)
    
    conv4 = conv(filtersz4, **params)(pool3)
    conv4 = conv(filtersz4, **params)(conv4)
    
    up5 = concatenate([upsamp(size=upscalesz)(conv4), conv3], axis=axis)
    conv5 = conv(filtersz3, **params)(up5)
    conv5 = conv(filtersz3, **params)(conv5)
    
    up6 = concatenate([upsamp(size=poolsz2)(conv5), conv2], axis=axis)
    conv6 = conv(filtersz2, **params)(up6)
    conv6 = conv(filtersz2, **params)(conv6)
    
    up7 = concatenate([upsamp(size=poolsz1)(conv6), conv1], axis=axis)
    conv7 = conv(filtersz1, **params)(up7)
    conv7 = conv(filtersz1, **params)(conv7)
    
    conv8 = conv(nroutputs, 1, activation='sigmoid')(conv7)
    
    model = Model(inputs=[inputs], outputs=[conv8])
    
    model.compile(optimizer = Adam(lr = learnrate), loss = dgbkeras.cross_entropy_balanced, metrics = ['accuracy'])

    return model
  
def dGBLeNet(input_shape, nroutputs):
  ndim = len(input_shape)-1
  conv = None
  if ndim==3:
    conv = Conv3D
  elif ndim==2:
    conv = Conv2D
  elif ndim==1 or ndim==0:
    conv = Conv1D
  kernel_sz1 = 5
  kernel_sz2 = 3
  stride_sz1 = 4
  stride_sz2 = 2
  dropout = 0.2
  filtersz = 50
  densesz = 10
  inputs = Input(input_shape)
  conv1 = conv(filtersz, kernel_sz1, strides=stride_sz1, padding='same')(inputs)
  conv1 = BatchNormalization()(conv1)
  conv1 = Activation('relu')(conv1)
  conv2 = conv(filtersz, kernel_sz2, strides=stride_sz2, padding='same')(conv1)
  conv2 = Dropout(dropout)(conv2)
  conv2 = BatchNormalization()(conv2)
  conv2 = Activation('relu')(conv2)
  conv3 = conv(filtersz, kernel_sz2, strides=stride_sz2, padding='same')(conv2)
  conv3 = Dropout(dropout)(conv3)
  conv3 = BatchNormalization()(conv3)
  conv3 = Activation('relu')(conv3)
  conv4 = conv(filtersz, kernel_sz2, strides=stride_sz2, padding='same')(conv3)
  conv4 = Dropout(dropout)(conv4)
  conv4 = BatchNormalization()(conv4)
  conv4 = Activation('relu')(conv4)
  conv5 = conv(filtersz, kernel_sz2, strides=stride_sz2, padding='same')(conv4)
  dense1 = Flatten()(conv5)
  dense1 = Dense(filtersz)(dense1)
  dense1 = BatchNormalization()(dense1)
  dense1 = Activation('relu')(dense1)
  dense2 = Dense(densesz)(dense1)
  dense2 = BatchNormalization()(dense2)
  dense2 = Activation('relu')(dense2)
  dense3 = Dense(nroutputs)(dense2)
  dense3 = BatchNormalization()(dense3)
  dense3 = Activation('softmax')(dense3)
  model = Model(inputs=[inputs], outputs=[dense3])
  return model
        
class dGB_LeNet_Classifier(UserModel):
  uiname = 'dGB LeNet classifier'
  uidescription = 'dGBs LeNet classifier Keras model in UserModel form'
  modtype = UserModel.classifiertypestr
  dims = 'any'
  
  def _make_model(self, input_shape, nroutputs, learnrate):
    model = dGBLeNet(input_shape, nroutputs)
    
    loss = 'binary_crossentropy'
    if nroutputs>2:
      loss = 'categorical_crossentropy'
    
    model.compile(optimizer=Adam(lr=learnrate), loss=loss, metrics=['accuracy'])
    
    return model

class dGB_LeNet_Regressor(UserModel):
  uiname = 'dGB LeNet regressor'
  uidescription = 'dGBs LeNet regressor Keras model in UserModel form'
  modtype = UserModel.regressortypestr
  dims = 'any'
  
  def _make_model(self, input_shape, nroutputs, learnrate):
    model = dGBLeNet(input_shape, nroutputs)
      
    model.compile(optimizer=RMSprop(lr=learnrate), loss=root_mean_squared_error, metrics=[root_mean_squared_error])
    
    return model
    
    
    
    
      
