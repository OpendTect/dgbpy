import os
import numpy as np

import keras
from keras.callbacks import (EarlyStopping,LearningRateScheduler)
from keras.layers import (Activation,Conv3D,Dense,Dropout,Flatten)
from keras.layers.normalization import BatchNormalization
from keras.models import (Sequential)
from odpy.common import *

lastlayernm = 'pre-softmax_layer'

# Function that takes the epoch as input and returns the desired learning rate
# input_int: the epoch that is currently being entered
def adaptive_lr(input_int):
  # return the learning rate (quite arbitrarily decaying)
  return 0.1**input_int

def getLayer( model, name ):
  for lay in model.layers:
    if lay.get_config()['name'] == name:
      return lay
  return None

def getNrClasses( model ):
  return getLayer(model,lastlayernm).get_config()['units']

def getDefaultModel(setup):
  nrclasses = len(setup['examples'])
  stepout = setup['stepout']
  model = Sequential()
  model.add(Conv3D(50, (5, 5, 5), strides=(4, 4, 4), padding='same', name='conv_layer1', \
             input_shape=(2*stepout[0]+1,2*stepout[1]+1,2*stepout[2]+1,1), \
             data_format="channels_last"))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding='same', name='conv_layer2'))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding='same', name='conv_layer3'))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding='same', name='conv_layer4'))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding='same', name='conv_layer5'))
  model.add(Flatten())
  model.add(Dense(50,name = 'dense_layer1'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dense(10,name = 'attribute_layer'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dense(nrclasses, name=lastlayernm))
  model.add(BatchNormalization())
  model.add(Activation('softmax'))

# initiate the Adam optimizer with a given learning rate (Note that this is adapted later)
  opt = keras.optimizers.adam(lr=0.001)

# Compile the model with the desired loss, optimizer, and metric
  model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
  return model

def train(model,training,params,trainfile=None):
  early_stopping = EarlyStopping(monitor='acc', patience=params['opt_patience'])
  LR_sched = LearningRateScheduler(schedule = adaptive_lr)
  num_bunch = params['num_tot_iterations']
  dec_fact = params['decimation']
  decimate = dec_fact != None
  x_train = {}
  y_train = {}
  if not decimate:
    x_train = training['train']['x']
    y_train = training['train']['y']
  for repeat in range(num_bunch):
    log_msg('Starting iteration',str(repeat+1)+'/'+str(num_bunch))
    log_msg('Starting training data creation:')
    if decimate and trainfile != None:
      import dgbpy.mlio as dgbmlio
      trainbatch = dgbmlio.getTrainingData( trainfile,dec_fact)
      x_train = trainbatch['train']['x']
      y_train = trainbatch['train']['y']
    log_msg('Finished creating',len(x_train),'examples!')

    x_train = np.expand_dims(x_train,axis=4)
    y_train = keras.utils.to_categorical(y_train, getNrClasses(model))
    history = model.fit(x=x_train,y=y_train,callbacks=[early_stopping, LR_sched],shuffle=True, \
                        validation_split=0.2, \
                        batch_size=params['batch_size'], \
                        epochs=params['epochs'])
    keras.utils.print_summary( model, print_fn=log_msg )

  return model

def save( model, fnm ):
  log_msg( 'Saving model.' )
  model.save( fnm )
  log_msg( 'Model saved.' )
