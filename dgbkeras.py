#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Nov 2018
#
# _________________________________________________________________________
# various tools machine learning using Keras platform
#

import os
import re
from datetime import datetime
import numpy as np
import math

from odpy.common import log_msg, get_log_file, redirect_stdout, restore_stdout
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from keras.callbacks import (EarlyStopping,LearningRateScheduler)
withtensorboard = True
if 'KERAS_WITH_TENSORBOARD' in os.environ:
  withtensorboard = not ( os.environ['KERAS_WITH_TENSORBOARD'] == False or \
                          os.environ['KERAS_WITH_TENSORBOARD'] == 'No' )
if withtensorboard:
  from keras.callbacks import TensorBoard

platform = (dgbkeys.kerasplfnm,'Keras (tensorflow)')
mltypes = (\
            ('lenet','LeNet - Malenov'),\
            ('unet','U-Net'),\
            ('squeezenet','SqueezeNet'),\
            ('other','MobilNet V2'),\
          )

cudacores = [ '8', '16', '32', '48', '64', '96', '128', '144', '192', '256', \
              '288',  '384',  '448',  '480',  '512',  '576',  '640',  '768', \
              '896',  '960',  '1024', '1152', '1280', '1344', '1408', '1536', \
              '1664', '1792', '1920', '2048', '2176', '2304', '2432', '2496', \
              '2560', '2688', '2816', '2880', '2944', '3072', '3584', '3840', \
              '4352', '4608', '4992', '5120' ]

def getMLPlatform():
  return platform[0]

def getUIMLPlatform():
  return platform[1]

letnetidx = 0
def isLeNet( mltype ):
  return mltype == mltypes[letnetidx][0] or mltype == mltypes[letnetidx][1]

unetidx = 1
def isUnet( mltype ):
  return mltype == mltypes[unetidx][0] or mltype == mltypes[unetidx][1]

squeezenetidx = 2
def isSqueezeNet( mltype ):
  return mltype == mltypes[squeezenetidx][0] or mltype == mltypes[squeezenetidx][1]

mobilnetv2idx = 3
def isMobilNetV2( mltype ):
  return mltype == mltypes[mobilnetv2idx][0] or mltype == mltypes[mobilnetv2idx][1]

def getUiModelTypes( learntype ):
  ret = ()
  if dgbhdf5.isImg2Img(learntype):
    ret += (mltypes[unetidx],)
  else:
    ret += (mltypes[letnetidx],)
#    ret += (mltypes[squeezenetidx],)
#    ret += (mltypes[mobilnetv2idx],)

  return dgbkeys.getNames( ret )

keras_dict = {
  dgbkeys.decimkeystr: False,
  'nbchunk': 10,
  'epoch': 15,
  'batch': 32,
  'patience': 5,
  'learnrate': 0.01,
  'epochdrop': 5,
  'type': mltypes[letnetidx][0],
}

def getParams( dodec=keras_dict[dgbkeys.decimkeystr], nbchunk=keras_dict['nbchunk'],
               epochs=keras_dict['epoch'],
               batch=keras_dict['batch'], patience=keras_dict['patience'],
               learnrate=keras_dict['learnrate'],
               epochdrop=keras_dict['epochdrop'],nntype=keras_dict['type'] ):
  ret = {
    dgbkeys.decimkeystr: dodec,
    'nbchunk': nbchunk,
    'epoch': epochs,
    'batch': batch,
    'patience': patience,
    'learnrate': learnrate,
    'epochdrop': epochdrop,
    'type': nntype
  }
  if not dodec:
    ret['nbchunk'] = 1
  return ret

# Function that takes the epoch as input and returns the desired learning rate
# input_int: the epoch that is currently being entered
def adaptive_schedule(initial_lrate=keras_dict['learnrate'],
                      epochs_drop=keras_dict['epochdrop']):
  def adaptive_lr(input_int):
    drop = 0.5
    return initial_lrate * math.pow(drop,
                                    math.floor((1+input_int)/epochs_drop))
    # return the learning rate (quite arbitrarily decaying)
  return LearningRateScheduler(adaptive_lr)

def cross_entropy_balanced(y_true, y_pred):
  from keras.models import K
  from keras.optimizers import tf
  _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
  y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
  y_pred   = tf.log(y_pred/ (1 - y_pred))

  y_true = tf.cast(y_true, tf.float32)
  count_neg = tf.reduce_sum(1. - y_true)
  count_pos = tf.reduce_sum(y_true)
  beta = count_neg / (count_neg + count_pos)
  pos_weight = beta / (1 - beta)
  cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)
  cost = tf.reduce_mean(cost * (1 - beta))
  return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)

def _to_tensor(x, dtype):
  from keras.optimizers import tf
  x = tf.convert_to_tensor(x)
  if x.dtype != dtype:
    x = tf.cast(x, dtype)
  return x

def getDataFormat( model ):
  layers = model.layers
  for i in range(len(layers)):
    laycfg = layers[i].get_config()
    if 'data_format' in laycfg:
      return laycfg['data_format']
  return None

def getCubeletStepout( model ):
  data_format = getDataFormat( model )
  if data_format == 'channels_first':
    cubeszs = model.input_shape[2:]
  elif data_format == 'channels_last':
    cubeszs = model.input_shape[1:-1]
  stepout = tuple()
  for cubesz in cubeszs:
    if cubesz%2 == 1 :
      stepout += (int((cubesz-1)/2),)
    else:
      stepout += (int(cubesz/2),)
  return (stepout,cubeszs)

def getLogDir( basedir, args ):
  logdir = basedir
  if not withtensorboard or logdir == None or not os.path.exists(logdir):
    return None

  if dgbkeys.surveydictstr in args:
    jobnm = args[dgbkeys.surveydictstr][0] + '_run'
  else:
    jobnm = 'run'

  nrsavedruns = 0
  with os.scandir(logdir) as it:
    for entry in it:
      if entry.name.startswith(jobnm) and entry.is_dir():
        nrsavedruns += 1
  logdir = os.path.join( logdir, jobnm+str(nrsavedruns+1)+'_'+'m'.join( datetime.now().isoformat().split(':')[:-1] ) )
  return logdir

def get_model_shape( step, nrattribs, attribfirst=True, allowodd=True ):
  ret = ()
  extraidx = 0
  if allowodd:
    extraidx = 1
  if attribfirst:
    ret += (nrattribs,)
  if isinstance( step, int ):
    ret += (2*step+allowodd,)
    if not attribfirst:
      ret += (nrattribs,)
    return ret
  else:
    for i in step:
      if i > 0:
        ret += (2*i+allowodd,)
  if attribfirst:
    if len(ret) == 1:
      ret += (1,)
  else:
    if len(ret) == 0:
      ret += (1,)
  if not attribfirst:
    ret += (nrattribs,)
  return ret

def getModelDims( model_shape, data_format ):
  if data_format == 'channels_first':
    ret = model_shape[1:]
  else:
    ret = model_shape[:-1]
  if len(ret) == 1 and ret[0] == 1:
    return 0
  return len(ret)

def getDefaultModel(setup,type=keras_dict['type'],
                    learnrate=keras_dict['learnrate'],
                    data_format='channels_first'):
  isclassification = setup[dgbhdf5.classdictstr]
  if isclassification:
    nroutputs = len(setup[dgbkeys.classesdictstr])
  else:
    nroutputs = dgbhdf5.getNrOutputs( setup )

  nrattribs = dgbhdf5.get_nr_attribs(setup)
  allowodd = True
  if isUnet(type):
    allowodd = False #TODO: support odd numbers?
  model_shape = get_model_shape( setup[dgbkeys.stepoutdictstr], nrattribs,
                                 attribfirst=data_format=='channels_first', \
                                 allowodd=allowodd )
  if isLeNet( type ):
    return getDefaultLeNet(setup,isclassification,model_shape,nroutputs,
                           learnrate=learnrate,data_format=data_format)
  elif isUnet( type ):
    return getDefaultUnet(setup,isclassification,model_shape,nroutputs,
                          learnrate=learnrate,data_format=data_format)
  else:
    return None

def getDefaultLeNet3D( model, shape, format ):
  from keras.layers import (Activation,Conv3D,Dropout)
  from keras.layers.normalization import BatchNormalization
  model.add(Conv3D(50, (5, 5, 5), strides=(4, 4, 4), padding='same', \
            name='conv_layer1',input_shape=shape,data_format=format))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding='same', name='conv_layer2', data_format=format))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding='same', name='conv_layer3',data_format=format))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding='same', name='conv_layer4',data_format=format))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding='same', name='conv_layer5',data_format=format))
  return model

def getDefaultLeNet2D( model, shape, format ):
  from keras.layers import (Activation,Conv2D,Dropout)
  from keras.layers.normalization import BatchNormalization
  model.add(Conv2D(50, (5, 5), strides=(4, 4), padding='same', \
            name='conv_layer1',input_shape=shape,data_format=format))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(50, (3, 3), strides=(2, 2), padding='same', name='conv_layer2', data_format=format))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(50, (3, 3), strides=(2, 2), padding='same', name='conv_layer3',data_format=format))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(50, (3, 3), strides=(2, 2), padding='same', name='conv_layer4',data_format=format))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(50, (3, 3), strides=(2, 2), padding='same', name='conv_layer5',data_format=format))
  return model

def getDefaultLeNet1D( model, shape, format ):
  from keras.layers import (Activation,Conv1D,Dropout)
  from keras.layers.normalization import BatchNormalization
  model.add(Conv1D(50, 5, strides=4, padding='same', \
            name='conv_layer1',input_shape=shape,data_format=format))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv1D(50, 3, strides=2, padding='same', name='conv_layer2', data_format=format))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv1D(50, 3, strides=2, padding='same', name='conv_layer3',data_format=format))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv1D(50, 3, strides=2, padding='same', name='conv_layer4',data_format=format))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv1D(50, 3, strides=2, padding='same', name='conv_layer5',data_format=format))
  return model

def getDefaultLeNet(setup,isclassification,model_shape,nroutputs,
                    learnrate=keras_dict['learnrate'],
                    data_format='channels_first'):
  
  redirect_stdout()
  import keras
  restore_stdout()
  from keras.layers import (Activation,Dense,Flatten)
  from keras.layers.normalization import BatchNormalization
  from keras.models import Sequential
  from keras.optimizers import Adam

  nrdims = getModelDims( model_shape, data_format )
  model = Sequential()
  if nrdims == 3:
    model = getDefaultLeNet3D( model, model_shape, data_format )
  elif nrdims == 2:
    model = getDefaultLeNet2D( model, model_shape, data_format )
  elif nrdims == 1 or nrdims == 0:
    model = getDefaultLeNet1D( model, model_shape, data_format )
  else:
    return None

  model.add(Flatten())
  model.add(Dense(50,name = 'dense_layer1'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dense(10,name = 'attribute_layer'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dense(nroutputs, name='pre-softmax_layer'))
  model.add(BatchNormalization())
  model.add(Activation('softmax'))

# initiate the model compiler options
  metrics = ['accuracy']
  if isclassification:
    opt = Adam(lr = learnrate)
    if nroutputs > 2:
      loss = 'categorical_crossentropy'
    else:
      loss = 'binary_crossentropy'
  else:
    opt = keras.optimizers.RMSprop(lr=learnrate)
    loss = cross_entropy_balanced
#    set_epsilon( 1 )
#    from keras import backend as K
#    def root_mean_squared_error(y_true, y_pred):
#      return K.sqrt(K.mean(K.square(y_pred - y_true)))
#    loss = root_mean_squared_error

# Compile the model with the desired optimizer, loss, and metric
  model.compile(optimizer=opt,loss=loss,metrics=metrics)
  return model

def getDefaultUnet(setup,isclassification,model_shape,nroutputs,
                    learnrate=keras_dict['learnrate'],
                    data_format='channels_first'):
  redirect_stdout()
  import keras
  restore_stdout()
  from keras.layers import Input
  from keras.models import Model
  from keras.optimizers import Adam

  inputs = Input(model_shape)
  if data_format == 'channels_first':
    axis = 1
  else:
    axis = len(model_shape)
  if isclassification:
    nroutputs = 1

  #TODO: support of odd numbers??
  nrdims = getModelDims( model_shape, data_format )
  if nrdims == 3:
    lastconv = getDefaultUnet3D( inputs, data_format, axis, nroutputs )
  elif nrdims == 2:
    lastconv = getDefaultUnet2D( inputs, data_format, axis, nroutputs )
  elif nrdims == 1:
    lastconv = getDefaultUnet1D( inputs, data_format, axis, nroutputs )
  else:
    return None
  
  model = Model(inputs=[inputs], outputs=[lastconv])

  metrics = ['accuracy']
  opt = Adam(lr = learnrate)
#  if isclassification:
#    if nroutputs > 2:
#      loss = 'categorical_crossentropy'
#    else:
#      loss = 'binary_crossentropy'
#      loss = 'sparse_categorical_crossentropy'
#      metrics = ['sparse_categorical_accuracy']
#  else:
#    loss = 'cross_entropy_balanced'
  loss = cross_entropy_balanced

  model.compile(optimizer=opt,loss=loss,metrics=metrics)
  return model

def getDefaultUnet3D( inputs, format, axis, nroutputs ):
  from keras.layers import (concatenate,Conv3D,MaxPooling3D,UpSampling3D)

  conv1 = Conv3D(2, (3,3,3), activation='relu', padding='same', data_format=format)(inputs)
  conv1 = Conv3D(2, (3,3,3), activation='relu', padding='same', data_format=format)(conv1)
  pool1 = MaxPooling3D(pool_size=(2,2,2),data_format=format)(conv1)

  conv2 = Conv3D(4, (3,3,3), activation='relu', padding='same', data_format=format)(pool1)
  conv2 = Conv3D(4, (3,3,3), activation='relu', padding='same', data_format=format)(conv2)
  pool2 = MaxPooling3D(pool_size=(2,2,2),data_format=format)(conv2)

  conv3 = Conv3D(8, (3,3,3), activation='relu', padding='same', data_format=format)(pool2)
  conv3 = Conv3D(8, (3,3,3), activation='relu', padding='same', data_format=format)(conv3)
  pool3 = MaxPooling3D(pool_size=(2,2,2),data_format=format)(conv3)

  conv4 = Conv3D(64, (3,3,3), activation='relu', padding='same', data_format=format)(pool3)
  conv4 = Conv3D(64, (3,3,3), activation='relu', padding='same', data_format=format)(conv4)

  up5 = concatenate([UpSampling3D(size=(2,2,2),data_format=format)(conv4), conv3], axis=axis)
  conv5 = Conv3D(8, (3,3,3), activation='relu', padding='same', data_format=format)(up5)
  conv5 = Conv3D(8, (3,3,3), activation='relu', padding='same', data_format=format)(conv5)

  up6 = concatenate([UpSampling3D(size=(2,2,2),data_format=format)(conv5), conv2], axis=axis)
  conv6 = Conv3D(4, (3,3,3), activation='relu', padding='same', data_format=format)(up6)
  conv6 = Conv3D(4, (3,3,3), activation='relu', padding='same', data_format=format)(conv6)

  up7 = concatenate([UpSampling3D(size=(2,2,2),data_format=format)(conv6), conv1], axis=axis)
  conv7 = Conv3D(2, (3,3,3), activation='relu', padding='same', data_format=format)(up7)
  conv7 = Conv3D(2, (3,3,3), activation='relu', padding='same', data_format=format)(conv7)

  conv8 = Conv3D(nroutputs, (1,1,1), activation='sigmoid', data_format=format)(conv7)
  return conv8


def getDefaultUnet2D( inputs, format, axis, nroutputs ):
  from keras.layers import (concatenate,Conv2D,MaxPooling2D,UpSampling2D)

  conv1 = Conv2D(2, (3,3), activation='relu', padding='same', data_format=format)(inputs)
  conv1 = Conv2D(2, (3,3), activation='relu', padding='same', data_format=format)(conv1)
  pool1 = MaxPooling2D(pool_size=(2,2),data_format=format)(conv1)

  conv2 = Conv2D(4, (3,3), activation='relu', padding='same', data_format=format)(pool1)
  conv2 = Conv2D(4, (3,3), activation='relu', padding='same', data_format=format)(conv2)
  pool2 = MaxPooling2D(pool_size=(2,2),data_format=format)(conv2)

  conv3 = Conv2D(8, (3,3), activation='relu', padding='same', data_format=format)(pool2)
  conv3 = Conv2D(8, (3,3), activation='relu', padding='same', data_format=format)(conv3)
  pool3 = MaxPooling2D(pool_size=(2,2),data_format=format)(conv3)

  conv4 = Conv2D(64, (3,3), activation='relu', padding='same', data_format=format)(pool3)
  conv4 = Conv2D(64, (3,3), activation='relu', padding='same', data_format=format)(conv4)

  up5 = concatenate([UpSampling2D(size=(2,2),data_format=format)(conv4), conv3], axis=axis)
  conv5 = Conv2D(8, (3,3), activation='relu', padding='same', data_format=format)(up5)
  conv5 = Conv2D(8, (3,3), activation='relu', padding='same', data_format=format)(conv5)

  up6 = concatenate([UpSampling2D(size=(2,2),data_format=format)(conv5), conv2], axis=axis)
  conv6 = Conv2D(4, (3,3), activation='relu', padding='same', data_format=format)(up6)
  conv6 = Conv2D(4, (3,3), activation='relu', padding='same', data_format=format)(conv6)

  up7 = concatenate([UpSampling2D(size=(2,2),data_format=format)(conv6), conv1], axis=axis)
  conv7 = Conv2D(2, (3,3), activation='relu', padding='same', data_format=format)(up7)
  conv7 = Conv2D(2, (3,3), activation='relu', padding='same', data_format=format)(conv7)

  conv8 = Conv2D(nroutputs, (1,1), activation='sigmoid', data_format=format)(conv7)
  return conv8

def getDefaultUnet1D( inputs, format, axis, nroutputs ):
  from keras.layers import (concatenate,Conv1D,MaxPooling1D,UpSampling1D)

  conv1 = Conv1D(2, 3, activation='relu', padding='same', data_format=format)(inputs)
  conv1 = Conv1D(2, 3, activation='relu', padding='same', data_format=format)(conv1)
  pool1 = MaxPooling1D(pool_size=2,data_format=format)(conv1)

  conv2 = Conv1D(4, 3, activation='relu', padding='same', data_format=format)(pool1)
  conv2 = Conv1D(4, 3, activation='relu', padding='same', data_format=format)(conv2)
  pool2 = MaxPooling1D(pool_size=2,data_format=format)(conv2)

  conv3 = Conv1D(8, 3, activation='relu', padding='same', data_format=format)(pool2)
  conv3 = Conv1D(8, 3, activation='relu', padding='same', data_format=format)(conv3)
  pool3 = MaxPooling1D(pool_size=2,data_format=format)(conv3)

  conv4 = Conv1D(64, 3, activation='relu', padding='same', data_format=format)(pool3)
  conv4 = Conv1D(64, 3, activation='relu', padding='same', data_format=format)(conv4)

  up5 = concatenate([UpSampling1D(2)(conv4), conv3], axis=axis)
  conv5 = Conv1D(8, 3, activation='relu', padding='same', data_format=format)(up5)
  conv5 = Conv1D(8, 3, activation='relu', padding='same', data_format=format)(conv5)

  up6 = concatenate([UpSampling1D(2)(conv5), conv2], axis=axis)
  conv6 = Conv1D(4, 3, activation='relu', padding='same', data_format=format)(up6)
  conv6 = Conv1D(4, 3, activation='relu', padding='same', data_format=format)(conv6)

  up7 = concatenate([UpSampling1D(2)(conv6), conv1], axis=axis)
  conv7 = Conv1D(2, 3, activation='relu', padding='same', data_format=format)(up7)
  conv7 = Conv1D(2, 3, activation='relu', padding='same', data_format=format)(conv7)

  conv8 = Conv1D(nroutputs, 1, activation='sigmoid',data_format=format)(conv7)
  return conv8

def train(model,training,params=keras_dict,trainfile=None,logdir=None):
  redirect_stdout()
  import keras
  restore_stdout()
  infos = training[dgbkeys.infodictstr]
  classification = infos[dgbkeys.classdictstr]
  if classification:
    monitor = 'acc'
  else:
    monitor = 'loss'
  early_stopping = EarlyStopping(monitor=monitor, patience=params['patience'])
  LR_sched = adaptive_schedule(params['learnrate'],params['epochdrop'])
  callbacks = [early_stopping,LR_sched]
  batchsize = params['batch']
  if logdir != None:
    tensor_board = TensorBoard(log_dir=logdir, histogram_freq=1, \
                               batch_size=batchsize,\
                         write_graph=True, write_grads=False, write_images=True)
    callbacks.append( tensor_board )
  nbchunks = len( infos[dgbkeys.trainseldicstr] )
  x_train = {}
  y_train = {}
  doshuffle = False
  decimate = nbchunks > 1
  if not decimate:
    if not dgbkeys.xtraindictstr in training:
      log_msg('No data to train the model')
      return model
    x_train = training[dgbkeys.xtraindictstr]
    y_train = training[dgbkeys.ytraindictstr]
    x_validate = training[dgbkeys.xvaliddictstr]
    y_validate = training[dgbkeys.yvaliddictstr]
    doshuffle = True
  for ichunk in range(nbchunks):
    log_msg('Starting iteration',str(ichunk+1)+'/'+str(nbchunks))
    log_msg('Starting training data creation:')
    if decimate and trainfile != None:
      import dgbpy.mlapply as dgbmlapply
      trainbatch = dgbmlapply.getScaledTrainingDataByInfo( infos, flatten=False,
                                                 scale=True, ichunk=ichunk )
      if not dgbkeys.xtraindictstr in trainbatch:
        continue
      x_train = trainbatch[dgbkeys.xtraindictstr]
      y_train = trainbatch[dgbkeys.ytraindictstr]
      x_validate = trainbatch[dgbkeys.xvaliddictstr]
      y_validate = trainbatch[dgbkeys.yvaliddictstr]
    log_msg('Finished creating',len(x_train),'examples!')
    log_msg('Validation done on', len(x_validate), 'examples.' )
    x_train = adaptToModel( model, x_train )
    x_validate = adaptToModel( model, x_validate )
    if len(y_train.shape) > 2:
      y_train = adaptToModel( model, y_train )
    if len(y_validate.shape) > 2:
      y_validate = adaptToModel( model, y_validate )
    if classification and not dgbhdf5.isImg2Img(infos):
      nrclasses = dgbhdf5.getNrClasses( infos )
      y_train = keras.utils.to_categorical(y_train,nrclasses)
      y_validate = keras.utils.to_categorical(y_validate,nrclasses)
    redirect_stdout()
    hist = model.fit(x=x_train,y=y_train,callbacks=callbacks,\
                  shuffle=doshuffle, validation_data=(x_validate,y_validate),\
                  batch_size=batchsize, \
                  epochs=params['epoch'])
    #log_msg( hist.history )
    restore_stdout()

  keras.utils.print_summary( model, print_fn=log_msg )
  return model

def save( model, outfnm ):
  model.save( outfnm )

def load( modelfnm, fortrain ):
  redirect_stdout()
  from keras.models import load_model
  ret = load_model( modelfnm, compile=fortrain )
  restore_stdout()
  return ret

def transfer( model ):
  from keras.layers import (Conv1D,Conv2D,Conv3D,Dense)
  layers = model.layers
  for layer in layers:
    layer.trainable = False

  ilaystart = 0
  for ilay in range(len(layers)):
    layers[ilay].trainable = True
    laytype = type( layers[ilay] )
    if laytype == Conv3D or laytype == Conv2D or laytype == Conv1D:
       break

  for ilay in range(len(layers)-1,0,-1):
    layers[ilay].trainable = True
    laytype = type( layers[ilay] )
    if laytype == Conv3D or laytype == Conv2D or laytype == Conv1D or laytype == Dense:
      break

  return model

def apply( model, samples, isclassification, withpred, withprobs, withconfidence, doprobabilities, scaler=None, batch_size=keras_dict['batch'] ):
  redirect_stdout()
  import keras
  restore_stdout()
  ret = {}
  res = None
  inp_shape = samples.shape
  data_format = 'channels_first'
  samples = adaptToModel( model, samples, sample_data_format=data_format )
    
  if withpred:
    if isclassification:
      try:
        res = model.predict_classes( samples, batch_size=batch_size )
      except AttributeError:
        res = model.predict( samples, batch_size=batch_size )
    else:
      res = model.predict( samples, batch_size=batch_size )
    res = adaptFromModel(model,isclassification,res,inp_shape,ret_data_format=data_format)
    ret.update({dgbkeys.preddictstr: res})
 
  if isclassification and (doprobabilities or withconfidence):
    allprobs = model.predict( samples, batch_size=batch_size )
    allprobs = adaptFromModel(model,False,allprobs,inp_shape,ret_data_format=data_format)
    if doprobabilities:
      res = np.copy(allprobs[:,withprobs],allprobs.dtype)
      ret.update({dgbkeys.probadictstr: res})
    if withconfidence:
      N = 2
      indices = np.argpartition(allprobs,-N,axis=1)[:,-N:]
      x = len(allprobs)
      sortedprobs = allprobs[np.repeat(np.arange(x),N),indices.ravel()].reshape(x,N)
      res = np.diff(sortedprobs,axis=1)
      ret.update({dgbkeys.confdictstr: res})

  return ret


def adaptToModel( model, samples, sample_data_format='channels_first' ):
  nrdims = len( model.input_shape ) - 2
  nrsamples = samples.shape[0]
  samples_nrdims = len(samples.shape)
  model_data_format = getDataFormat( model )
  (modelstepout,modelcubeszs) = getCubeletStepout( model )
  if sample_data_format == 'channels_first':
    nrattribs = samples.shape[1]
    cube_shape = samples.shape[2:]
  else:
    nrattribs = samples.shape[-1]
    cube_shape = samples.shape[1:-1]
  shapelims = ()
  idx = 0
  shrinked = False
  for i in cube_shape:
    if i == 1:
      dimsz = 1
    else:
      dimsz = min(i,modelcubeszs[idx])
      if dimsz < i:
        shrinked = True
      idx += 1
    shapelims += (dimsz,)
  cube_shape = np.squeeze( np.empty( shapelims, dtype='uint8' ) ).shape
  datadims = len(cube_shape)
  out_shape = (nrsamples,)
  if model_data_format == 'channels_first':
    out_shape += (nrattribs,)
  for i in cube_shape:
    out_shape += (i,)
  if len(cube_shape) < 1:
    out_shape += (1,)
  if model_data_format == 'channels_last':
    out_shape += (nrattribs,)
  switchedattribs = model_data_format != sample_data_format
  if switchedattribs or nrdims != datadims or shrinked or len(cube_shape) < len(shapelims):
    ret = np.empty( out_shape, dtype=samples.dtype )
    if model_data_format == 'channels_last':
      if nrdims == 3:
        if switchedattribs:
          for iattr in range(nrattribs):
            ret[:,:,:,:,iattr] = np.reshape( samples[:,iattr,:shapelims[0],:shapelims[1],:shapelims[2]], ret[:,:,:,:,iattr].shape )
        else:
          for iattr in range(nrattribs):
            ret[:,:,:,:,iattr] = np.reshape( samples[:,:shapelims[0],:shapelims[1],:shapelims[2],iattr], ret[:,:,:,:,iattr].shape )
      elif nrdims == 2:
        if switchedattribs:
          if samples_nrdims == 5:
            for iattr in range(nrattribs):
              ret[:,:,:,iattr] = np.reshape( np.squeeze( samples[:,iattr,:shapelims[0],:shapelims[1],:shapelims[2]] ), ret[:,:,:,iattr].shape )
          elif samples_nrdims == 4:
            for iattr in range(nrattribs):
              ret[:,:,:,iattr] = np.reshape( np.squeeze( samples[:,iattr,:shapelims[0],:shapelims[1]] ), ret[:,:,:,iattr].shape )
          elif samples_nrdims == 3:
            for iattr in range(nrattribs):
              ret[:,:,:,iattr] = np.reshape( np.squeeze( samples[:,iattr,:shapelims[0]] ), ret[:,:,:,iattr].shape )
        else:
          if samples_nrdims == 5:
            for iattr in range(nrattribs):
              ret[:,:,:,iattr] = np.reshape( np.squeeze( samples[:,:shapelims[0],:shapelims[1],:shapelims[2],iattr] ), ret[:,:,:,iattr].shape )
          elif samples_nrdims == 4:
            for iattr in range(nrattribs):
              ret[:,:,:,iattr] = np.reshape( np.squeeze( samples[:,:shapelims[0],:shapelims[1],iattr] ), ret[:,:,:,iattr].shape )
          elif samples_nrdims == 3:
            for iattr in range(nrattribs):
              ret[:,:,:,iattr] = np.reshape( np.squeeze( samples[:,:shapelims[0],iattr] ), ret[:,:,:,iattr].shape )
      elif nrdims == 1:
        if switchedattribs:
          if samples_nrdims == 5:
            for iattr in range(nrattribs):
              ret[:,:,iattr] = np.reshape( np.squeeze( samples[:,iattr,:shapelims[0],:shapelims[1],:shapelims[2]] ), ret[:,:,iattr].shape )
          elif samples_nrdims == 4:
            for iattr in range(nrattribs):
              ret[:,:,iattr] = np.reshape( np.squeeze( samples[:,iattr,:shapelims[0],:shapelims[1]] ), ret[:,:,iattr].shape )
          elif samples_nrdims == 3:
            for iattr in range(nrattribs):
              ret[:,:,iattr] = np.reshape( samples[:,iattr,:shapelims[0]], ret[:,:,iattr].shape )
        else:
          if samples_nrdims == 5:
            for iattr in range(nrattribs):
              ret[:,:,iattr] = np.reshape( np.squeeze( samples[:,:shapelims[0],:shapelims[1],:shapelims[2],iattr] ), ret[:,:,iattr].shape )
          elif samples_nrdims == 4:
            for iattr in range(nrattribs):
              ret[:,:,iattr] = np.reshape( np.squeeze( samples[:,:shapelims[0],:shapelims[1],iattr] ), ret[:,:,iattr].shape )
          elif samples_nrdims == 3:
            for iattr in range(nrattribs):
              ret[:,:,iattr] = np.reshape( samples[:,:shapelims[0],iattr], ret[:,:,iattr].shape )
      else:
        return samples
      return ret
    else:
      if nrdims == 3:
        if switchedattribs:
          for iattr in range(nrattribs):
            ret[:,iattr] = np.reshape( samples[:,:shapelims[0],:shapelims[1],:shapelims[2],iattr], ret[:,iattr].shape )
        else:
          for iattr in range(nrattribs):
            ret[:,iattr] = np.reshape( samples[:,iattr,:shapelims[0],:shapelims[1],:shapelims[2]], ret[:,iattr].shape )
      elif nrdims == 2:
        if switchedattribs:
          if samples_nrdims == 5:
            for iattr in range(nrattribs):
              ret[:,iattr] = np.reshape( np.squeeze( samples[:,:shapelims[0],:shapelims[1],:shapelims[2],iattr] ), ret[:,iattr].shape )
          elif samples_nrdims == 4:
            for iattr in range(nrattribs):
              ret[:,iattr] = np.reshape( np.squeeze( samples[:,:shapelims[0],:shapelims[1],iattr] ), ret[:,iattr].shape )
          elif samples_nrdims == 3:
            for iattr in range(nrattribs):
              ret[:,iattr] = np.reshape( np.squeeze( samples[:,:shapelims[0],iattr] ), ret[:,iattr].shape )
        else:
          if samples_nrdims == 5:
            for iattr in range(nrattribs):
              ret[:,iattr] = np.reshape( np.squeeze( samples[:,iattr,:shapelims[0],:shapelims[1],:shapelims[2]] ), ret[:,iattr].shape )
          elif samples_nrdims == 4:
            for iattr in range(nrattribs):
              ret[:,iattr] = np.reshape( np.squeeze( samples[:,iattr,:shapelims[0],:shapelims[1]] ), ret[:,iattr].shape )
          elif samples_nrdims == 3:
            for iattr in range(nrattribs):
              ret[:,iattr] = np.reshape( np.squeeze( samples[:,iattr,:shapelims[0]] ), ret[:,iattr].shape )
      elif nrdims == 1:
        if switchedattribs:
          if samples_nrdims == 5:
            for iattr in range(nrattribs):
              ret[:,iattr] = np.reshape( np.squeeze( samples[:,:shapelims[0],:shapelims[1],:shapelims[2],iattr] ), ret[:,iattr].shape )
          elif samples_nrdims == 4:
            for iattr in range(nrattribs):
              ret[:,iattr] = np.reshape( np.squeeze( samples[:,:shapelims[0],:shapelims[1],iattr] ), ret[:,iattr].shape )
          elif samples_nrdims == 3:
            for iattr in range(nrattribs):
              ret[:,iattr] = np.reshape( samples[:,:shapelims[0],iattr], ret[:,iattr].shape )
        else:
          if samples_nrdims == 5:
            for iattr in range(nrattribs):
              ret[:,iattr] = np.reshape( np.squeeze( samples[:,iattr,:shapelims[0],:shapelims[1],:shapelims[2]] ), ret[:,iattr].shape )
          elif samples_nrdims == 4:
            for iattr in range(nrattribs):
              ret[:,iattr] = np.reshape( np.squeeze( samples[:,iattr,:shapelims[0],:shapelims[1]] ), ret[:,iattr].shape )
          elif samples_nrdims == 3:
            for iattr in range(nrattribs):
              ret[:,iattr] = np.reshape( samples[:,iattr,:shapelims[0]], ret[:,iattr].shape )
      else:
        return samples
      return ret
  return samples

def adaptFromModel( model, isclassification, samples, inp_shape, ret_data_format ):
  nrdims = len( model.output_shape )
  if nrdims == 2:
    if isclassification:
      return samples
    else:
      return samples.transpose()

  nrpts = inp_shape[0]
  cube_shape = (nrpts,)
  model_data_format = getDataFormat( model )
  switchedattribs = model_data_format != ret_data_format
  shapelims = ()
  if model_data_format == 'channels_first':
    nrattribs = model.output_shape[1]
    for i in range(2,nrdims):
      shapelims += (model.output_shape[i],)
  else:
    nrattribs = model.output_shape[-1]
    for i in range(1,nrdims-1):
      shapelims += (model.output_shape[i],)

  data_dims = len(inp_shape)
  if ret_data_format == 'channels_first':
    cube_shape += (nrattribs,)
  if ret_data_format == 'channels_first':
    for i in range(2,data_dims):
      cube_shape += (inp_shape[i],)
  else:
    for i in range(1,data_dims-1):
      cube_shape += (inp_shape[i],)
  if ret_data_format == 'channels_last':
    cube_shape += (nrattribs,)

  res = np.zeros( cube_shape, samples.dtype )
  if model_data_format == 'channels_last':
    if nrdims == 5:
      if switchedattribs:
        for iattr in range(nrattribs):
          res[:,iattr,:shapelims[0],:shapelims[1],:shapelims[2]] = samples[:,:,:,:,iattr]
      else:
        res[:,:shapelims[0],:shapelims[1],:shapelims[2]] = samples
    if nrdims == 4:
      if data_dims == 5:
        if switchedattribs:
          for iattr in range(nrattribs):
            res[:,iattr,:,:shapelims[0],:shapelims[1]] = samples[:,:,:,iattr]
        else:
          res[:,:,:shapelims[0],:shapelims[1]] = samples
      elif data_dims == 4:
        if switchedattribs:
          for iattr in range(nrattribs):
            res[:,iattr,:shapelims[0],:shapelims[1]] = samples[:,:,:,iattr]
        else:
          res[:,:shapelims[0],:shapelims[1]] = samples
    if nrdims == 3:
      if data_dims == 5:
        if switchedattribs:
          for iattr in range(nrattribs):
            res[:,iattr,:,:,:shapelims[0]] = samples[:,:,iattr]
        else:
          res[:,:,:,:shapelims[0]] = samples
      elif data_dims == 4:
        if switchedattribs:
          for iattr in range(nrattribs):
            res[:,iattr,:,:shapelims[0]] = samples[:,:,iattr]
        else:
          res[:,:,:shapelims[0]] = samples
      elif data_dims == 3:
        if switchedattribs:
          for iattr in range(nrattribs):
            res[:,iattr,:shapelims[0]] = samples[:,:,iattr]
        else:
          res[:,:shapelims[0]] = samples
  else:
    if nrdims == 5:
      if switchedattribs:
        for iattr in range(nrattribs):
          res[:,:shapelims[0],:shapelims[1],:shapelims[2],iattr] = samples[:,iattr]
      else:
        res[:,:,:shapelims[0],:shapelims[1],:shapelims[2]] = samples
    if nrdims == 4:
      if data_dims == 5:
        if switchedattribs:
          for iattr in range(nrattribs):
            res[:,:,:shapelims[0],:shapelims[1],iattr] = samples[:,iattr]
        else:
          res[:,:,:,:shapelims[0],:shapelims[1]] = samples
      elif data_dims == 4:
        if switchedattribs:
          for iattr in range(nrattribs):
            res[:,:shapelims[0],:shapelims[1],iattr] = samples[:,iattr]
        else:
          res[:,:,:shapelims[0],:shapelims[1]] = samples
    if nrdims == 3:
      if data_dims == 5:
        if switchedattribs:
          for iattr in range(nrattribs):
            res[:,:,:,:shapelims[0],iattr] = samples[:,iattr]
        else:
          res[:,:,:,:,:shapelims[0]] = samples
      elif data_dims == 4:
        if switchedattribs:
          for iattr in range(nrattribs):
            res[:,:,:shapelims[0],iattr] = samples[:,iattr]
        else:
          res[:,:,:,:shapelims[0]] = samples
      elif data_dims == 3:
        if switchedattribs:
          for iattr in range(nrattribs):
            res[:,:shapelims[0],iattr] = samples[:,iattr]
        else:
          res[:,:,:shapelims[0]] = samples

  return res

def plot( model, outfnm, showshapes=True, withlaynames=False, vertical=True ):
  try:
    import pydot
  except ImportError:
    log_msg( 'Cannot plot the model without pydot module' )
    return
  rankdir = 'TB'
  if not vertical:
    rankdir = 'LR'
  from keras.utils import plot_model
  plot_model( model, to_file=outfnm, show_shapes=showshapes,
              show_layer_names=withlaynames, rankdir=rankdir )

def compute_capability_from_device_desc(device_desc):
  match = re.search(r"compute capability: (\d+)\.(\d+)", \
                      device_desc.physical_device_desc)
  if not match:
    return 0, 0
  return int(match.group(1)), int(match.group(2))

def getDevicesInfo( gpusonly=True ):
  from tensorflow.python.client import device_lib
  local_device_protos = device_lib.list_local_devices()
  if gpusonly:
    ret = [x for x in local_device_protos if x.device_type == 'GPU']
  else:
    ret = [x for x in local_device_protos]
  return ret

def is_gpu_ready():
  import tensorflow as tf
  devs = getDevicesInfo()
  if len(devs) < 1:
    return False
  cc = compute_capability_from_device_desc( devs[0] )
  return tf.test.is_gpu_available(True,cc)
