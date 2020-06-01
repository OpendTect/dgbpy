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
import json
from datetime import datetime
import numpy as np
import math

from odpy.common import log_msg, get_log_file, redirect_stdout, restore_stdout
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
import dgbpy.keras_classes as kc

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
withtensorboard = True
if 'KERAS_WITH_TENSORBOARD' in os.environ:
  withtensorboard = not ( os.environ['KERAS_WITH_TENSORBOARD'] == False or \
                          os.environ['KERAS_WITH_TENSORBOARD'] == 'No' )
withaugmentation = False
if 'KERAS_WITH_AUGMENTATION' in os.environ:
  withaugmentation = not ( os.environ['KERAS_WITH_AUGMENTATION'] == False or \
                           os.environ['KERAS_WITH_AUGMENTATION'] == 'No' )


platform = (dgbkeys.kerasplfnm,'Keras (tensorflow)')
mltypes = (\
            ('lenet','LeNet - Malenov'),\
            ('unet','U-Net'),\
            ('squeezenet','SqueezeNet'),\
            ('other','MobilNet V2'),\
          )

cudacores = [ '1', '2', '4', '8', '16', '32', '48', '64', '96', '128', '144', '192', '256', \
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
unet_smallsz = (2,64)
unet_mediumsz = (16,512)
unet_largesz = (32,512)

squeezenetidx = 2
def isSqueezeNet( mltype ):
  return mltype == mltypes[squeezenetidx][0] or mltype == mltypes[squeezenetidx][1]

mobilnetv2idx = 3
def isMobilNetV2( mltype ):
  return mltype == mltypes[mobilnetv2idx][0] or mltype == mltypes[mobilnetv2idx][1]

def getUiModelTypes( learntype, ndim ):
  ret = ()
  if dgbhdf5.isImg2Img(learntype):
    ret += (mltypes[unetidx],)
    for model in kc.UserModel.getNamesByType(model_type='img2img', dims=str(ndim)):
      ret += ((model,),)
  else:
    ret += (mltypes[letnetidx],)
    for model in kc.UserModel.getNamesByType(model_type='other', dims=str(ndim)):
      ret += ((model,),)
#    ret += (mltypes[squeezenetidx],)
#    ret += (mltypes[mobilnetv2idx],)

  return dgbkeys.getNames( ret )

prefercpustr = 'prefercpu'
defbatchstr = 'defaultbatchsz'

keras_dict = {
  dgbkeys.decimkeystr: False,
  'nbchunk': 10,
  'epoch': 15,
  'batch': 32,
  'patience': 5,
  'learnrate': 1e-4,
  'epochdrop': 5,
  'unetnszs': unet_mediumsz,
  'type': mltypes[letnetidx][0],
  'prefercpu': None
}

def can_use_gpu():
  from tensorflow import config as tfconfig
  return len(tfconfig.list_physical_devices('GPU')) > 0

def get_cpu_preference():
  from tensorflow import config as tfconfig
  return len(tfconfig.list_physical_devices('GPU')) < 1

def get_keras_infos():
  ret = {
    'haskerasgpu': can_use_gpu(),
    prefercpustr: get_cpu_preference(),
    'batchsizes': cudacores,
    defbatchstr: keras_dict['batch']
   }
  return json.dumps( ret )

def set_compute_device( prefercpu=get_cpu_preference() ):
  if not prefercpu:
      return
  from tensorflow import config as tfconfig
  cpudevs = tfconfig.list_physical_devices('CPU')
  tfconfig.set_visible_devices( cpudevs )

def getParams( dodec=keras_dict[dgbkeys.decimkeystr], nbchunk=keras_dict['nbchunk'],
               epochs=keras_dict['epoch'],
               batch=keras_dict['batch'], patience=keras_dict['patience'],
               learnrate=keras_dict['learnrate'],epochdrop=keras_dict['epochdrop'],
               unetnszs=keras_dict['unetnszs'],nntype=keras_dict['type'],
               prefercpu=keras_dict['prefercpu']):
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
  if isUnet(nntype):
    ret.update({'unetnszs': unetnszs})
  if prefercpu == None:
    prefercpu = get_cpu_preference()
  ret.update({'prefercpu': prefercpu})
  if not dodec:
    ret['nbchunk'] = 1
  return ret

# Function that takes the epoch as input and returns the desired learning rate
# input_int: the epoch that is currently being entered
def adaptive_schedule(initial_lrate=keras_dict['learnrate'],
                      epochs_drop=keras_dict['epochdrop']):
  from keras.callbacks import (EarlyStopping,LearningRateScheduler)
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
  y_pred   = tf.math.log(y_pred/ (1 - y_pred))

  y_true = tf.cast(y_true, tf.float32)
  count_neg = tf.reduce_sum(input_tensor=1. - y_true)
  count_pos = tf.reduce_sum(input_tensor=y_true)
  beta = count_neg / (count_neg + count_pos)
  pos_weight = beta / (1 - beta)
  cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
  cost = tf.reduce_mean(input_tensor=cost * (1 - beta))
  return tf.compat.v1.where(tf.equal(count_pos, 0.0), 0.0, cost)

def _to_tensor(x, dtype):
  from keras.optimizers import tf
  x = tf.convert_to_tensor(value=x)
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

def getCubeletShape( model ):
  data_format = getDataFormat( model )
  if data_format == 'channels_first':
    cubeszs = model.input_shape[2:]
  elif data_format == 'channels_last':
    cubeszs = model.input_shape[1:-1]
  return cubeszs

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

def get_model_shape( shape, nrattribs, attribfirst=True ):
  ret = ()
  if attribfirst:
    ret += (nrattribs,)
  if isinstance( shape, int ):
    ret += (shape,)
    if not attribfirst:
      ret += (nrattribs,)
    return ret
  else:
    for i in shape:
      if i > 1:
        ret += (i,)
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
                    unetnszs=keras_dict['unetnszs'],
                    learnrate=keras_dict['learnrate'],
                    data_format='channels_first'):
  isclassification = setup[dgbhdf5.classdictstr]
  if isclassification:
    nroutputs = len(setup[dgbkeys.classesdictstr])
  else:
    nroutputs = dgbhdf5.getNrOutputs( setup )

  nrattribs = dgbhdf5.getNrAttribs(setup)
  model_shape = get_model_shape( setup[dgbkeys.inpshapedictstr], nrattribs,
                                 attribfirst=data_format=='channels_first' )
  if isLeNet( type ):
    return getDefaultLeNet(isclassification,model_shape,nroutputs,
                           learnrate=learnrate,data_format=data_format)
  elif isUnet( type ):
    return getDefaultUnet(isclassification,model_shape,nroutputs,
                          unetnszs=unetnszs,
                          learnrate=learnrate,data_format=data_format)
  elif kc.UserModel.findName(type):
    return kc.UserModel.findName(type).model(model_shape, nroutputs,
                                             learnrate,
                                             data_format=data_format)
  else:
    return None


def getDefaultLeNet(isclassification,model_shape,nroutputs,
                    learnrate=keras_dict['learnrate'],
                    data_format='channels_first'):
  
  redirect_stdout()
  import keras
  restore_stdout()
  from tensorflow.keras.layers import (Activation,BatchNormalization,Dense,Flatten)
  from tensorflow.keras.layers import (Conv1D,Conv2D,Conv3D)
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.optimizers import Adam, RMSprop

  input_shape = model_shape
  if need_channels_last() and data_format == 'channels_first':
#Tensorflow bug; cannot use channel_first on CPU: crash or no accuracy
    data_format = 'channels_last'
    dims = model_shape[1:]
    input_shape = ( *dims, model_shape[0] )

  filtersz = 50
  densesz = 10

  nrdims = getModelDims( input_shape, data_format )

  layers = list()
  if nrdims == 3:
    layers = getDefaultLeNetND( layers, input_shape, filtersz, data_format, Conv3D )
  elif nrdims == 2:
    layers = getDefaultLeNetND( layers, input_shape, filtersz, data_format, Conv2D )
  elif nrdims == 1 or nrdims == 0:
    layers = getDefaultLeNetND( layers, input_shape, filtersz, data_format, Conv1D )
  else:
    return None

  layers.extend([
    Flatten(),
    Dense(filtersz,name = 'dense_layer1'),
    BatchNormalization(),
    Activation('relu'),
    Dense(densesz,name = 'attribute_layer'),
    BatchNormalization(),
    Activation('relu'),
    Dense(nroutputs, name='pre-softmax_layer'),
    BatchNormalization(),
    Activation('softmax')
  ])
  
  redirect_stdout()
  model = Sequential( layers )
  restore_stdout()

# initiate the model compiler options
  if isclassification:
    opt = Adam(lr = learnrate)
    if nroutputs > 2:
      loss = 'categorical_crossentropy'
    else:
      loss = 'binary_crossentropy'
    metrics = ['accuracy']
  else:
    opt = RMSprop(lr=learnrate)
    from keras import backend as kb
    def root_mean_squared_error(y_true, y_pred):
      return kb.sqrt(kb.mean(kb.square(y_pred - y_true)))
    loss = root_mean_squared_error
    metrics = ['mean_squared_error']

# Compile the model with the desired optimizer, loss, and metric
  model.compile(optimizer=opt,loss=loss,metrics=metrics)
  return model

def getDefaultLeNetND( layers, shape, filtersz, format, conv_clss ):
  from tensorflow.keras.layers import (Activation,BatchNormalization,Dropout)
  kernel_sz1 = 5
  kernel_sz2 = 3
  stride_sz1 = 4
  stride_sz2 = 2
  dropout = 0.2
  
  layers.extend([
    conv_clss(filtersz, kernel_sz1, strides=stride_sz1, padding='same', name='conv_layer1',input_shape=shape,data_format=format),
    BatchNormalization(),
    Activation('relu'),
    conv_clss(filtersz, kernel_sz2, strides=stride_sz2, padding='same', name='conv_layer2',data_format=format),
    Dropout(dropout),
    BatchNormalization(),
    Activation('relu'),
    conv_clss(filtersz, kernel_sz2, strides=stride_sz2, padding='same', name='conv_layer3',data_format=format),
    Dropout(dropout),
    BatchNormalization(),
    Activation('relu'),
    conv_clss(filtersz, kernel_sz2, strides=stride_sz2, padding='same', name='conv_layer4',data_format=format),
    Dropout(dropout),
    BatchNormalization(),
    Activation('relu'),
    conv_clss(filtersz, kernel_sz2, strides=stride_sz2, padding='same', name='conv_layer5',data_format=format)
  ])
  return layers

def getDefaultUnet(isclassification,model_shape,nroutputs,
                    unetnszs=keras_dict['unetnszs'],
                    learnrate=keras_dict['learnrate'],
                    data_format='channels_last'):
  redirect_stdout()
  import keras
  restore_stdout()
  from keras.layers import Input
  from keras.models import Model
  from keras.optimizers import Adam

  input_shape = model_shape
  if data_format == 'channels_first':
#Tensorflow bug; always bad accuracy, no training
    data_format = 'channels_last'
    dims = model_shape[1:]
    input_shape = ( *dims, model_shape[0] )

  if isclassification:
    nroutputs = 1

  inputs = Input(input_shape)
  nrdims = getModelDims( model_shape, data_format )
  axis = -1
  from keras.layers import (Conv1D,Conv2D,Conv3D)
  from keras.layers import (MaxPooling1D,MaxPooling2D,MaxPooling3D)
  from keras.layers import (UpSampling1D,UpSampling2D,UpSampling3D)
  if nrdims == 3:
    lastconv = getDefaultUnetND( inputs, unetnszs, data_format, axis, nroutputs, Conv3D, MaxPooling3D, UpSampling3D )
  elif nrdims == 2:
    lastconv = getDefaultUnetND( inputs, unetnszs, data_format, axis, nroutputs, Conv2D, MaxPooling2D, UpSampling2D  )
  elif nrdims == 1:
    lastconv = getDefaultUnetND( inputs, unetnszs, data_format, axis, nroutputs, Conv1D, MaxPooling1D, UpSampling1D  )
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

def getDefaultUnetND( inputs, unetnszs, format, axis, nroutputs, conv_clss, pool_clss, upsamp_clss ):
  from keras.layers import concatenate

  poolsz1 = 2
  poolsz2 = 2
  poolsz3 = 2
  upscalesz = 2
  filtersz1 = unetnszs[0]
  filtersz2 = filtersz1 * poolsz2
  filtersz3 = filtersz2 * poolsz3
  filtersz4 = unetnszs[1]

  params = dict(kernel_size=3, activation='relu', padding='same', data_format=format)

  conv1 = conv_clss(filtersz1, **params)(inputs)
  conv1 = conv_clss(filtersz1, **params)(conv1)

  pool1 = pool_clss(pool_size=poolsz1,data_format=format)(conv1)

  conv2 = conv_clss(filtersz2, **params)(pool1)
  conv2 = conv_clss(filtersz2, **params)(conv2)
  pool2 = pool_clss(pool_size=poolsz2,data_format=format)(conv2)

  conv3 = conv_clss(filtersz3, **params)(pool2)
  conv3 = conv_clss(filtersz3, **params)(conv3)
  pool3 = pool_clss(pool_size=poolsz3,data_format=format)(conv3)

  conv4 = conv_clss(filtersz4, **params)(pool3)
  conv4 = conv_clss(filtersz4, **params)(conv4)

  up5 = concatenate([upsamp_clss(size=upscalesz,data_format=format)(conv4), conv3], axis=axis)
  conv5 = conv_clss(filtersz3, **params)(up5)
  conv5 = conv_clss(filtersz3, **params)(conv5)

  up6 = concatenate([upsamp_clss(size=poolsz2,data_format=format)(conv5), conv2], axis=axis)
  conv6 = conv_clss(filtersz2, **params)(up6)
  conv6 = conv_clss(filtersz2, **params)(conv6)

  up7 = concatenate([upsamp_clss(size=poolsz1,data_format=format)(conv6), conv1], axis=axis)
  conv7 = conv_clss(filtersz1, **params)(up7)
  conv7 = conv_clss(filtersz1, **params)(conv7)

  conv8 = conv_clss(nroutputs, 1, activation='sigmoid', data_format=format)(conv7)
  return conv8

def train(model,training,params=keras_dict,trainfile=None,logdir=None,withaugmentation=withaugmentation):
  redirect_stdout()
  import keras
  from keras.callbacks import EarlyStopping
  from dgbpy.keras_classes import TrainingSequence
  restore_stdout()
  
  infos = training[dgbkeys.infodictstr]
  classification = infos[dgbkeys.classdictstr]
  if classification:
    monitor = 'accuracy'
  else:
    monitor = 'loss'
  early_stopping = EarlyStopping(monitor=monitor, patience=params['patience'])
  LR_sched = adaptive_schedule(params['learnrate'],params['epochdrop'])
  callbacks = [early_stopping,LR_sched]
  batchsize = params['batch']
  if logdir != None:
    from keras.callbacks import TensorBoard
    tensor_board = TensorBoard(log_dir=logdir, \
                               batch_size=batchsize,\
                         write_graph=True, write_grads=False, write_images=True)
    callbacks.append( tensor_board )
  train_datagen = TrainingSequence( training, False, model, exfilenm=trainfile, batch_size=batchsize, with_augmentation=withaugmentation )
  validate_datagen = TrainingSequence( training, True, model, exfilenm=trainfile, batch_size=batchsize, with_augmentation=withaugmentation )
  nbchunks = len( infos[dgbkeys.trainseldicstr] )
  for ichunk in range(nbchunks):
    log_msg('Starting training iteration',str(ichunk+1)+'/'+str(nbchunks))
    if not train_datagen.set_chunk(ichunk) or not validate_datagen.set_chunk(ichunk):
      continue

    if batchsize == 1:
      log_msg( 'Training on', len(train_datagen), 'samples' )
      log_msg( 'Validate on', len(validate_datagen), 'samples' )
    else:
      log_msg( 'Training on', len(train_datagen), 'batches of', batchsize, 'samples' )
      log_msg( 'Validate on', len(validate_datagen), 'batches of', batchsize, 'samples' )

    redirect_stdout()
    model.fit(x=train_datagen,epochs=params['epoch'],\
              validation_data=validate_datagen,callbacks=callbacks)
    restore_stdout()

  keras.utils.print_summary( model, print_fn=log_msg )
  infos = updateModelShape( infos, model, True )
  infos = updateModelShape( infos, model, False )

  return model

def updateModelShape( infos, model, forinput ):
  if forinput:
    shapekey = dgbkeys.inpshapedictstr
    modelshape = model.input_shape
  else:
    shapekey = dgbkeys.outshapedictstr
    modelshape = model.output_shape
    
  exshape = infos[shapekey]
  if getDataFormat(model) == 'channels_first':
    modelshape = modelshape[2:]
  else:
    modelshape = modelshape[1:-1]

  if isinstance(exshape,int):
    return infos

  if len(exshape) == len(modelshape) and \
     np.prod(exshape,dtype=np.int64) == np.prod(modelshape,dtype=np.int64):
    return infos

  ret = ()
  i = 0
  for exdim in exshape:
    if exdim < 2:
      ret += (exdim,)
    else:
      ret += (modelshape[i],)
      i += 1
      
  infos[shapekey] = ret
  return infos

def save( model, outfnm ):
  try:
    model.save( outfnm, save_format='h5' )
  except Exception as e:
    model.save( outfnm )

def load( modelfnm, fortrain ):
  redirect_stdout()
  from tensorflow.keras.models import load_model
  try:
    ret = load_model( modelfnm, compile=fortrain )
  except ValueError:
    configfile = os.path.splitext( modelfnm )[0] + '.json'
    if not os.path.isfile(configfile):
      return None
    import json
    with open(configfile,'r') as f:
      model_json = json.load(f)
    from keras.models import model_from_json
    try:
      ret = model_from_json(model_json)
    except TypeError:
      model_json_str = json.dumps( model_json )
      ret = model_from_json( model_json_str )
    ret.load_weights(modelfnm)
      
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

def apply( model, samples, isclassification, withpred, withprobs, withconfidence, doprobabilities, \
           scaler=None, batch_size=None ):
  if batch_size == None:
    batch_size = keras_dict['batch']
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
      if not (doprobabilities or withconfidence):
        try:
          res = model.predict_classes( samples, batch_size=batch_size )
        except AttributeError:
          res = model.predict( samples, batch_size=batch_size )
    else:
      res = model.predict( samples, batch_size=batch_size )
    res = adaptFromModel(model,isclassification,res,inp_shape,ret_data_format=data_format)
    ret.update({dgbkeys.preddictstr: res})
 
  if isclassification and (doprobabilities or withconfidence or (withpred and len(ret)<1)):
    allprobs = model.predict( samples, batch_size=batch_size )
    allprobs = adaptFromModel(model,False,allprobs,inp_shape,ret_data_format=data_format)
    indices = None
    if withpred or withconfidence:
      N = 2
      indices = np.argpartition(allprobs,-N,axis=0)[-N:]
    if withpred:
      res = indices[-1:]
      ret.update({dgbkeys.preddictstr: res})
    if doprobabilities and len(withprobs) > 0:
      res = np.copy(allprobs[withprobs])
      ret.update({dgbkeys.probadictstr: res})
    if withconfidence:
      x = allprobs.shape[-1]
      sortedprobs = allprobs[indices.ravel(),np.tile(np.arange(x),N)].reshape(N,x)
      res = np.diff(sortedprobs,axis=0)
      ret.update({dgbkeys.confdictstr: res})

  return ret


def adaptToModel( model, samples, sample_data_format='channels_first' ):
  nrdims = len( model.input_shape ) - 2
  nrsamples = samples.shape[0]
  samples_nrdims = len(samples.shape)
  model_data_format = getDataFormat( model )
  modelcubeszs = getCubeletShape( model )
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

def need_channels_last():
  import tensorflow as tf
  if tf.test.is_built_with_cuda():
    gpudevs = tf.config.get_visible_devices('GPU')
    if len(gpudevs) > 0:
      return False
  return True
