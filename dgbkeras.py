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
withtensorboard = True
if 'KERAS_WITH_TENSORBOARD' in os.environ:
  withtensorboard = not ( os.environ['KERAS_WITH_TENSORBOARD'] == False or \
                          os.environ['KERAS_WITH_TENSORBOARD'] == 'No' )
if withtensorboard:
  from keras.callbacks import (EarlyStopping,LearningRateScheduler,TensorBoard)
else:
  from keras.callbacks import (EarlyStopping,LearningRateScheduler)


platform = (dgbkeys.kerasplfnm,'Keras (tensorflow)')
mltypes = (\
            ('lenet','LeNet - Malenov'),\
#            ('squeezenet','SqueezeNet'),\
#            ('other','MobilNet V2'),\
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

def getUiModelTypes():
  return dgbkeys.getNames( mltypes )

def isLeNet( mltype ):
  return mltype == mltypes[0][0] or mltype == mltypes[0][1]

def isSqueezeNet( mltype ):
  return mltype == mltypes[1][0] or mltype == mltypes[1][1]

def isMobilNetV2( mltype ):
  return mltype == mltypes[2][0] or mltype == mltypes[2][1]

firstconvlayernm = 'conv_layer1'
lastlayernm = 'pre-softmax_layer'
keras_dict = {
  dgbkeys.decimkeystr: False,
  'nbchunk': 10,
  'epoch': 15,
  'batch': 32,
  'patience': 5,
  'learnrate': 0.01,
  'epochdrop': 5,
  'type': mltypes[0][0],
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

def getLayer( model, name ):
  for lay in model.layers:
    if lay.get_config()['name'] == name:
      return lay
  return None

def getDataFormat( model ):
  convlay1_config = getLayer(model,firstconvlayernm).get_config()
  return convlay1_config['data_format']

def getCubeletStepout( model ):
  convlay1_config = getLayer(model,firstconvlayernm).get_config()
  data_format = getDataFormat( model )
  if data_format == 'channels_first':
    cubeszs = convlay1_config['batch_input_shape'][2:]
  elif data_format == 'channels_last':
    cubeszs = convlay1_config['batch_input_shape'][1:-1]
  stepout = tuple()
  for cubesz in cubeszs:
    stepout += (int((cubesz-1)/2),)
  return (stepout,cubeszs)

def getNrClasses( model ):
  return getLayer(model,lastlayernm).get_config()['units']

def getDefaultModel(setup,type=keras_dict['type'],
                    learnrate=keras_dict['learnrate'],
                    data_format='channels_first'):
  redirect_stdout()
  import keras
  restore_stdout()
  from keras.layers import (Activation,Conv3D,Dense,Dropout,Flatten)
  from keras.layers.normalization import BatchNormalization
  from keras.models import (Sequential)

  nrinputs = dgbhdf5.get_nr_attribs(setup)
  isclassification = setup[dgbhdf5.classdictstr]
  if isclassification:
    nroutputs = len(setup[dgbkeys.classesdictstr])
  else:
    nroutputs = 1
  stepout = setup[dgbkeys.stepoutdictstr]
  try: 
    steps = (nrinputs,2*stepout[0]+1,2*stepout[1]+1,2*stepout[2]+1)
  except TypeError:
    steps = (nrinputs,1,1,2*stepout+1)
  model = Sequential()
  model.add(Conv3D(50, (5, 5, 5), strides=(4, 4, 4), padding='same', \
            name=firstconvlayernm,input_shape=steps,data_format=data_format))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding='same', name='conv_layer2', data_format=data_format))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding='same', name='conv_layer3',data_format=data_format))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding='same', name='conv_layer4',data_format=data_format))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding='same', name='conv_layer5',data_format=data_format))
  model.add(Flatten())
  model.add(Dense(50,name = 'dense_layer1'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dense(10,name = 'attribute_layer'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dense(nroutputs, name=lastlayernm))
  model.add(BatchNormalization())
  model.add(Activation('softmax'))

# initiate the model compiler options
  metrics = ['accuracy']
  if isclassification:
    opt = keras.optimizers.adam(lr=learnrate)
    if nroutputs > 2:
      loss = 'categorical_crossentropy'
    else:
      loss = 'binary_crossentropy'
  else:
    opt = keras.optimizers.RMSprop(lr=learnrate)
#    set_epsilon( 1 )
    from keras import backend as K
    def root_mean_squared_error(y_true, y_pred):
      return K.sqrt(K.mean(K.square(y_pred - y_true)))
    loss = root_mean_squared_error

# Compile the model with the desired optimizer, loss, and metric
  model.compile(optimizer=opt,loss=loss,metrics=metrics)
  return model

def train(model,training,params=keras_dict,trainfile=None):
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
  logfilenm = get_log_file()
  if logfilenm != None:
    logdir = os.path.dirname(logfilenm)
    survdir = os.path.basename(logdir)
    logdir = os.path.dirname(logdir)
    logdir = os.path.join( logdir, 'MachineLearning' )
    jobnm = survdir + '_run'
    if withtensorboard and os.path.exists(logdir):
      nrsavedruns = 0
      with os.scandir(logdir) as it:
        for entry in it:
          if entry.name.startswith(jobnm) and entry.is_dir():
            nrsavedruns += 1
      logdir = os.path.join( logdir, jobnm+str(nrsavedruns+1)+'_'+'m'.join( datetime.now().isoformat().split(':')[:-1] ) )
      tensor_board = TensorBoard(log_dir=logdir, histogram_freq=1, batch_size=batchsize,\
                               write_graph=True, write_grads=True, write_images=True)
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
    while len(x_train.shape) < 5:
      x_train = np.expand_dims(x_train,axis=len(x_train.shape)-1)
    if classification:
      nrclasses = getNrClasses(model)
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

def load( modelfnm ):
  redirect_stdout()
  from keras.models import load_model
  ret = load_model( modelfnm, compile=False )
  restore_stdout()
  return ret

def apply( model, samples, isclassification, withpred, withprobs, withconfidence, doprobabilities, scaler=None, batch_size=keras_dict['batch'] ):
  redirect_stdout()
  import keras
  restore_stdout()
  ret = {}
  res = None
  if getDataFormat(model) == 'channels_last':
    samples = transform( samples )
    
  if withpred:
    if isclassification:
      res = model.predict_classes( samples, batch_size=batch_size )
    else:
      res = model.predict( samples, batch_size=batch_size )
      res = res.transpose()
      #TODO: make one output array for each column
    ret.update({dgbkeys.preddictstr: res})
 
  if isclassification and (doprobabilities or withconfidence):
    allprobs = model.predict( samples, batch_size=batch_size )
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

def transform( samples ):
  nrsamples = samples.shape[0]
  nrattribs = samples.shape[1]
  cube_shape = samples.shape[2:]
  ret = np.empty( (nrsamples, cube_shape[0], cube_shape[1], cube_shape[2], nrattribs ), dtype = samples.dtype )
  for iattr in range(nrattribs):
    ret[:,:,:,:,iattr] = samples[:,iattr]
  return ret

def transformBack( samples ):
  nrsamples = samples.shape[0]
  cube_shape = samples.shape[1:-1]
  nrattribs = samples.shape[-1]
  ret = np.empty( (nrsamples, nrattribs, cube_shape[0], cube_shape[1], cube_shape[2] ), dtype = samples.dtype )
  for iattr in range(nrattribs):
    ret[:,iattr] = samples[:,:,:,:,iattr]
  return ret

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
