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
import numpy as np

from odpy.common import log_msg
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
platform = (dgbkeys.kerasplfnm,'Keras (tensorflow)')

def getMLPlatform():
  return platform[0]

def getUIMLPlatform():
  return platform[1]

lastlayernm = 'pre-softmax_layer'
keras_dict = {
  dgbkeys.decimkeystr: False,
  'dec': 0.1,
  'iters': 15,
  'epoch': 15,
  'batch': 32,
  'patience': 10
}

def getParams( dodec=keras_dict[dgbkeys.decimkeystr], dec=keras_dict['dec'],
               iters=keras_dict['iters'], epochs=keras_dict['epoch'],
               batch=keras_dict['batch'], patience=keras_dict['patience'] ):
  ret = {
    dgbkeys.decimkeystr: dodec,
    'dec': dec,
    'iters': iters,
    'epoch': epochs,
    'batch': batch,
    'patience': patience
  }
  if not dodec:
    ret['dec'] = 0
    ret['iters'] = 1
  return ret

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
  from odpy.common import redirect_stdout,restore_stdout
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
            name='conv_layer1',input_shape=steps,data_format="channels_first"))
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
  model.add(Dense(nroutputs, name=lastlayernm))
  model.add(BatchNormalization())
  model.add(Activation('softmax'))

# initiate the model compiler options
  learnrate = 0.001
  metrics = ['accuracy']
  if isclassification:
    opt = keras.optimizers.adam(lr=learnrate)
    if nroutputs > 2:
      loss = 'categorical_crossentropy'
    else:
      loss = 'binary_crossentropy'
  else:
    opt = keras.optimizers.RMSprop(lr=learnrate)
    from keras import backend as K
    def root_mean_squared_error(y_true, y_pred):
      return K.sqrt(K.mean(K.square(y_pred - y_true)))
    loss = root_mean_squared_error

# Compile the model with the desired optimizer, loss, and metric
  model.compile(optimizer=opt,loss=loss,metrics=metrics)
  return model

def train(model,training,params=keras_dict,trainfile=None):
  from odpy.common import redirect_stdout,restore_stdout
  redirect_stdout()
  import keras
  restore_stdout()
  from keras.callbacks import (EarlyStopping,LearningRateScheduler)
  classification = training[dgbkeys.infodictstr][dgbkeys.classdictstr]
  if classification:
    monitor = 'acc'
  else:
    monitor = 'loss'
  early_stopping = EarlyStopping(monitor=monitor, patience=params['patience'])
  LR_sched = LearningRateScheduler(schedule = adaptive_lr)
  num_bunch = params['iters']
  decimate = params[dgbkeys.decimkeystr]
  dec_fact = params['dec']
  x_train = {}
  y_train = {}
  if not decimate:
    if not dgbkeys.xtraindictstr in training:
      log_msg('No data to train the model')
      return model
    x_train = training[dgbkeys.xtraindictstr]
    y_train = training[dgbkeys.ytraindictstr]
  for repeat in range(num_bunch):
    log_msg('Starting iteration',str(repeat+1)+'/'+str(num_bunch))
    log_msg('Starting training data creation:')
    if decimate and trainfile != None:
      import dgbpy.mlio as dgbmlio
      trainbatch = dgbmlio.getTrainingData( trainfile,dec_fact)
      if not dgbkeys.xtraindictstr in trainbatch:
        continue
      x_train = trainbatch[dgbkeys.xtraindictstr]
      y_train = trainbatch[dgbkeys.ytraindictstr]
    log_msg('Finished creating',len(x_train),'examples!')
    while len(x_train.shape) < 5:
      x_train = np.expand_dims(x_train,axis=len(x_train.shape)-1)
    if classification:
      y_train = keras.utils.to_categorical(y_train,getNrClasses(model))
    redirect_stdout()
 
    mask = np.random.permutation(len(x_train))
    x_train = x_train[mask]
    y_train = y_train[mask]
   
    hist = model.fit(x=x_train,y=y_train,callbacks=[early_stopping, LR_sched],shuffle=True, \
                        validation_split=0.2, \
                        batch_size=params['batch'], \
                        epochs=params['epoch'])
    #log_msg( hist.history )
    restore_stdout()

  keras.utils.print_summary( model, print_fn=log_msg )
  return model

def save( model, inpfnm, outfnm ):
  log_msg( 'Saving model.' )
  model.save( outfnm ) #Keep first!
  dgbhdf5.addInfo( inpfnm, getMLPlatform(), outfnm )
  log_msg( 'Model saved.' )

def load( modelfnm ):
  from odpy.common import redirect_stdout,restore_stdout
  redirect_stdout()
  from keras.models import load_model
  ret = load_model( modelfnm, compile=False )
  restore_stdout()
  return ret

def apply( model, samples, scaler=None, applyinfo=None, batch_size=keras_dict['batch'] ):
  if applyinfo == None:
    isclassification = True
    withclass = isclassification
    withprobs = []
    withconfidence=False
  else:
    isclassification = applyinfo[dgbkeys.classdictstr]
    withclass = applyinfo[dgbkeys.withclass]
    withconfidence= applyinfo[dgbkeys.withconfidence]
    withprobs = applyinfo[dgbkeys.withprobs]

  doprobabilities = len(withprobs) > 0

  import keras
  ret = {}
  if isclassification and withclass:
    ret.update({dgbkeys.preddictstr: \
                model.predict_classes( samples, batch_size=batch_size )})
  else:
    ret.update({dgbkeys.preddictstr: \
                model.predict( samples, batch_size=batch_size )})

  if isclassification and (doprobabilities or withconfidence):
    allprobs = model.predict( samples, batch_size=batch_size )
    if doprobabilities:
      ret.update({dgbkeys.probadictstr: \
                  np.copy(allprobs[:,withprobs],allprobs.dtype)})
    if withconfidence:
      N = 2
      indices = np.argpartition(allprobs,-N,axis=1)[:,-N:]
      x = len(allprobs)
      sortedprobs = allprobs[np.repeat(np.arange(x),N),indices.ravel()].reshape(x,N)
      ret.update({dgbkeys.confdictstr: np.diff(sortedprobs,axis=1)})

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
