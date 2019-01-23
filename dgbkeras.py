import os
import numpy as np

from odpy.common import log_msg
import dgbpy.hdf5 as dgbhdf5

platform = ('keras','Keras (tensorflow)')

def getMLPlatform():
  return platform[0]

def getUIMLPlatform():
  return platform[1]

lastlayernm = 'pre-softmax_layer'
keras_dict = {
  'decimation': False,
  'iters': 15,
  'epoch': 15,
  'batch': 32,
  'patience': 10
}

def getParams( dec=keras_dict['decimation'], iters=keras_dict['iters'],
              epochs=keras_dict['epoch'], batch=keras_dict['batch'],
              patience=keras_dict['patience'] ):
  ret = {
    'decimation': dec,
    'iters': iters,
    'epoch': epochs,
    'batch': batch,
    'patience': patience
  }
  if not dec:
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
  nrclasses = len(setup['examples'])
  multiclass = isclassification and nrclasses == 2
  stepout = setup['stepout']
  model = Sequential()
  model.add(Conv3D(50, (5, 5, 5), strides=(4, 4, 4), padding='same', name='conv_layer1', \
             input_shape=(nrinputs,2*stepout[0]+1,2*stepout[1]+1,2*stepout[2]+1), \
             data_format="channels_first"))
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

# initiate the Adam optimizer with a given learning rate
#  opt = 'rmsprop'
  opt = keras.optimizers.adam(lr=0.001)

# Compile the model with the desired optimizer, loss, and metric
  metrics = ['accuracy']
  if isclassification:
    if multiclass:
      loss = 'categorical_crossentropy'
    else:
      loss = 'binary_crossentropy'
    model.compile(optimizer=opt,loss=loss,metrics=metrics)
  else:
    model.compile(optimizer=opt,loss='rmsprop')
  return model

def train(model,training,params=keras_dict,trainfile=None):
  from odpy.common import redirect_stdout,restore_stdout
  redirect_stdout()
  import keras
  restore_stdout()
  from keras.callbacks import (EarlyStopping,LearningRateScheduler)
  early_stopping = EarlyStopping(monitor='acc', patience=params['patience'])
  LR_sched = LearningRateScheduler(schedule = adaptive_lr)
  num_bunch = params['iters']
  dec_fact = params['decimation']
  decimate = dec_fact
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
    if len(x_train.shape) < 4:
      x_train = np.expand_dims(x_train,axis=1)
    y_train = keras.utils.to_categorical(y_train,getNrClasses(model))
    redirect_stdout()
    hist = model.fit(x=x_train,y=y_train,callbacks=[early_stopping, LR_sched],shuffle=True, \
                        validation_split=0.2, \
                        batch_size=params['batch'], \
                        epochs=params['epoch'])
    #log_msg( hist.history )
    restore_stdout()

  keras.utils.print_summary( model, print_fn=log_msg )
  return model

def apply( model, samples, isclassification, batch_size=keras_dict['batch'] ):
  import keras
  ret = model.predict( samples )
  if not isclassification:
    return ret
  classes = model.predict_classes( samples )
  return (classes,ret)

def save( model, inpfnm, outfnm ):
  log_msg( 'Saving model.' )
  model.save( outfnm ) #Keep first!
  dgbhdf5.addInfo( inpfnm, getMLPlatform(), outfnm )
  log_msg( 'Model saved.' )

def load( modelfnm ):
  from odpy.common import redirect_stdout,restore_stdout
  redirect_stdout()
  from keras.models import load_model
  ret = load_model( modelfnm )
  restore_stdout()
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
