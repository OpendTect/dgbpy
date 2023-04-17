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
from pathlib import Path
from functools import partial
import time

from odpy.common import log_msg, redirect_stdout, restore_stdout
import odpy.hdf5 as odhdf5
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
try:
  import dgbpy.keras_classes as kc
  from dgbpy.mlmodel_keras_dGB import root_mean_squared_error, cross_entropy_balanced
  from keras.callbacks import Callback
except ModuleNotFoundError:
  pass
from dgbpy.mlio import announceShowTensorboard, announceTrainingFailure, announceTrainingSuccess

def hasKeras():
  try:
    import tensorflow,keras
  except ModuleNotFoundError:
    return False
  return True

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

withtensorboard = dgbkeys.getDefaultTensorBoard()

platform = (dgbkeys.kerasplfnm,'Keras (tensorflow)')

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

defbatchstr = 'defaultbatchsz'

default_transforms = []
keras_dict = {
  dgbkeys.decimkeystr: False,
  'nbchunk': 10,
  'epochs': 15,
  'batch': 32,
  'patience': 5,
  'learnrate': 1e-4,
  'epochdrop': 5,
  'split': 0.2,
  'nbfold': 5,
  'type': None,
  'prefercpu': None,
  'transform': default_transforms,
  'scale': dgbkeys.globalstdtypestr,
  'withtensorboard': withtensorboard,
  'tofp16': True
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
    dgbkeys.prefercpustr: get_cpu_preference(),
    'batchsizes': cudacores,
    defbatchstr: keras_dict['batch']
   }
  return json.dumps( ret )

def set_compute_device( prefercpu ):
  if not prefercpu:
      return
  from tensorflow import config as tfconfig
  cpudevs = tfconfig.list_physical_devices('CPU')
  tfconfig.set_visible_devices( cpudevs )

def use_mixed_precision():
  '''
      Use this function to set the global policy to mixed precision.
  '''
  import tensorflow as tf
  if can_use_gpu(): 
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

def getParams( dodec=keras_dict[dgbkeys.decimkeystr], nbchunk=keras_dict['nbchunk'],
               epochs=keras_dict['epochs'],
               batch=keras_dict['batch'], patience=keras_dict['patience'],
               learnrate=keras_dict['learnrate'],epochdrop=keras_dict['epochdrop'],
               nntype=keras_dict['type'],prefercpu=keras_dict['prefercpu'],transform=keras_dict['transform'],
               validation_split=keras_dict['split'], nbfold=keras_dict['nbfold'],
               scale = keras_dict['scale'],withtensorboard=keras_dict['withtensorboard'], tofp16=keras_dict['tofp16']):
  ret = {
    dgbkeys.decimkeystr: dodec,
    'nbchunk': nbchunk,
    'epochs': epochs,
    'batch': batch,
    'patience': patience,
    'learnrate': learnrate,
    'epochdrop': epochdrop,
    'split': validation_split,
    'nbfold': nbfold,
    'type': nntype,
    'transform': transform,
    'scale': scale,
    'withtensorboard': withtensorboard,
    'tofp16': tofp16
  }
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
  from keras.callbacks import LearningRateScheduler
  def adaptive_lr(input_int):
    drop = 0.5
    return initial_lrate * math.pow(drop,
                                    math.floor((1+input_int)/epochs_drop))
    # return the learning rate (quite arbitrarily decaying)
  return LearningRateScheduler(adaptive_lr)


def get_data_format( model ):
  layers = model.layers
  for i in range(len(layers)):
    laycfg = layers[i].get_config()
    if 'data_format' in laycfg:
      return laycfg['data_format']
  return None

def hasValidCubeletShape( cubeszs ):
  if None in cubeszs:
    return False
  return True

def getCubeletShape( model ):
  data_format = get_data_format( model )
  if data_format == 'channels_first':
    cubeszs = model.input_shape[2:]
  elif data_format == 'channels_last':
    cubeszs = model.input_shape[1:-1]
  return cubeszs

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

def getModelsByType( learntype, classification, ndim ):
    predtype = kc.DataPredType.Continuous
    outtype = kc.OutputType.Pixel
    dimtype = kc.DimType(ndim)
    if dgbhdf5.isImg2Img(learntype):
      outtype = kc.OutputType.Image
    if classification or dgbhdf5.isSeisClass( learntype ):
      predtype = kc.DataPredType.Classification
    if dgbhdf5.isSegmentation(learntype):
      predtype = kc.DataPredType.Segmentation
    return kc.UserModel.getNamesByType(pred_type=predtype, out_type=outtype, dim_type=dimtype)

def getModelsByInfo( infos ):
    shape = infos[dgbkeys.inpshapedictstr]
    if isinstance(shape,int):
        ndim = 1
    else:
        ndim = len(shape)-1
    modelstypes = getModelsByType( infos[dgbkeys.learntypedictstr],
                                   infos[dgbhdf5.classdictstr],
                                   ndim )
    if len(modelstypes) < 1:
        return None
    return modelstypes[0]

def getDefaultModel(setup,type=keras_dict['type'],
                     learnrate=keras_dict['learnrate'],
                     data_format='channels_first'):
  isclassification = setup[dgbhdf5.classdictstr]
  if isclassification:
    nroutputs = len(setup[dgbkeys.classesdictstr])
  else:
    nroutputs = dgbhdf5.getNrOutputs( setup )
  if len(kc.UserModel.mlmodels) < 1:
    kc.UserModel.mlmodels = kc.UserModel.findModels()
  if type==None:
      type = getModelsByInfo( setup )

  nrattribs = dgbhdf5.getNrAttribs(setup)
  model_shape = get_model_shape( setup[dgbkeys.inpshapedictstr], nrattribs,
                                 attribfirst=data_format=='channels_first' )

  if kc.UserModel.findName(type):
    return kc.UserModel.findName(type).model(model_shape, nroutputs,
                                             learnrate,
                                             data_format=data_format)
  return None

def hasFastprogress():
    try:
        import fastprogress
    except ModuleNotFoundError:
        return False
    return True

if hasFastprogress():
    from fastprogress.fastprogress import master_bar, progress_bar
class ProgressBarCallback(Callback):
  def __init__(self, config):
    """This method is called before training begins """
    self.train_datagen = config.get('train_datagen')
    self.valid_datagen = config.get('valid_datagen')

  def on_train_begin(self, logs=None):
    epochs = self.params['epochs']
    self.mbar = master_bar(range(epochs))
    self.logger = partial(self.mbar.write, table=True)  

  def on_epoch_begin (self, epoch, logs=None): 
    self.epoch = epoch
    self.pb = progress_bar(self.train_datagen, parent=self.mbar)
    self.mbar.update(epoch)
    self.start_time = time.time()

  def on_epoch_end(self, epoch, logs=None):
    if not epoch:
      names = [i.name for i in self.model.metrics] 
      results = ['Epochs']
      for name in names: results += [f'Train {name}']
      for name in names: results += [f'Valid {name}']
      results+=['Time']
      self.logger(results) 
    stats = [f'{stat:.4f}' for stat in logs.values()]
    stats += [dgbkeys.format_time(time.time() - self.start_time)]
    self.logger([epoch+1]+stats)

  def on_test_begin(self, logs=None):
    self.pb = progress_bar(self.valid_datagen, parent=self.mbar)
    self.mbar.update(self.epoch)

  def on_train_batch_end(self, batch, logs=None): 
    self.pb.update(batch)

  def on_test_batch_end(self, batch, logs=None): 
    self.pb.update(batch)

  def on_train_end(self, logs=None):
   self.mbar.on_iter_end()

class ProgressNoBarCallback(Callback):
  def __init__(self, config): pass

  def on_train_begin(self, logs=None):
    self.logger = log_msg

  def on_epoch_begin(self, epoch, logs=None):
    self.start_time = time.time()

  def on_epoch_end(self, epoch, logs=None):
    _logs = logs.copy()
    _logs.update({'Time': dgbkeys.format_time(time.time() - self.start_time)})
    self.logger(f'----------------- Epoch {epoch+1} ------------------')
    for key,val in _logs.items():
      self.logger(f'{key}: {val}')

class BokehProgressCallback(Callback):
    """Send progress message to bokeh"""
    def __init__(self, config):
      self.train_datagen, self.valid_datagen = config.get('train_datagen'), config.get('valid_datagen')
      self.ichunk, self.nbchunks = config.get('ichunk'), config.get('nbchunks')
      self.ifold, self.nbfolds = config.get('ifold'), config.get('nbfolds')
      self.isCrossVal = config.get('isCrossVal')

    def on_train_begin(self, logs=None):
      self.on_train_begin_chunk()
      self.on_train_begin_fold()
  
    def on_epoch_begin(self, epoch, logs=None):
      self.ntrain_steps = len(self.train_datagen)
      self.nvalid_steps = len(self.valid_datagen)
      if epoch==0:
        restore_stdout()
        print('--Epoch '+str(epoch)+' of '+str(self.params['epochs'])+' --', flush=True)
        restore_stdout()

    def on_epoch_end(self, epoch, logs=None):
      restore_stdout()
      print('--Epoch '+str(epoch+1)+' of '+str(self.params['epochs'])+' --', flush=True)
      restore_stdout()
      if epoch+1 == self.params['epochs']:
        restore_stdout()
        print('--Training Ended--', flush=True)
        restore_stdout()

    def on_train_batch_begin(self, batch, logs=None):
      if batch == 0:
        restore_stdout()
        print('--Iter '+str(batch)+' of '+str(self.ntrain_steps)+' --', flush=True)
        restore_stdout()

    def on_test_batch_begin(self, batch, logs=None):
      if batch == 0:
        restore_stdout()
        print('--Iter '+str(batch)+' of '+str(self.nvalid_steps)+' --', flush=True)
        restore_stdout()

    def on_train_batch_end(self, batch, logs=None):
      restore_stdout()
      print('--Iter '+str(batch+1)+' of '+str(self.ntrain_steps)+' --', flush=True)
      restore_stdout()

    def on_test_batch_end(self, batch, logs=None):
      restore_stdout()
      print('--Iter '+str(batch+1)+' of '+str(self.nvalid_steps)+' --', flush=True)
      restore_stdout()
    
    def on_train_begin_chunk(self):
      restore_stdout()
      print('--Chunk_Number '+str(self.ichunk)+' of '+str(self.nbchunks)+' --', flush=True)
      restore_stdout()

    def on_train_begin_fold(self):
      if self.isCrossVal:
        restore_stdout()
        print('--Fold_bkh '+str(self.ifold)+' of '+str(self.nbfolds)+' --', flush=True)
        restore_stdout()

class LogNrOfSamplesCallback(Callback):
  def __init__(self, config):
    self.logger = log_msg
    self.train_datagen, self.valid_datagen = config.get('train_datagen'), config.get('valid_datagen')
    self.ifold, self.nbfolds = config.get('ifold'), config.get('nbfolds')
    self.batchsize, self.isCrossVal = config.get('batchsize'), config.get('isCrossVal')
  def on_train_begin(self, logs=None):
    if self.batchsize == 1:
      log_msg( 'Training on', len(self.train_datagen), 'samples' )
      log_msg( 'Validate on', len(self.valid_datagen), 'samples' )
    else:
      log_msg( 'Training on', len(self.train_datagen), 'batches of', self.batchsize, 'samples' )
      log_msg( 'Validate on', len(self.valid_datagen), 'batches of', self.batchsize, 'samples' )
    self.on_train_begin_fold()

  def on_train_begin_fold(self):
    if self.isCrossVal:
      restore_stdout()
      self.logger(f'----------------- Fold {self.ifold}/{self.nbfolds} ------------------')
      restore_stdout()

class TransformCallback(Callback):
  def __init__(self, config):
    self.train_datagen = config.get('train_datagen')

  def on_epoch_begin(self, epoch, logs=None):
    self.train_datagen.set_transform_seed()


def epoch0endCB(epoch, logs):
  if epoch==0:
    announceShowTensorboard()
    
def init_callbacks(monitor,params,logdir,silent,custom_config, cbfn=None):
  from keras.callbacks import EarlyStopping, LambdaCallback
  early_stopping = EarlyStopping(monitor=monitor, patience=params['patience'])
  LR_sched = adaptive_schedule(params['learnrate'],params['epochdrop'])
  callbacks = [early_stopping,LR_sched]
  if params['withtensorboard']:
    epoch0end = LambdaCallback(on_epoch_end=epoch0endCB)
    callbacks.append(epoch0end)
  if logdir != None:
    from keras.callbacks import TensorBoard
    tensor_board = TensorBoard(log_dir=logdir, \
                         write_graph=True, write_grads=False, write_images=True)
    callbacks.append( tensor_board )

  _custom_builtin_cbs = [TransformCallback, LogNrOfSamplesCallback, ProgressBarCallback, ProgressNoBarCallback, BokehProgressCallback]

  if hasFastprogress() and not silent:
     prog_cb = ProgressBarCallback
  else: 
    prog_cb = ProgressNoBarCallback

  custom_cbs = [prog_cb, LogNrOfSamplesCallback, TransformCallback]
  for cb in cbfn+custom_cbs:
    if cb in _custom_builtin_cbs:
      cb = cb(custom_config)
    callbacks = [cb]+callbacks
  return callbacks


def train(model,training,params=keras_dict,trainfile=None,silent=False,cbfn=None,logdir=None,tempnm=None):
  redirect_stdout()
  import keras
  from dgbpy.keras_classes import TrainingSequence
  import tensorflow as tf
  restore_stdout()

  infos = training[dgbkeys.infodictstr]
  classification = infos[dgbkeys.classdictstr]
  cbfn = dgbkeys.listify(cbfn)
  if classification:
    monitor = 'accuracy'
  else:
    monitor = 'loss'
  batchsize = params['batch']
  transform, scale = params['transform'], params['scale']
  train_datagen = TrainingSequence( training, False, model, exfilenm=trainfile, batch_size=batchsize, scale=scale, transform=transform, tempnm=tempnm )
  validate_datagen = TrainingSequence( training, True, model, exfilenm=trainfile, batch_size=batchsize, scale=scale )
  nbchunks = len( infos[dgbkeys.trainseldicstr] )

  for ichunk in range(nbchunks):
    log_msg('Starting training iteration',str(ichunk+1)+'/'+str(nbchunks))
    try:
      if not train_datagen.set_chunk(ichunk) or not validate_datagen.set_chunk(ichunk):
        continue
    except Exception as e:
      log_msg('')
      log_msg('Data loading failed because of insufficient memory')
      log_msg('Try to lower the batch size and restart the training')
      log_msg('')
      announceTrainingFailure()
      raise e
    if  len(train_datagen) < 1 or len(validate_datagen) < 1:
      log_msg('')
      log_msg('There is not enough data to train on')
      log_msg('Extract more data and restart')
      log_msg('')
      announceTrainingFailure()
      raise TypeError
    redirect_stdout()
    isCrossVal = dgbhdf5.isCrossValidation(infos)
    config = { 'train_datagen':train_datagen, 'valid_datagen':validate_datagen,
                'ichunk':ichunk+1, 'nbchunks':nbchunks,'isCrossVal':isCrossVal, 'batchsize':batchsize }
    try:
      if not isCrossVal:
        callbacks = init_callbacks(monitor, params,logdir,silent,config,cbfn=cbfn)
        model.fit(x=train_datagen,epochs=params['epochs'],verbose=0,
                  validation_data=validate_datagen,callbacks=callbacks)
      else:
        nbfolds = len(infos[dgbkeys.trainseldicstr][ichunk])
        for ifold in range(1, nbfolds+1):
          train_datagen.set_fold(ichunk, ifold)
          validate_datagen.set_fold(ichunk, ifold)
          config['ifold'], config['nbfolds'] = ifold, nbfolds
          callbacks = init_callbacks(monitor,params,logdir,silent,config,cbfn=cbfn)
          if ifold != 1: # start transfer from second fold
            transfer(model)
          model.fit(x=train_datagen,epochs=params['epochs'],verbose=0,validation_data=validate_datagen,callbacks=callbacks)
      announceTrainingSuccess()
    except Exception as e:
      log_msg('')
      log_msg('Training failed because of insufficient memory')
      log_msg('Try to lower the batch size and restart the training')
      log_msg('')
      announceTrainingFailure()
      raise e

    restore_stdout()

  try:
    keras.utils.print_summary( model, print_fn=log_msg )
  except:
    pass
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
  if get_data_format(model) == 'channels_first':
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
  except Exception:
    model.save( outfnm )

def load( modelfnm, fortrain, infos=None, pars=keras_dict ):
  redirect_stdout()
  dgb_defs = {
    'cross_entropy_balanced': cross_entropy_balanced,
    'root_mean_squared_error': root_mean_squared_error,
  }
  from tensorflow.keras.models import load_model
  try:
    ret = load_model( modelfnm, compile=fortrain, custom_objects=dgb_defs )
    if fortrain and not infos == None:
        iscompiled = True
        try:
          h5file = odhdf5.openFile( modelfnm, 'r' )
          iscompiled = odhdf5.hasAttr( h5file, 'training_config' )
        finally:
          h5file.close()
        if not iscompiled:
          from dgbpy.mlmodel_keras_dGB import compile_model
          learnrate = keras_dict['learnrate']
          if not pars == None and 'learnrate' in pars:
            learnrate = pars['learnrate']
          nroutputs = dgbhdf5.getNrOutputs( infos )
          isregression = dgbhdf5.isRegression( infos )
          isunet = dgbhdf5.isImg2Img( infos )
          ret = compile_model( ret, nroutputs, isregression, isunet, learnrate )
  except ValueError:
    configfile = os.path.splitext( modelfnm )[0] + '.json'
    if not os.path.isfile(configfile):
      return None
    import json
    with open(configfile,'r') as f:
      model_json = json.load(f)
    from keras.models import model_from_json
    try:
      ret = model_from_json(model_json, custom_objects=dgb_defs)
    except TypeError:
      model_json_str = json.dumps( model_json )
      ret = model_from_json( model_json_str, custom_objects=dgb_defs )
    ret.load_weights(modelfnm)

  restore_stdout()
  return ret

def transfer( model ):
  from keras.layers import (Conv1D,Conv2D,Conv3D,Dense)
  layers = model.layers
  for layer in layers:
    layer.trainable = False

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

def apply( model, samples, isclassification, withpred, withprobs, \
           withconfidence, doprobabilities, dictinpshape=None, scaler=None, batch_size=None ):
  if batch_size == None:
    batch_size = keras_dict['batch']
  redirect_stdout()
  import keras
  restore_stdout()
  ret = {}

  inp_shape = samples.shape
  data_format = 'channels_first'
  samples = adaptToModel( model, samples, dictinpshape, sample_data_format=data_format )
  model_outshape = model.output_shape
  img2img = len(model_outshape) > 2
  if len(model_outshape) <= 2:
    nroutputs = model_outshape[-1]
  else:
    model_data_format = get_data_format( model )
    if model_data_format == 'channels_first':
      nroutputs = model_outshape[1]
    else:
      nroutputs = model_outshape[-1]

  res = None
  if withpred:
    if isclassification:
      if not (doprobabilities or withconfidence) and hasattr(model, 'predict_classes'):
        try:
          res = model.predict_classes( x=samples, batch_size=batch_size )
        except AttributeError:
          pass
    if not isinstance( res, np.ndarray ):
      res = model.predict( x=samples, batch_size=batch_size )
      res = adaptFromModel(model,res,inp_shape,ret_data_format=data_format)
      ret.update({dgbkeys.preddictstr: res})

  if isclassification and (doprobabilities or withconfidence or withpred):
    if len(ret)<1:
      allprobs = model.predict( x=samples, batch_size=batch_size )
      allprobs = adaptFromModel(model,allprobs,inp_shape,ret_data_format=data_format)
    else:
      allprobs = ret[dgbkeys.preddictstr]
    indices = None
    if withconfidence or not img2img or (img2img and nroutputs>2):
      N = 2
      if img2img:
        indices = np.argpartition(allprobs,-N,axis=1)[:,-N:]
      else:
        indices = np.argpartition(allprobs,-N,axis=0)[-N:]
    if withpred and isinstance( indices, np.ndarray ):
      if img2img:
        ret.update({dgbkeys.preddictstr: indices[:,-1:]})
      else:
        ret.update({dgbkeys.preddictstr: indices[-1:]})
    if doprobabilities and len(withprobs) > 0:
      res = np.copy(allprobs[withprobs])
      ret.update({dgbkeys.probadictstr: res})
    if withconfidence:
      x = allprobs.shape[-1]
      sortedprobs = allprobs[indices.ravel(),np.tile(np.arange(x),N)].reshape(N,x)
      res = np.diff(sortedprobs,axis=0)
      ret.update({dgbkeys.confdictstr: res})

  return ret

def adaptToModel( model, samples, dictinpshape=None, sample_data_format='channels_first' ):
  nrdims = len( model.input_shape ) - 2
  nrsamples = samples.shape[0]
  samples_nrdims = len(samples.shape)
  model_data_format = get_data_format( model )
  modelcubeszs = getCubeletShape( model )
  if not hasValidCubeletShape(modelcubeszs) and dictinpshape != None:
    modelcubeszs = dictinpshape
  if not hasValidCubeletShape(modelcubeszs):
    raise Exception("Invalid input shape found")
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

def adaptFromModel( model, samples, inp_shape, ret_data_format ):
  nrdims = len( model.output_shape )
  if nrdims == 2:
    return samples.transpose()

  nrpts = inp_shape[0]
  cube_shape = (nrpts,)
  model_data_format = get_data_format( model )
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

def get_validation_data( trainseq ):
    x_data = list()
    y_data = list()
    for i in range(len(trainseq)):
        (x,y) = trainseq.__getitem__(i)
        if len(x) > 0 and len(y) > 0:
            x_data.append( x )
            y_data.append( y )
    x_data = np.concatenate( x_data )
    y_data = np.concatenate( y_data )
    batch_size = trainseq._batch_size
    return (x_data,y_data,batch_size)
