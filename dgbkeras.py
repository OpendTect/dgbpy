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

from odpy.common import log_msg, redirect_stdout, restore_stdout
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
  'type': None,
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
               nntype=keras_dict['type'],prefercpu=keras_dict['prefercpu']):
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

def getCubeletShape( model ):
  data_format = get_data_format( model )
  if data_format == 'channels_first':
    cubeszs = model.input_shape[2:]
  elif data_format == 'channels_last':
    cubeszs = model.input_shape[1:-1]
  return cubeszs

def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            try:
              rm_tree(child)
            except OSError:
              pass
    try:
      pth.rmdir()
    except OSError:
      pass

def getLogDir( examplenm, basedir, clearlogs, args ):
  if not withtensorboard or basedir == None or not Path(basedir).exists():
    return None
  logdir = Path(basedir) / Path(examplenm).stem
  if logdir.exists():
      if clearlogs:
         for child in logdir.glob('*'):
            rm_tree(child)
  else:
      try:
         logdir.mkdir()
      except:
         return None

  if dgbkeys.surveydictstr in args:
    jobnm = args[dgbkeys.surveydictstr][0] + '_run'
  else:
    jobnm = 'run'

  nrsavedruns = 0
  with os.scandir(logdir) as it:
    for entry in logdir.iterdir():
      if entry.name.startswith(jobnm) and entry.is_dir():
        nrsavedruns += 1
  logdir = logdir / Path(jobnm+str(nrsavedruns+1)+'_'+'m'.join( datetime.now().isoformat().split(':')[:-1] ))
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

def getModelsByType( learntype, classification, ndim ):
    predtype = kc.DataPredType.Continuous
    outtype = kc.OutputType.Pixel
    dimtype = kc.DimType(ndim)
    if dgbhdf5.isImg2Img(learntype):
        outtype = kc.OutputType.Image
    if classification or dgbhdf5.isSeisClass( learntype ):
            predtype = kc.DataPredType.Classification
    return kc.UserModel.getNamesByType(pred_type=predtype, out_type=outtype, dim_type=dimtype)

def getDefaultModel(setup,type=keras_dict['type'],
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

  if kc.UserModel.findName(type):
    return kc.UserModel.findName(type).model(model_shape, nroutputs,
                                             learnrate,
                                             data_format=data_format)
  return None


def train(model,training,params=keras_dict,trainfile=None,logdir=None,
          withaugmentation=withaugmentation,tempnm=None):
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
  train_datagen = TrainingSequence( training, False, model, exfilenm=trainfile, batch_size=batchsize, with_augmentation=withaugmentation, tempnm=tempnm )
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
           withconfidence, doprobabilities, scaler=None, batch_size=None ):
  if batch_size == None:
    batch_size = keras_dict['batch']
  redirect_stdout()
  import keras
  restore_stdout()
  ret = {}
           
  inp_shape = samples.shape
  data_format = 'channels_first'
  samples = adaptToModel( model, samples, sample_data_format=data_format )
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
          res = model.predict_classes( samples, batch_size=batch_size )
        except AttributeError:
          pass
    if not isinstance( res, np.ndarray ):
      res = model.predict( samples, batch_size=batch_size )
      res = adaptFromModel(model,res,inp_shape,ret_data_format=data_format)
      ret.update({dgbkeys.preddictstr: res})

  if isclassification and (doprobabilities or withconfidence or withpred):
    if len(ret)<1:
      allprobs = model.predict( samples, batch_size=batch_size )
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


def adaptToModel( model, samples, sample_data_format='channels_first' ):
  nrdims = len( model.input_shape ) - 2
  nrsamples = samples.shape[0]
  samples_nrdims = len(samples.shape)
  model_data_format = get_data_format( model )
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
