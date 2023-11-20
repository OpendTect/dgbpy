#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        O. Ibrahim
# Date:          Sept 2021
#
# _________________________________________________________________________
# various tools machine learning using PyTorch platform
#
import os, json, pickle, joblib
import numpy as np
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
import odpy.hdf5 as odhdf5

try:
  import torch
  import torch.nn as nn
  from torch.utils.data import DataLoader
  import dgbpy.torch_classes as tc
  device = torch.device('cpu')
except ModuleNotFoundError:
  device = None
  pass

def hasTorch():
  try:
    import torch
  except ModuleNotFoundError:
    return False
  return True

platform = (dgbkeys.torchplfnm, 'PyTorch')

withtensorboard = dgbkeys.getDefaultTensorBoard()

default_transforms = []

defbatchstr = 'defaultbatchsz'

torch_infos = None

torch_dict = {
    dgbkeys.decimkeystr: False,
    'nbchunk': 10,
    'epochs': 15,
    'patience': 10,
    'epochdrop': 5,
    'split': 0.2,
    'nbfold': 5,
    'criterion': nn.CrossEntropyLoss() if hasTorch() else None,
    'batch': 8,
    'learnrate': 1e-4,
    'type': None,
    'prefercpu': None,
    'scale': dgbkeys.globalstdtypestr,
    'transform':default_transforms,
    'withtensorboard': withtensorboard,
    'tofp16': True,
}

def getMLPlatform():
  return platform[0]

cudacores = [ '1', '2', '4', '8', '16', '32', '48', '64', '96', '128', '144', '192', '256', \
              '288',  '384',  '448',  '480',  '512',  '576',  '640',  '768', \
              '896',  '960',  '1024', '1152', '1280', '1344', '1408', '1536', \
              '1664', '1792', '1920', '2048', '2176', '2304', '2432', '2496', \
              '2560', '2688', '2816', '2880', '2944', '3072', '3584', '3840', \
              '4352', '4608', '4992', '5120' ]


def can_use_gpu():
  if torch.cuda.is_available():
    return True
  return False

def set_compute_device(prefercpu):
  global device
  if not prefercpu:
    prefercpu = not can_use_gpu()
  device = torch.device('cuda:0' if not prefercpu else 'cpu')

def get_torch_infos():
  global torch_infos
  if torch_infos:
    return torch_infos
  ret = {
    'hastorchgpu': can_use_gpu(),
    dgbkeys.prefercpustr: not can_use_gpu(),
    'batchsizes': cudacores,
    defbatchstr: torch_dict['batch']
  }
  torch_infos = json.dumps( ret )
  return torch_infos

def getParams( 
    nntype=torch_dict['type'],
    dodec = torch_dict[dgbkeys.decimkeystr],
    nbchunk = torch_dict['nbchunk'],
    learnrate=torch_dict['learnrate'],
    epochdrop = torch_dict['epochdrop'],
    epochs=torch_dict['epochs'],
    patience=torch_dict['patience'],
    batch=torch_dict['batch'],
    prefercpu = torch_dict['prefercpu'],
    validation_split=torch_dict['split'],
    nbfold=torch_dict['nbfold'],
    scale=torch_dict['scale'],
    transform=torch_dict['transform'],
    withtensorboard=torch_dict['withtensorboard'],
    tofp16=torch_dict['tofp16']):
  ret = {
    dgbkeys.decimkeystr: dodec,
    'type': nntype,
    'nbchunk': nbchunk,
    'learnrate': learnrate,
    'epochdrop': epochdrop,
    'epochs': epochs,
    'patience': patience,
    'split':validation_split,
    'nbfold':nbfold,
    'batch': batch,
    'scale': scale,
    'transform': transform,
    'withtensorboard': withtensorboard,
    'tofp16': tofp16,
  }
  if prefercpu == None:
    prefercpu = not can_use_gpu()
  ret.update({'prefercpu': prefercpu})
  if not dodec:
    ret['nbchunk'] = 1
  return ret

def getDefaultModel(setup,type=torch_dict['type']):
  isclassification = setup[dgbhdf5.classdictstr]
  inp_shape = setup[dgbkeys.inpshapedictstr]
  attribs = dgbhdf5.getNrAttribs(setup)
  model_shape = get_model_shape(inp_shape, attribs, True)
  if isclassification:
    nroutputs = len(setup[dgbkeys.classesdictstr])
  else:
    nroutputs = dgbhdf5.getNrOutputs( setup )
  if len(tc.TorchUserModel.mlmodels) < 1:
    tc.TorchUserModel.mlmodels = tc.TorchUserModel.findModels()
  if type==None:
      type = getModelsByInfo( setup )
  if tc.TorchUserModel.findName(type):
    return tc.TorchUserModel.findName(type).model(model_shape, nroutputs, attribs)
  return None

def getModelsByType( learntype, classification, ndim ):
    predtype = tc.DataPredType.Continuous
    outtype = tc.OutputType.Pixel
    dimtype = tc.DimType(ndim)
    if dgbhdf5.isImg2Img(learntype):
        outtype = tc.OutputType.Image
    if classification or dgbhdf5.isSeisClass( learntype ):
            predtype = tc.DataPredType.Classification
    return tc.TorchUserModel.getNamesByType(pred_type=predtype, out_type=outtype, dim_type=dimtype)

def getModelsByInfo( infos ):
    shape = infos[dgbkeys.inpshapedictstr]
    if isinstance(shape,int):
        ndim = 1
    else:
        ndim = len(shape) - shape.count(1)
    modelstypes = getModelsByType( infos[dgbkeys.learntypedictstr],
                                   infos[dgbhdf5.classdictstr], 
                                   ndim )                             
    if len(modelstypes) < 1:
        return None
    return modelstypes[0]

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

savetypes = ( 'onnx', 'joblib', 'pickle' )
defsavetype = savetypes[0]

def load( modelfnm ):
  model = None
  h5file = odhdf5.openFile( modelfnm, 'r' )
  modelgrp = h5file['model']
  savetype = odhdf5.getText( modelgrp, 'type' )
  
  modeltype = odhdf5.getText(h5file, 'type')
  if modeltype=='Sequential' or modeltype=='Net':
    savetype = savetypes[1]
  if savetype == savetypes[0]:
    modfnm = odhdf5.getText( modelgrp, 'path' )
    modfnm = dgbhdf5.translateFnm( modfnm, modelfnm )
    from dgbpy.torch_classes import OnnxTorchModel
    model = OnnxTorchModel(str(modfnm))
  elif savetype == savetypes[1]:
    modfnm = odhdf5.getText( modelgrp, 'path' )
    modfnm = dgbhdf5.translateFnm( modfnm, modelfnm )
    model = joblib.load( modfnm )
  elif savetype == savetypes[2]:
    modeldata = modelgrp['object']
    model = pickle.loads( modeldata[:].tostring() )
  h5file.close()
  return model

def onnx_from_torch(model, infos):
  attribs = dgbhdf5.getNrAttribs(infos)
  isclassification = infos[dgbhdf5.classdictstr]
  model_shape = get_model_shape(infos[dgbkeys.inpshapedictstr], attribs, True)
  dims = getModelDims(model_shape, True)
  if isclassification:
    nroutputs = len(infos[dgbkeys.classesdictstr])
    predtype = tc.DataPredType.Classification
  else:
    predtype = tc.DataPredType.Continuous
    nroutputs = dgbhdf5.getNrOutputs( infos )
  input_size = torch_dict['batch']
  if model.__class__.__name__ == 'UNet' or 'UNet_VGG19': 
    input_size = 1
  if dims  == 3:
    dummy_input = torch.randn(input_size, model_shape[0], model_shape[1], model_shape[2], model_shape[3])
  elif dims == 2:
    dummy_input = torch.randn(input_size, model_shape[0], model_shape[1], model_shape[2])
  elif dims == 1 or dims == 0:
    dummy_input = torch.randn(input_size, model_shape[0], model_shape[1])
  if model.__class__.__name__ == 'Sequential':
    from dgbpy.mlmodel_torch_dGB import ResNet18
    model_instance = ResNet18(nroutputs, dims, attribs)
  elif model.__class__.__name__ == 'Net':
    from dgbpy.torch_classes import Net
    model_instance = Net(nroutputs, dims, attribs)
  elif model.__class__.__name__ == "dGBLeNet":
    from dgbpy.torch_classes import dGBLeNet
    model_instance = dGBLeNet(model_shape, nroutputs, dims, attribs, predtype)
  elif model.__class__.__name__ == 'UNet':
    from dgbpy.torch_classes import UNet
    model_instance = UNet(out_channels=nroutputs, dim=dims, in_channels=attribs, n_blocks=model.n_blocks)
  elif model.__class__.__name__ == 'UNet_VGG19':
    from dgbpy.torch_classes import UNet_VGG19
    model_instance = UNet_VGG19(model_shape, nroutputs, attribs)
  elif model.__class__.__name__ == 'GraphModule':
    model_instance = torch.jit.trace(model.cpu(), dummy_input)
  model_instance.load_state_dict(model.state_dict())
  return model_instance, dummy_input

def save( model, outfnm, infos, save_type=defsavetype ):
  h5file = odhdf5.openFile( outfnm, 'w' )
  odhdf5.setAttr( h5file, 'backend', 'PyTorch' )
  odhdf5.setAttr( h5file, 'torch_version', torch.__version__ )
  odhdf5.setAttr( h5file, 'type', model.__class__.__name__ )
  odhdf5.setAttr( h5file, 'model_config', json.dumps((str(model)) ))
  modelgrp = h5file.create_group( 'model' )
  odhdf5.setAttr( modelgrp, 'type', save_type )
  if model.__class__.__name__ == 'Sequential' or model.__class__.__name__ == "Net":
    save_type = savetypes[1]
  if save_type == savetypes[0]:
    joutfnm = os.path.splitext( outfnm )[0] + '.onnx'
    retmodel, dummies = onnx_from_torch(model, infos)
    input_name = ['input']
    dynamic_axes = {'input': {0: 'batch_size'}}
    torch.onnx.export(retmodel, dummies, joutfnm, input_names=input_name, dynamic_axes=dynamic_axes)
    odhdf5.setAttr( modelgrp, 'path', joutfnm )
  elif save_type == savetypes[1]:
    joutfnm = os.path.splitext( outfnm )[0] + '.joblib'
    joblib.dump( model.state_dict(), joutfnm )
    odhdf5.setAttr( modelgrp, 'path', joutfnm )
  elif save_type == savetypes[2]:
    exported_modelstr = pickle.dumps(model.state_dict())
    exported_model = np.frombuffer( exported_modelstr, dtype='S1', count=len(exported_modelstr) )
    modelgrp.create_dataset('object',data=exported_model)
  h5file.close()

def train(model, imgdp, params, cbfn=None, logdir=None, silent=False):
    from dgbpy.torch_classes import Trainer, AdaptiveLR
    trainloader, testloader = DataGenerator(imgdp,batchsize=params['batch'],scaler=params['scale'],transform=params['transform'])
    criterion = params['criterion']
    if imgdp[dgbkeys.infodictstr][dgbkeys.classdictstr]==False:
      criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learnrate'])
    if params['epochdrop'] < 1:
      scheduler = None
    else:
      scheduler = AdaptiveLR(optimizer, params['learnrate'], params['epochdrop'])
    tensorboard = None
    if logdir != None and params['withtensorboard']:
      from torch.utils.tensorboard import SummaryWriter
      tensorboard = SummaryWriter(log_dir=logdir)
    set_compute_device(params['prefercpu'])
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device = device,
        training_DataLoader=trainloader,
        validation_DataLoader=testloader,
        tensorboard=tensorboard,
        epochs=params['epochs'],
        earlystopping = params['patience'],
        imgdp=imgdp,
        silent = silent,
        tofp16=params['tofp16']
    )
    model = trainer.fit(cbs = cbfn)
    return model

def transfer(model):
  """
    Transfer learning utility function for fine-tuning a Torch model.

    This function takes a Torch model and prepares it for transfer learning by selectively
    setting layers to be trainable. The layers to be made trainable are determined as follows:

    1. All layers before the first Conv1D, Conv2D, or Conv3D layer (or a Sequential containing such layers)
       are set to trainable.

    2. All layers after the last Conv1D, Conv2D, Conv3D, or Dense layer (or a Sequential containing
       such layers) are set to trainable.

    3. All layers between the first and last Conv1D, Conv2D, Conv3D, or Dense layer (or a Sequential
        containing such layers) are set to non-trainable.
  """
  def getTrainableLayers(model):
    first_lyr = None
    last_lyr = None

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if first_lyr is None:
                first_lyr = name
            last_lyr = name
        elif isinstance(layer, nn.Linear) and first_lyr:
          last_lyr = name
    return first_lyr, last_lyr

  from dgbpy.torch_classes import OnnxTorchModel
  if isinstance(model, OnnxTorchModel):
    model = model.convert_to_torch()

  first_conv_layer, last_layer = getTrainableLayers(model)
  found_first, found_last = False, False

  for name, param in model.named_parameters():
    if first_conv_layer and name.startswith(first_conv_layer):
      param.requires_grad = True
      found_first = True
    elif last_layer and name.startswith(last_layer):
      param.requires_grad = True
      found_last = True
    elif found_first and not found_last:
      param.requires_grad = False
    else:
      param.requires_grad = True
  return model

def apply( model, info, samples, scaler, isclassification, withpred, withprobs, withconfidence, doprobabilities ):
  attribs = dgbhdf5.getNrAttribs(info)
  model_shape = get_model_shape(info[dgbkeys.inpshapedictstr], attribs, True)
  ndims = getModelDims(model_shape, 'channels_first')
  sampleDataset = tc.DatasetApply(samples, info, isclassification, 1, ndims=ndims)
  ret = {}
  res = None
  try:
    img2img = info[dgbkeys.seisimgtoimgtypestr]
    img2img = True
  except KeyError:
    img2img = False
  
  if info[dgbkeys.learntypedictstr] == dgbkeys.seisclasstypestr or \
      info[dgbkeys.learntypedictstr] == dgbkeys.loglogtypestr:
      drop_last = True
  else:
    drop_last = False
  batch_size = torch_dict['batch']
  dataloader = getDataLoader(sampleDataset, batch_size=batch_size, drop_last=drop_last)
  if isclassification:
    nroutputs = len(info[dgbkeys.classesdictstr])
    predtype = tc.DataPredType.Classification
  else:
    nroutputs = dgbhdf5.getNrOutputs(info)
    predtype = tc.DataPredType.Continuous
  
  if model.__class__.__name__ == "OrderedDict":
    from dgbpy.mlmodel_torch_dGB import ResNet18
    from dgbpy.torch_classes import dGBLeNet, Net
    seis_log_arch = [ResNet18, dGBLeNet, Net]
    if info[dgbkeys.learntypedictstr] == dgbkeys.seisclasstypestr or \
      info[dgbkeys.learntypedictstr] == dgbkeys.loglogtypestr or info[dgbkeys.learntypedictstr] == dgbkeys.seisproptypestr:
      for arch in seis_log_arch:
        try:
          if arch.__name__ == 'dGBLeNet': dfdm = arch(nroutputs, dim=ndims, nrattribs=attribs, predtype=predtype)
          else: dfdm = arch(nroutputs, dim=ndims, nrattribs=attribs)
          dfdm.load_state_dict(model)
        except RuntimeError: pass
    elif info[dgbkeys.learntypedictstr] == dgbkeys.seisimgtoimgtypestr:
      from dgbpy.torch_classes import UNet
      if isclassification:
        dfdm = UNet(out_channels=nroutputs, n_blocks=1, dim=ndims)
      else:
        dfdm = UNet(out_channels=1,  n_blocks=1, dim=ndims)
      dfdm.load_state_dict(model)
  elif model.__class__.__name__ == 'OnnxTorchModel':
    dfdm = model

  predictions = []
  predictions_prob = []
  dfdm.eval()
  for input in dataloader:
      with torch.no_grad():
        out = dfdm(input)
        pred = out.detach().numpy()
        pred_prob = pred.copy()
        if isclassification:
          pred = np.argmax(pred, axis=1)
        for _ in pred:
          predictions.append(_)
        for _prob in pred_prob:
          predictions_prob.append(_prob)

  if withpred:
    if isclassification:
      if not (doprobabilities or withconfidence):
        try:
          res = np.array(predictions_prob)
          res = np.transpose(np.array(predictions))
        except AttributeError:
          pass
    
  if isclassification and (doprobabilities or withconfidence or withpred):
    if len(ret)<1:
      allprobs = (np.array(predictions)).transpose()
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
      N = 2
      predictions_prob = np.array(predictions_prob)
      x = predictions_prob.shape[0]
      indices = np.argpartition(predictions_prob.transpose(),-N,axis=0)[-N:].transpose()
      sortedprobs = predictions_prob.transpose()[indices.ravel(),np.tile(np.arange(x),N)].reshape(N,x)
      res = np.diff(sortedprobs,axis=0)
      ret.update({dgbkeys.confdictstr: res})
  
  if withpred:
    res = np.transpose(np.array(predictions))
    ret.update({dgbkeys.preddictstr: res})
  
  if doprobabilities:
    res = np.transpose( np.array(predictions_prob) )
    ret.update({dgbkeys.probadictstr: res})
  if info[dgbkeys.learntypedictstr] == dgbkeys.seisimgtoimgtypestr and withpred:
    if ndims==3:
      if isclassification:
        ret[dgbkeys.preddictstr] = ret[dgbkeys.preddictstr].transpose(3, 2, 1, 0)
      else:
        ret[dgbkeys.preddictstr] = ret[dgbkeys.preddictstr].transpose(4, 3, 2, 1, 0)
    elif ndims==2:
      if isclassification:
        ret[dgbkeys.preddictstr] = ret[dgbkeys.preddictstr].transpose(2, 1, 0)
      else:
        ret[dgbkeys.preddictstr] = ret[dgbkeys.preddictstr].transpose(3, 2, 1, 0)  
  return ret

def getDataLoader(dataset, batch_size=torch_dict['batch'], drop_last=False):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)
    return dataloader

class ChunkedDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def set_chunk(self, ichunk):
        return self.dataset.set_chunk(ichunk)

    def set_fold(self, ichunk, ifold):
        return self.dataset.set_fold(ichunk, ifold)

    def set_transform_seed(self):
        self.dataset.set_transform_seed()

    def get_batchsize(self):
        return self.batch_size

    def __iter__(self):
        for batch in super().__iter__():
            yield batch

def getDataLoaders(traindataset, testdataset, batchsize=torch_dict['batch']):
    trainloader = ChunkedDataLoader(dataset=traindataset, batch_size=batchsize, shuffle=False, drop_last=True)
    testloader= ChunkedDataLoader(dataset=testdataset, batch_size=batchsize, shuffle=False, drop_last=True)
    return trainloader, testloader

def getDatasetPars(imgdp, _forvalid):
    info = imgdp[dgbkeys.infodictstr]
    if _forvalid:
      x_data = imgdp[dgbkeys.xvaliddictstr]
      y_data = imgdp[dgbkeys.yvaliddictstr]
    else:
      x_data = imgdp[dgbkeys.xtraindictstr]
      y_data = imgdp[dgbkeys.ytraindictstr]
    inp_ch = x_data.shape[1]
    attribs = dgbhdf5.getNrAttribs(info)
    model_shape = get_model_shape(info[dgbkeys.inpshapedictstr], attribs, True)
    ndims = getModelDims(model_shape, True)
    return x_data, y_data, info, inp_ch, ndims

def DataGenerator(imgdp, batchsize, scaler=None, transform=list()):
    from dgbpy.torch_classes import TrainDatasetClass, TestDatasetClass
    train_dataset = TrainDatasetClass(imgdp, scaler, transform=transform)
    test_dataset = TestDatasetClass(imgdp, scaler)

    trainloader, testloader = getDataLoaders(train_dataset, test_dataset, batchsize)
    return trainloader, testloader
