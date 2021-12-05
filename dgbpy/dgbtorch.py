from dgbpy.torch_classes import DatasetApply
from sklearn.metrics import accuracy_score, f1_score
import torch, os, json, pickle, joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import PurePosixPath, PureWindowsPath
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
import odpy.hdf5 as odhdf5
from odpy.common import log_msg,  redirect_stdout, restore_stdout, isWin
import dgbpy.torch_classes as tc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch_dict = {
    'epochs': 10,
    'epochdrop': 5,
    'criterion': nn.CrossEntropyLoss(),
    'batch_size': 8,
    'learnrate': 0.0001,
    'type': None
}
platform = (dgbkeys.torchplfnm, 'PyTorch')
cudacores = [ '1', '2', '4', '8', '16', '32', '48', '64', '96', '128', '144', '192', '256', \
              '288',  '384',  '448',  '480',  '512',  '576',  '640',  '768', \
              '896',  '960',  '1024', '1152', '1280', '1344', '1408', '1536', \
              '1664', '1792', '1920', '2048', '2176', '2304', '2432', '2496', \
              '2560', '2688', '2816', '2880', '2944', '3072', '3584', '3840', \
              '4352', '4608', '4992', '5120' ]

def getMLPlatform():
  return platform[0]

def getParams( 
    nntype=torch_dict['type'], 
    learnrate=torch_dict['learnrate'],
    epochs=torch_dict['epochs'],
    epochdrop=torch_dict['epochdrop'],
    batch=torch_dict['batch_size']):
  ret = {
    'type': nntype,
    'learnrate': learnrate,
    'epochs': epochs,
    'epochdrop': epochdrop,
    'batch': batch
  }
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
        ndim = len(shape)-1
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

def load( modelfnm ):
  model = None
  h5file = odhdf5.openFile( modelfnm, 'r' )
  modelpars = json.loads( odhdf5.getAttr(h5file,'model_config') )
  modelgrp = h5file['model']
  savetype = odhdf5.getText( modelgrp, 'type' )
  if savetype == savetypes[0]:
    modfnm = odhdf5.getText( modelgrp, 'path' )
    modfnm = dgbhdf5.translateFnm( modfnm, modelfnm )
    model = joblib.load( modfnm )
  elif savetype == savetypes[1]:
    modeldata = modelgrp['object']
    model = pickle.loads( modeldata[:].tostring() )
  h5file.close()
  return model

savetypes = ( 'joblib', 'pickle' )
defsavetype = savetypes[0]

def save( model, outfnm, save_type=defsavetype ):
  h5file = odhdf5.openFile( outfnm, 'w' )
  odhdf5.setAttr( h5file, 'backend', 'PyTorch' )
  odhdf5.setAttr( h5file, 'torch_version', torch.__version__ )
  odhdf5.setAttr( h5file, 'type', 'Simple Net' )
  odhdf5.setAttr( h5file, 'model_config', json.dumps((str(model)) ))
  modelgrp = h5file.create_group( 'model' )
  odhdf5.setAttr( modelgrp, 'type', save_type )
  if save_type == savetypes[0]:
    joutfnm = os.path.splitext( outfnm )[0] + '.joblib'
    joblib.dump( model.state_dict(), joutfnm )
    odhdf5.setAttr( modelgrp, 'path', joutfnm )
  elif save_type == savetypes[1]:
    exported_modelstr = pickle.dumps(model.state_dict())
    exported_model = np.frombuffer( exported_modelstr, dtype='S1', count=len(exported_modelstr) )
    modelgrp.create_dataset('object',data=exported_model)
  h5file.close()

def train(model, imgdp, params):
    from dgbpy.torch_classes import Trainer
    trainloader, testloader = DataGenerator(imgdp, batchsize=params['batch'])
    criterion = torch_dict['criterion']
    if imgdp[dgbkeys.infodictstr][dgbkeys.classdictstr]==False:
      criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learnrate'])
    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        training_DataLoader=trainloader,
        validation_DataLoader=testloader,
        lr_scheduler=None,
        epochs=params['epochs'],
        earlystopping=params['epochdrop'],
        epoch=0,
        notebook=True,
        imgdp=imgdp
    )
    model, training_losses, validation_losses, training_accs, validation_accs, lr_rates = trainer.run_trainer()
    return model

def apply( model, info, samples, scaler, isclassification, withpred, withprobs, withconfidence, doprobabilities ):
  if scaler != None:
    samples = scaler.transform( samples )
  attribs = dgbhdf5.getNrAttribs(info)
  model_shape = get_model_shape(info[dgbkeys.inpshapedictstr], attribs, True)
  ndims = getModelDims(model_shape, 'channels_first')
  sampleDataset = DatasetApply(samples, isclassification, 1, ndims=ndims)
  dataloader = getDataLoader(sampleDataset, batchsize=torch_dict['batch_size'])
  ret = {}
  res = None
  try:
    img2img = info[dgbkeys.seisimgtoimgtypestr]
    img2img = True
  except KeyError:
    img2img = False
  if isclassification:
    nroutputs = len(info[dgbkeys.classesdictstr])
  else:
    nroutputs = dgbhdf5.getNrOutputs(info)
  from dgbpy.mlmodel_torch_dGB import ResNet18
  dfdm = ResNet18(nroutputs, dim=ndims, nrattribs=attribs)
  if info[dgbkeys.learntypedictstr] == dgbkeys.seisclasstypestr or \
    info[dgbkeys.learntypedictstr] == dgbkeys.loglogtypestr or info[dgbkeys.learntypedictstr] == dgbkeys.seisproptypestr:
    try:
      dfdm.load_state_dict(model)
    except RuntimeError:
      from dgbpy.torch_classes import Net
      dfdm = Net(output_classes=nroutputs, dim=ndims, nrattribs=attribs)
      dfdm.load_state_dict(model)
  elif info[dgbkeys.learntypedictstr] == dgbkeys.seisimgtoimgtypestr:
    from dgbpy.torch_classes import UNet
    if isclassification:
      dfdm = UNet(out_channels=nroutputs, n_blocks=1, dim=ndims)
    else:
      dfdm = UNet(out_channels=1,  n_blocks=1, dim=ndims)
    dfdm.load_state_dict(model)

  predictions = []
  predictions_prob = []
  dfdm.eval()
  for input in dataloader:
      with torch.no_grad():
        if info[dgbkeys.learntypedictstr] == dgbkeys.seisimgtoimgtypestr:
          pass
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
  if info[dgbkeys.learntypedictstr] == dgbkeys.seisimgtoimgtypestr:
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

def getTrainTestDataLoaders(traindataset, testdataset, batchsize=torch_dict['batch_size']):
    trainloader = DataLoader(dataset=traindataset, batch_size=batchsize, shuffle=True, drop_last=True)
    testloader= DataLoader(dataset=testdataset, batch_size=batchsize, shuffle=False, drop_last=True)
    return trainloader, testloader

def getDataLoader(dataset, batchsize=torch_dict['batch_size']):
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=False, drop_last=False)
    return dataloader

def getDataLoaders(traindataset, testdataset, batchsize=torch_dict['batch_size']):
    trainloader = DataLoader(dataset=traindataset, batch_size=batchsize, shuffle=True, drop_last=False)
    testloader= DataLoader(dataset=testdataset, batch_size=batchsize, shuffle=False, drop_last=False)
    return trainloader, testloader

def DataGenerator(imgdp, batchsize):
    info = imgdp[dgbkeys.infodictstr]
    x_train = imgdp[dgbkeys.xtraindictstr]
    y_train = imgdp[dgbkeys.ytraindictstr]
    x_test = imgdp[dgbkeys.xvaliddictstr]
    y_test = imgdp[dgbkeys.yvaliddictstr]
    inp_ch = x_train.shape[1]
    attribs = dgbhdf5.getNrAttribs(info)
    inp_shape = len(x_train)
    out_shape = len(y_train)
    model_shape = get_model_shape(info[dgbkeys.inpshapedictstr], attribs, True)
    ndims = getModelDims(model_shape, True)

    from dgbpy.torch_classes import SeismicTrainDataset, SeismicTestDataset
    train_dataset = SeismicTrainDataset(x_train, y_train, info, inp_ch, ndims)
    test_dataset = SeismicTestDataset(x_test, y_test, info, inp_ch, ndims)

    trainloader, testloader = getDataLoaders(train_dataset, test_dataset, batchsize)
    return trainloader, testloader