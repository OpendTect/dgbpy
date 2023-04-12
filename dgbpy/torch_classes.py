#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        Olawale I.
# Date:          Sept 2021
#
# _________________________________________________________________________
# various tools machine learning using PyTorch platform
#

import re
import time
from functools import partial
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import Linear, ReLU, Sequential, Conv1d, Conv2d, Conv3d, Dropout, Dropout2d, Dropout3d
from torch.nn import MaxPool1d, MaxPool2d, MaxPool3d, Softmax, BatchNorm1d, BatchNorm2d, BatchNorm3d
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
import odpy.common as odcommon
from dgbpy.mlio import announceShowTensorboard, announceTrainingFailure, announceTrainingSuccess

import onnxruntime as rt
def Tensor2Numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def Numpy2tensor(nparray):
    return torch.from_numpy(nparray)

def hasFastprogress():
    try:
        import fastprogress
    except ModuleNotFoundError:
        return False
    return True

if hasFastprogress():
    from fastprogress.fastprogress import master_bar, progress_bar

class OnnxModel():
    def __init__(self, filepath : str):
        self.name = filepath

    def __call__(self, inputs):
        self.inputs = inputs
        ort_session = rt.InferenceSession(self.name)
        ort_inputs = {ort_session.get_inputs()[0].name: Tensor2Numpy(self.inputs)}
        ort_outs = np.array(ort_session.run(None, ort_inputs))[-1]
        return Numpy2tensor(ort_outs)

    def eval(self):
        pass

class Net(nn.Module):   
    def __init__(self, model_shape, output_classes, dim, nrattribs):
        super(Net, self).__init__()
        
        self.output_classes = output_classes
        self.dim, self.nrattribs = dim, nrattribs
        self.model_shape, self.pool_padding = model_shape, 1
        if output_classes==1:
            self.activation = ReLU()
        else:
            self.activation = Softmax(dim=1)
        if dim==3:
            BatchNorm = BatchNorm3d
            Conv = Conv3d
            MaxPool = MaxPool3d
        elif dim==2:
            BatchNorm = BatchNorm2d
            Conv = Conv2d
            MaxPool = MaxPool2d
        elif dim==1:
            BatchNorm = BatchNorm1d
            Conv = Conv1d
            MaxPool = MaxPool1d
        elif dim==0:
            BatchNorm = BatchNorm1d
            Conv = Conv1d
            MaxPool = MaxPool1d
            self.padding = 0

        self.cnn_layers = Sequential(
            Conv(nrattribs, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm(4),
            ReLU(inplace=True),
            MaxPool(kernel_size=2, stride=2, padding=self.pool_padding),
        )

        self.after_cnn_size = self.after_cnn(torch.randn(self.model_shape).unsqueeze(0))
        self.linear_layers = Sequential(
            Linear(self.after_cnn_size, self.output_classes),
            self.activation,
        )

    def after_cnn(self, x):
        x = self.cnn_layers[0](x)
        x = self.cnn_layers[-1](x)
        return int(np.prod(x.size()[1:]))

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x   

class dGBLeNet(nn.Module):   
    def __init__(self, model_shape, output_classes, dim, nrattribs, predtype):        
        super(dGBLeNet, self).__init__()
        
        self.output_classes = output_classes
        self.dim, self.nrattribs = dim, nrattribs
        self.model_shape, self.pool_padding = model_shape, 1

        if output_classes==1: self.activation = ReLU()
        else: self.activation = Softmax(dim=1)

        if dim==3:
            BatchNorm = BatchNorm3d
            DropOut = Dropout3d
            Conv = Conv3d
        elif dim==2:
            BatchNorm = BatchNorm2d
            DropOut = Dropout2d
            Conv = Conv2d
        elif dim==1 or dim == 0:
            BatchNorm = BatchNorm1d
            DropOut = Dropout
            Conv = Conv1d

        filtersz = 50
        densesz = 10  
        kernel_sz1, kernel_sz2 = 5, 3
        stride_sz1, stride_sz2 = 4, 2
        dropout = 0.2

        self.cnn_1 = Sequential(
          Conv(nrattribs, filtersz, kernel_size=3, stride=stride_sz1, padding=1),
          BatchNorm(filtersz),
          ReLU(inplace=True))

        self.cnn_2 = Sequential(
          Conv(filtersz, filtersz, kernel_size=kernel_sz2, stride=stride_sz2, padding=1),
          DropOut(p=dropout, inplace=True),
          BatchNorm(filtersz),
          ReLU(inplace=True))

        self.cnn_3 = Sequential(
          Conv(filtersz, filtersz, kernel_size=kernel_sz2, stride=stride_sz2, padding=1),
          DropOut(p=dropout, inplace=True),
          BatchNorm(filtersz),
          ReLU(inplace=True))

        self.cnn_4 = Sequential(
          Conv(filtersz, filtersz, kernel_size=kernel_sz2, stride=stride_sz2, padding=1),
          DropOut(p=dropout, inplace=True),
          BatchNorm(filtersz),
          ReLU(inplace=True))
        
        self.cnn_5 = Conv(filtersz, filtersz, kernel_size=kernel_sz2, stride=stride_sz2, padding=1)
        
        self.after_cnn_size = self.after_cnn(torch.randn(self.model_shape).unsqueeze(0))
        self.linear = Sequential(
          Linear(self.after_cnn_size, filtersz),
          BatchNorm1d(filtersz),
          ReLU(inplace=True),
          Linear(filtersz, densesz),
          BatchNorm1d(densesz),
          ReLU(inplace=True))

        self.final = None
        if isinstance(predtype, DataPredType) and predtype==DataPredType.Continuous:
          self.final = Linear(densesz, self.output_classes)
        else:
          self.final = Sequential(
            Linear(densesz, self.output_classes),
            BatchNorm1d(self.output_classes),
            Softmax(dim=1)
          )

    def after_cnn(self, x):
        x = self.cnn_1[0](x)
        x = self.cnn_2[0](x)
        x = self.cnn_3[0](x)
        x = self.cnn_4[0](x)
        x = self.cnn_5(x)
        return int(np.prod(x.size()[1:]))

    def forward(self, x):
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        x = self.cnn_4(x)
        x = self.cnn_5(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.final(x)
        return x   

def flatten(out, target):
    out = out.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    return out,target

def accuracy(out, target):
    pred, target = flatten(out.detach(), target)
    return accuracy_score(pred, target)

def f1(out, target):
    pred, target = flatten(out.detach(), target)
    return f1_score(pred, target, average='weighted')

def mae(out, target):
    pred, target = flatten(out.detach(), target)
    return mean_absolute_error(pred, target)

def reformat_str(name):
    _camel_re1 = re.compile('(.)([A-Z][a-z]+)')
    _camel_re2 = re.compile('([a-z0-9])([A-Z])')
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

class Callback():
    _order=0
    def set_runner(self, run): self.run=run
    def __getattr__(self, k): return getattr(self.run, k)
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return reformat_str(name or 'callback')
    
    def __call__(self, cb_name):
        fnc = getattr(self, cb_name, None)
        if fnc and fnc(): return True
        return False

class TrainEvalCallback(Callback):
    _order = 1
    def begin_fit(self):
        self.run.model = self.run.model.to(self.run.device)

    def begin_batch(self):
        if self.classification: self.run.target = self.target.type(torch.LongTensor)
        self.run.input = self.run.input.to(self.run.device)
        self.run.target  = self.run.target.to(self.run.device)
        
    def begin_epoch(self):
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False

class AvgStats():
    def __init__(self, metrics): self.metrics = dgbkeys.listify(metrics)
    
    def reset(self):
        self.tot_loss,self.count = 0.,0
        self.tot_mets = [0.] * len(self.metrics)
        
    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]

    def accumulate(self, run):
        bn = run.input.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.out, run.target) * bn

class AvgStatsCallback(Callback):
    _order = 2
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = AvgStats(metrics),AvgStats(metrics)
    
    def begin_fit(self):
        met_names = ['loss'] + [m.__name__ for m in self.train_stats.metrics]
        self.names = ['Epoch']
        for n in met_names: self.names += [f'Train_{n}'] + [f'Valid_{n}'] 
        self.names += ['time']
        if not self.silent: self.logger(self.names)
    
    def begin_epoch(self):  
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()
        
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    
    def after_epoch(self):
        stats = [str(self.epoch+1)] 
        for tr,vl in zip(self.train_stats.avg_stats, self.valid_stats.avg_stats):
            stats += [f'{tr:.4f}', f'{vl:.4f}'] 
        stats += [dgbkeys.format_time(time.time() - self.start_time)]
        if self.silent:
            for n,(name,stat) in enumerate(zip(self.names,stats)):
                if n == 0: self.logger(f'----------------- Epoch {stat} ------------------')
                else: self.logger(f'{name}: {stat}')
        else: self.logger(stats)

class ProgressBarCallback(Callback):
    _order=-1

    def begin_fit(self):
        self.mbar = master_bar(range(self.run.epochs))
        self.run.logger = partial(self.mbar.write, table=True)
        
    def after_fit(self): self.mbar.on_iter_end()
    def after_batch(self): self.pb.update(self.iter)
    def begin_epoch   (self): self.set_pb()
    def begin_validate(self): self.set_pb()
        
    def set_pb(self):
        self.pb = progress_bar(self.dl, parent=self.mbar)
        self.mbar.update(self.epoch)

class BokehProgressCallback(Callback):
    """Send progress message to bokeh"""
    _order = -1
    def begin_batch(self):
        if self.iter==0:
            odcommon.restore_stdout()
            print('--Iter '+str(self.iter)+' of '+str(self.iters)+' --', flush=True)
            odcommon.restore_stdout()

    def begin_epoch(self):
        if self.epoch==0:
            odcommon.restore_stdout()
            print('--Epoch '+str(self.epoch)+' of '+str(self.epochs)+' --', flush=True)
            odcommon.restore_stdout()

    def after_batch(self):
        odcommon.restore_stdout()
        print('--Iter '+str(self.iter+1)+' of '+str(self.iters)+' --', flush=True)
        odcommon.restore_stdout()

    def after_epoch(self):
        odcommon.restore_stdout()
        print('--Epoch '+str(self.epoch+1)+' of '+str(self.epochs)+' --', flush=True)
        odcommon.restore_stdout()

    def after_fit(self):
        odcommon.restore_stdout()
        print('--Training Ended--', flush=True)
        odcommon.restore_stdout()

    def before_fit_chunk(self):
        odcommon.restore_stdout()
        print('--Chunk_Number '+str(self.ichunk+1)+' of '+str(self.nbchunks)+' --', flush=True)
        odcommon.restore_stdout()

    def begin_fold(self):
        if self.run.isCrossVal:
            odcommon.restore_stdout()
            print('--Fold_bkh '+str(self.ifold)+' of '+str(self.nbfolds)+' --', flush=True)
            odcommon.restore_stdout()

class EarlyStoppingCallback(Callback):
    _order = 3
    def __init__(self, patience):
        self.patience = patience
        self.patience_cnt = 0

    def le_gr(self):
        """
        Return the right ( > ) or ( < ) operator.  
        """
        if self.run.classification:
            return lambda x,y:x>y
        return lambda x,y:x<y

    def begin_fit(self):
        self.earlystop_operator = self.le_gr()

    def after_epoch(self):
        if self.epoch == 0:
            self.best = self.avg_stats.valid_stats.avg_stats[1]
            self.best_epoch = self.epoch
            return 
        if self.patience_cnt < self.patience:
            self.patience_cnt += 1
            if self.earlystop_operator(self.avg_stats.valid_stats.avg_stats[1], self.best):
                self.best = self.avg_stats.valid_stats.avg_stats[1]
                self.best_epoch = self.epoch
                self.run.savemodel = self.model
                self.patience_cnt = 0
        else: raise CancelTrainException()

    def after_fit(self):
        try:
            odcommon.log_msg(f'Best validation accuracy at epoch {self.best_epoch+1} with validation metric score: {self.best:.4f}')
        except: 
            pass

class TensorBoardLogCallback(Callback):
    _order = 10
    def begin_batch(self):
        if self.run.tensorboard and self.epoch+1==self.epochs and self.iter==0:
            self.run.tensorboard.add_graph(self.model, self.input)

    def after_epoch(self):
        if self.run.tensorboard:
            if self.epoch==0: announceShowTensorboard()
            self.run.tensorboard.add_scalars("Loss", 
                {   
                    "training":self.avg_stats.train_stats.avg_stats[0], 
                    "validation":self.avg_stats.valid_stats.avg_stats[0] if self.train_dl else np.nan
                }, self.epoch) 
            self.run.tensorboard.add_scalars(self.avg_stats.train_stats.metrics[0].__name__,
                { 
                    "training":self.avg_stats.train_stats.avg_stats[1],
                    "validation":self.avg_stats.valid_stats.avg_stats[1] if self.valid_dl else np.nan
                }, self.epoch)

class LogNrOfSamplesCallback(Callback):
    _order=0
    def begin_fit(self):
        batchsize = self.train_dl.get_batchsize()
        if batchsize == 1:
            odcommon.log_msg( 'Training on', len(self.run.train_dl), 'samples' )
            odcommon.log_msg( 'Validate on', len(self.run.valid_dl), 'samples' )
        else:
            odcommon.log_msg( 'Training on', len(self.run.train_dl), 'batches of', batchsize, 'samples' )
            odcommon.log_msg( 'Validate on', len(self.run.valid_dl), 'batches of', batchsize, 'samples' )

    def begin_fold(self):
        if self.run.isCrossVal:
            odcommon.log_msg(f'----------------- Fold {self.run.ifold}/{self.run.nbfolds} ------------------')

class TransformCallback(Callback):
    def begin_epoch(self):
        # set new transform seed for each epoch
        self.run.train_dl.set_transform_seed() 

class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 tensorboard = None,
                 epochs: int = 100,
                 earlystopping: int = 5,
                 imgdp = None,
                 cbs = None,
                 silent = None,
                 tofp16 = False
                 ):

        self.model, self.criterion, self.optimizer = model, criterion, optimizer
        self.imgdp, self.train_dl, self.valid_dl = imgdp, training_DataLoader, validation_DataLoader
        self.epochs, self.device, self.savemodel = epochs, device, model
        self.tensorboard, self.silent, self.earlystopping = tensorboard, silent, earlystopping

        self.classification = self.imgdp[dgbkeys.infodictstr][dgbkeys.classdictstr]
        if self.classification: self.metrics = [accuracy, f1]
        else: self.metrics = [mae]
            
        self.tofp16 = tofp16 and torch.cuda.is_available()
        self.gradScaler = torch.cuda.amp.GradScaler() if self.tofp16 else None
        self.in_train, self.logger = False, odcommon.log_msg

    def init_callbacks(self, cbs):
        self.cbs = []
        defaultCBS = [ TrainEvalCallback(), AvgStatsCallback(self.metrics),
                        EarlyStoppingCallback(self.earlystopping), LogNrOfSamplesCallback(),
                        TransformCallback() ]
        if self.tensorboard: defaultCBS.append( TensorBoardLogCallback())
        if hasFastprogress() and not self.silent: defaultCBS.append(ProgressBarCallback())
        self.add_cbs(defaultCBS)
        self.add_cbs(cbs)

    def add_cbs(self, cbs):
        for cb in dgbkeys.listify(cbs):
            cb.set_runner(self)
            setattr(self, cb.name, cb)
            self.cbs.append(cb)

    def remove_cbs(self, cbs):
        for cb in dgbkeys.listify(cbs): self.cbs.remove(cb)

    def compute_loss_func(self):
        if self.classification:
            self.loss = self.criterion(self.out, self.target.squeeze(1))
            self.out = torch.argmax(self.out, axis=1)
        else:
            self.loss = self.criterion(self.out, self.target)

    def one_batch(self, i, input, target):
        try:
            self.iter = i
            self.input, self.target = input, target 
            self('begin_batch')
            if self.tofp16:
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    self.out = self.model(self.input)
                    self('after_pred')
                    self.compute_loss_func()
                    self('after_loss')
                    if not self.in_train: return
                self.gradScaler.scale(self.loss).backward()
                self('after_backward')
                self.gradScaler.step(self.optimizer) 
                self('after_step')
                self.gradScaler.update()
            else:
                self.out = self.model(self.input) 
                self('after_pred')
                self.compute_loss_func() 
                self('after_loss')
                if not self.in_train: return
                self.loss.backward()
                self('after_backward')
                self.optimizer.step() 
                self('after_step')
            self.optimizer.zero_grad()
        except CancelBatchException: self('after_cancel_batch')
        finally: self('after_batch')

    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for i, (input, target) in enumerate(self.dl): self.one_batch(i, input, target)
        except CancelEpochException: self('after_cancel_epoch')

    def do_begin_epoch(self, epoch):
        self.epoch,self.dl = epoch,self.train_dl
        return self('begin_epoch')

    def fit_one_chunk(self, ichunk, cbs):
        self.isCrossVal = dgbhdf5.isCrossValidation(self.imgdp[dgbkeys.infodictstr])
        if not self.isCrossVal:
            return self.train_fn()
        else:
            from dgbpy.dgbtorch import transfer
            self.nbfolds = len(self.imgdp[dgbkeys.infodictstr][dgbkeys.trainseldicstr][ichunk])
            for ifold in range(1, self.nbfolds+1):
                self.ifold = ifold
                if ifold>1: self.init_callbacks(cbs)
                self('begin_fold')
                self.train_dl.set_fold(ichunk, ifold)
                self.valid_dl.set_fold(ichunk, ifold)
                if ifold!=1: # start transfer from second fold
                    transfer(self.savemodel)
                self.savemodel = self.train_fn()
            return self.savemodel

    def train_fn(self):
        try:
            self('begin_fit')
            for epoch in range(self.epochs):
                if not self.do_begin_epoch(epoch): self.all_batches()
                if self.valid_dl:
                    with torch.no_grad():
                        self.dl = self.valid_dl
                        if not self('begin_validate'): self.all_batches()
                self('after_epoch')
            return self.savemodel
        except CancelTrainException:
            self('after_cancel_train')
            return self.savemodel
        finally:
            self('after_fit')

    def fit(self, cbs=None):
        self.nbchunks = len(self.imgdp[dgbkeys.infodictstr][dgbkeys.trainseldicstr])
        for ichunk in range(self.nbchunks):
            self.init_callbacks(cbs)
            self.ichunk = ichunk
            odcommon.log_msg('Starting training iteration',str(ichunk+1)+'/'+str(self.nbchunks))
            try:
                self('before_fit_chunk')
                if not self.train_dl.set_chunk(ichunk) or not self.valid_dl.set_chunk(ichunk):
                    continue
            except Exception as e:
                odcommon.log_msg('')
                odcommon.log_msg('Data loading failed because of insufficient memory')
                odcommon.log_msg('Try to lower the batch size and restart the training')
                odcommon.log_msg('')
                announceTrainingFailure()
                raise e

            if  len(self.train_dl.dataset) < 1 or len(self.valid_dl.dataset) < 1:
                odcommon.log_msg('')
                odcommon.log_msg('There is not enough data to train on')
                odcommon.log_msg('Extract more data and restart')
                odcommon.log_msg('')
                announceTrainingFailure()
                raise 
            
            self.savemodel = self.fit_one_chunk(ichunk, cbs)
        announceTrainingSuccess()
        return self.savemodel
    
    ALL_CBS = { 'begin_batch', 'after_pred', 'after_loss', 'after_backward', 'after_step',
                'after_cancel_batch', 'after_batch', 'after_cancel_epoch', 'begin_fit',
                'begin_epoch', 'begin_validate', 'after_epoch', 'after_cancel_train', 
                'after_fit', 'before_fit_chunk', 'begin_fold'}
            
    def __call__(self, cb_name):
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) and res
        return res

########### 3D RESNET 18 ARCHITECTURE START #############

import torch.nn.functional as F
class ResidualBlock(nn.Module):
    '''
    Residual Block within a ResNet CNN model
    '''
    def __init__(self, input_channels, num_channels,
                 use_1x1_conv = False, strides = 1, ndims=3):
        # super(ResidualBlock, self).__init__()
        super().__init__()
        self.ndims = ndims

        if self.ndims==3:
            Conv = Conv3d
            BatchNorm = BatchNorm3d
            MaxPool = MaxPool3d
        elif self.ndims==2:
            Conv = Conv2d
            BatchNorm = BatchNorm2d
            MaxPool = MaxPool2d
        elif self.ndims==1 or self.ndims==0:
            Conv = Conv1d
            BatchNorm = BatchNorm1d
            MaxPool = MaxPool1d

        self.conv1 = Conv(
            in_channels = input_channels, out_channels = num_channels,
            kernel_size = 3, padding = 1, stride = strides,
            bias = False
            )
        self.conv2 = Conv(
            in_channels = num_channels, out_channels = num_channels,
            kernel_size = 3, padding = 1, stride = 1,
            bias = False
            )
        
        if use_1x1_conv:
            self.conv3 = Conv(
                in_channels = input_channels, out_channels = num_channels,
                kernel_size = 1, stride = strides
                )
        else:
            self.conv3 = None
        
        self.bn1 = BatchNorm(num_features = num_channels)
        self.bn2 = BatchNorm(num_features = num_channels)
        self.relu = nn.ReLU(inplace = True)

        self.initialize_weights()
        
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        
        if self.conv3:
            X = self.conv3(X)
        
        Y += X
        return F.relu(Y)
    
    def shape_computation(self, X):
        Y = self.conv1(X)
        Y = self.conv2(Y)
        
        if self.conv3:
            h = self.conv3(X)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


def create_resnet_block(input_filters, output_filters, num_residuals, ndims, first_block = False):
    # Python list to hold the created ResNet blocks-
    resnet_blk = []
    
    for i in range(num_residuals):
        if i == 0 and first_block:
            resnet_blk.append(ResidualBlock(input_channels = input_filters, num_channels = output_filters, use_1x1_conv = True, strides = 2, ndims=ndims))
        else:
            resnet_blk.append(ResidualBlock(input_channels = output_filters, num_channels = output_filters, use_1x1_conv = False, strides = 1, ndims=ndims))
    
    return resnet_blk

@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging (concatenation) between levels/blocks is possible.
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:  # 2D
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                            ]
        elif encoder_layer.dim() == 5:  # 3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
                            ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
                            ]
    return encoder_layer, decoder_layer


def conv_layer(dim: int):
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d

def get_conv_layer(in_channels: int,
                   out_channels: int,
                   kernel_size: int = 3,
                   stride: int = 1,
                   padding: int = 1,
                   bias: bool = True,
                   dim: int = 2):
    return conv_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias)

def conv_transpose_layer(dim: int):
    if dim == 3:
        return nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d

def get_up_layer(in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: int = 2,
                 dim: int = 3,
                 up_mode: str = 'transposed',
                 ):
    if up_mode == 'transposed':
        return conv_transpose_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    else:
        return nn.Upsample(scale_factor=2.0, mode=up_mode)

def maxpool_layer(dim: int):
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d

def get_maxpool_layer(kernel_size: int = 2,
                      stride: int = 2,
                      padding: int = 0,
                      dim: int = 2):
    return maxpool_layer(dim=dim)(kernel_size=kernel_size, stride=stride, padding=padding)

def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()

def get_normalization(normalization: str,
                      num_channels: int,
                      dim: int):
    if normalization == 'batch':
        if dim == 3:
            return nn.BatchNorm3d(num_channels)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels)
    elif normalization == 'instance':
        if dim == 3:
            return nn.InstanceNorm3d(num_channels)
        elif dim == 2:
            return nn.InstanceNorm2d(num_channels)
    elif 'group' in normalization:
        num_groups = int(normalization.partition('group')[-1]) 
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)

        return x

class DownBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 MaxPool.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pooling: bool = True,
                 activation: str = 'relu',
                 normalization: str = None,
                 dim: str = 2,
                 conv_mode: str = 'same'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.dim = dim
        self.activation = activation

        # conv layers
        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)

        # pooling layer
        if self.pooling:
            self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=0, dim=self.dim)

        # activation layers
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)

    def forward(self, x):
        y = self.conv1(x)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # activation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2

        before_pooling = y  # save the outputs before the pooling operation
        if self.pooling:
            y = self.pool(y)  # pooling
        return y, before_pooling

class UpBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 UpConvolution/Upsample.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: str = 'relu',
                 normalization: str = None,
                 dim: int = 3,
                 conv_mode: str = 'same',
                 up_mode: str = 'transposed'
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.dim = dim
        self.activation = activation
        self.up_mode = up_mode

        # upconvolution/upsample layer
        self.up = get_up_layer(self.in_channels, self.out_channels, kernel_size=2, stride=2, dim=self.dim,
                               up_mode=self.up_mode)

        # conv layers
        self.conv0 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0,
                                    bias=True, dim=self.dim)
        self.conv1 = get_conv_layer(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1,
                                    padding=self.padding,
                                    bias=True, dim=self.dim)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)

        # activation layers
        self.act0 = get_activation(self.activation)
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm0 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)

        # concatenate layer
        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):
        """ Forward pass
        Arguments:
            encoder_layer: Tensor from the encoder pathway
            decoder_layer: Tensor from the decoder pathway (to be up'd)
        """
        up_layer = self.up(decoder_layer)  # up-convolution/up-sampling
        cropped_encoder_layer, dec_layer = autocrop(encoder_layer, up_layer)  # cropping

        if self.up_mode != 'transposed':
            # We need to reduce the channel dimension with a conv layer
            up_layer = self.conv0(up_layer)  # convolution 0
        up_layer = self.act0(up_layer)  # activation 0
        if self.normalization:
            up_layer = self.norm0(up_layer)  # normalization 0

        merged_layer = self.concat(up_layer, cropped_encoder_layer)  # concatenation
        y = self.conv1(merged_layer)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # acivation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2
        return y

class UNet(nn.Module):
    """
    activation: 'relu', 'leaky', 'elu'
    normalization: 'batch', 'instance', 'group{group_size}'
    conv_mode: 'same', 'valid'
    dim: 2, 3
    up_mode: 'transposed', 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 n_blocks: int = 1,
                 start_filters: int = 32,
                 activation: str = 'relu',
                 normalization: str = 'batch',
                 conv_mode: str = 'same',
                 dim: int = 2,
                 up_mode: str = 'transposed'
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filters = start_filters
        self.activation = activation
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.dim = dim
        self.up_mode = up_mode

        self.down_blocks = []
        self.up_blocks = []

        # create encoder path
        for i in range(self.n_blocks):
            num_filters_in = self.in_channels if i == 0 else num_filters_out
            num_filters_out = self.start_filters * (2 ** i)
            pooling = True if i < self.n_blocks - 1 else False

            down_block = DownBlock(in_channels=num_filters_in,
                                   out_channels=num_filters_out,
                                   pooling=pooling,
                                   activation=self.activation,
                                   normalization=self.normalization,
                                   conv_mode=self.conv_mode,
                                   dim=self.dim)

            self.down_blocks.append(down_block)

        # create decoder path (requires only n_blocks-1 blocks)
        for i in range(n_blocks - 1):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            up_block = UpBlock(in_channels=num_filters_in,
                               out_channels=num_filters_out,
                               activation=self.activation,
                               normalization=self.normalization,
                               conv_mode=self.conv_mode,
                               dim=self.dim,
                               up_mode=self.up_mode)

            self.up_blocks.append(up_block)

        # final convolution
        self.conv_final = get_conv_layer(num_filters_out, self.out_channels, kernel_size=1, stride=1, padding=0,
                                         bias=True, dim=self.dim)

        # add the list of modules to current module
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        # initialize the weights
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def forward(self, x: torch.tensor):
        encoder_output = []

        # Encoder pathway
        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Decoder pathway
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)

        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'


class SeismicTrainDataset(Dataset):
    def __init__(self, imgdp, scale, transform=list(), transform_copy = False):
        from dgbpy import transforms as T
        self.imgdp = imgdp
        self._data_IDs = []
        self.scale, self.isDefScaler = dgbhdf5.isDefaultScaler(scale, imgdp[dgbkeys.infodictstr])
        self.transform = dgbkeys.listify(transform)
        self.transformer = False
        self.transform_seed = dgbhdf5.getSeed(imgdp[dgbkeys.infodictstr])
        self.trfm_copy, self.trfm_multiplier = transform_copy, 0

    def __len__(self):
        return len(self._data_IDs)

    def __getitem__(self, _idx):
        idx, rem = np.divmod(_idx, self.trfm_multiplier+1)
        if self.ndims < 2:
            X, Y = self._adaptShape(self.X[idx], self.y[idx])
            return X, Y
        if self.transformer:
            X, Y = self.transformer(self.X[idx], self.y[idx], _idx, mixed_val = rem)
            X, Y = self._adaptShape(X, Y)
            return X, Y
        else:
            X, Y = self.X[idx], self.y[idx]
            X, Y = self._adaptShape(X, Y)
            return X, Y

    def set_chunk(self, ichunk):
        self.info = self.imgdp[dgbkeys.infodictstr]
        nbchunks = len(self.info[dgbkeys.trainseldicstr])
        if nbchunks > 1 or dgbhdf5.isCrossValidation(self.info):
            return self.set_fold(ichunk, 1)
        else:
            return self.get_data(self.imgdp, ichunk)

    def set_fold(self, ichunk, ifold):
        from dgbpy import mlapply as dgbmlapply
        trainchunk  = dgbmlapply.getScaledTrainingDataByInfo( self.info,
                                                flatten=False,
                                                scale=self.isDefScaler, ichunk=ichunk, ifold=ifold)
        return self.get_data(trainchunk, ichunk)

    def set_transform_seed(self):
        if self.transform_seed:
            self.transform_seed+=1
        self.transformer.set_uniform_generator_seed(self.transform_seed, len(self))

    def get_data(self, trainchunk, ichunk):
        from dgbpy import dgbtorch
        X, y, info, im_ch, self.ndims = dgbtorch.getDatasetPars(trainchunk, False)
        self.X = X.astype('float32')
        self.y = y.astype('float32')

        if ichunk == 0: # initialise transforms on first chunk only
            self.set_transforms(info)
        self._data_IDs = range(len(self.X)*(self.trfm_multiplier+1))
        return True

    def set_transforms(self, info):
        from dgbpy import transforms as T
        if not self.isDefScaler:
            self.transform.append(self.scale)
        self.transformer = T.TransformCompose(self.transform, info, self.ndims, mixed = self.trfm_copy)
        self.trfm_multiplier = self.transformer.multiplier

    def _adaptShape(self, X, Y):
        classification = self.info[dgbkeys.classdictstr]
        if self.ndims == 3:
            if len(X.shape)==len(Y.shape) and len(X.shape)==4 and classification:     #segmentation
                data = X[:, :, :, :]
                label = Y[:, :, :, :]
            elif len(X.shape)>len(Y.shape) and classification:     #supervised
                data = X[:, :, :]
                label = Y[:]
            elif not self.info[dgbkeys.classdictstr]:
                if len(X.shape)==len(Y.shape):
                    data = X[:, :, :, :]
                    label = Y[:, :, :, :]
                elif len(X.shape)>len(Y.shape):    #supervised regression
                    data = X[:, :, :, :]
                    label = Y[:]
        elif self.ndims == 2:
            if len(X.shape)==len(Y.shape) and len(X.shape)==4 and classification:     #segmentation
                data = X[:, 0, :, :]
                label = Y[:, 0, :, :]
            elif len(X.shape)>len(Y.shape) and classification:     #supervised
                data = X[:, 0, :, :]
                label = Y[:]
            elif not self.info[dgbkeys.classdictstr]:
                if len(X.shape)==len(Y.shape):
                    data = X[:, 0, :, :]
                    label = Y[:, 0, :, :]
                elif len(X.shape)>len(Y.shape):    #supervised regression
                    data = X[:, 0, :, :]
                    label = Y[:]
        elif self.ndims == 1:
            if len(X.shape)==len(Y.shape) and len(X.shape)==5 and classification:     #segmentation
                data = X[:, 0, 0, :]
                label = Y[:, 0, 0, :]
            elif len(X.shape)>len(Y.shape) and classification:     #supervised classification
                data = X[:, 0, 0, :]
                label = Y[:]
            elif not self.info[dgbkeys.classdictstr]:
                if len(X.shape)==len(Y.shape):
                    data = X[:, 0, 0, :]
                    label = Y[:, 0, 0, :]
                elif len(X.shape)>len(Y.shape):    #supervised regression
                    data = X[:, 0, 0, :]
                    label = Y[:]
        elif classification:
            return X[:, 0, 0, :], Y[:]
        else:
            return X[:, 0, 0, :], Y[:]
        return data, label

class SeismicTestDataset(Dataset):
    def __init__(self, imgdp, scale):
        self.imgdp = imgdp
        self.scale, self.isDefScaler=dgbhdf5.isDefaultScaler(scale, imgdp[dgbkeys.infodictstr])
        self.transform = []

    def __len__(self):
        return self.X.shape[0]

    def transformer(self, image, label, index):
        if self.transform:
            return self.transform(image, label, index)
        return image, label

    def set_chunk(self, ichunk):
        self.info = self.imgdp[dgbkeys.infodictstr]
        nbchunks = len(self.info[dgbkeys.trainseldicstr])
        if nbchunks > 1 or dgbhdf5.isCrossValidation(self.info):
            return self.set_fold(ichunk, 1)
        else:
            return self.get_data(self.imgdp, ichunk)

    def set_fold(self, ichunk, ifold):
        from dgbpy import mlapply as dgbmlapply
        validchunk  = dgbmlapply.getScaledTrainingDataByInfo( self.info,
                                                flatten=False,
                                                scale=True, ichunk=ichunk, ifold=ifold)
        return self.get_data(validchunk, ichunk)

    def get_data(self, validchunk, ichunk):
        from dgbpy import dgbtorch
        from dgbpy import transforms as T
        X, y, info, im_ch, self.ndims = dgbtorch.getDatasetPars(validchunk, True)
        self.X = X.astype('float32')
        self.y = y.astype('float32')

        if ichunk == 0:
            if not self.isDefScaler:
                self.transform = T.TransformCompose(self.scale, info, self.ndims)
        return True

    def __getitem__(self,index):
        classification = self.info[dgbkeys.classdictstr]
        if self.ndims == 3:
            if len(self.X.shape)==len(self.y.shape) and len(self.X.shape)==5 and classification:   #segmentation
                data = self.X[index, :, :, :, :]
                label = self.y[index, :, :, :, :]
            elif len(self.X.shape)>len(self.y.shape) and classification:    #supervised
                data = self.X[index, :, :, :, :]
                label = self.y[index, :]
            elif not self.info[dgbkeys.classdictstr]:
                if len(self.X.shape)==len(self.y.shape):
                    data = self.X[index, :, :, :, :]
                    label = self.y[index, :, :, :, :]
                elif len(self.X.shape)>len(self.y.shape):    #supervised regression
                    data = self.X[index, :, :, :, :]
                    label = self.y[index, :]
        elif self.ndims == 2:
            if len(self.X.shape)==len(self.y.shape) and len(self.X.shape)==5 and classification:   #segmentation
                data = self.X[index, :, 0, :, :]
                label = self.y[index, :, 0, :, :]
            elif len(self.X.shape)>len(self.y.shape) and classification:    #supervised
                data = self.X[index, :, 0, :, :]
                label = self.y[index, :]
            elif not self.info[dgbkeys.classdictstr]:
                if len(self.X.shape)==len(self.y.shape):
                    data = self.X[index, :, 0, :, :]
                    label = self.y[index, :, 0, :, :]
                elif len(self.X.shape)>len(self.y.shape):    #supervised regression
                    data = self.X[index, :, 0, :, :]
                    label = self.y[index, :]
        elif self.ndims == 1:
            if len(self.X.shape)==len(self.y.shape) and len(self.X.shape)==5 and classification:   #segmentation
                data = self.X[index, :, 0, 0, :]
                label = self.y[index, :, 0, 0, :]
            elif len(self.X.shape)>len(self.y.shape) and classification:    #supervised classification
                data = self.X[index, :, 0, 0, :]
                label = self.y[index, :]
            elif not self.info[dgbkeys.classdictstr]:
                if len(self.X.shape)==len(self.y.shape):
                    data = self.X[index, :, 0, 0, :]
                    label = self.y[index, :, 0, 0, :]
                elif len(self.X.shape)>len(self.y.shape):    #supervised regression
                    data = self.X[index, :, 0, 0, :]
                    label = self.y[index, :]
        elif classification:
            return self.X[index, :, 0, 0, :], self.y[index, :]
        else:
            return self.X[index, :, 0, 0, :], self.y[index, :]

        return self.transformer(data, label, index)

class DatasetApply(Dataset):
    def __init__(self, X, info, isclassification, im_ch, ndims):
        super().__init__()
        self.im_ch = im_ch
        self.ndims = ndims
        self.X = X.astype('float32')
        self.isclassification = isclassification

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,index):
        if self.ndims == 3:
            return self.X[index, :, :, :, :]
        elif self.ndims == 2:
            return self.X[index, :, 0, :, :]
        elif self.ndims == 1:
            return self.X[index, :, 0, 0, :]

import importlib
import pkgutil
import inspect
import os,re

from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum

class DataPredType(Enum):
  Continuous = 'Continuous Data'
  Classification = 'Classification Data'
  Segmentation = 'Segmentation'
  Any = 'Any'

class OutputType(Enum):
  Pixel = 1
  Image = 2
  Any = 3
    
class DimType(Enum):
  D1 = 1
  D2 = 2
  D3 = 3
  Any = 4

class TorchUserModel(ABC):
  """Abstract base class for user defined Torch machine learning models
  
  This module provides support for users to add their own machine learning
  models to OpendTect.

  It defines an abstract base class. Users derive there own model classes from this base
  class and implement the _make_model static method to define the structure of the torch model.
  The users model definition should be saved in a file name with "mlmodel_" as a prefix and be 
  at the top level of the module search path so it can be discovered.
  
  The "mlmodel_" class should also define some class variables describing the class:
  uiname : str - this is the name that will appear in the user interface
  uidescription : str - this is a short description which may be displayed to help the user
  predtype : DataPredType enum - type of prediction (must be member of DataPredType enum)
  outtype: OutputType enum - output shape type (OutputType.Pixel or OutputType.Image)
  dimtype : DimType enum - the input dimensions supported by model (must be member of DimType enum)
    
  """
  mlmodels = []
  
  def __init__(self, ):
    self._learnrate = None
    self._nroutputs = None
    self._data_format = None
    self._model = None

  @staticmethod
  def findModels():
    """Static method that searches the PYTHONPATH for modules containing user
    defined torch machine learning models (TorchUserModels).
    
    The module name must be prefixed by "mlmodel_". All subclasses of the
    TorchUserModel base class is each found module will be added to the mlmodels
    class variable.
    """

    mlm = []

    for _, name, ispkg in pkgutil.iter_modules(path=[Path(__file__).parent.absolute()]):
      if name.startswith("mlmodel_torch_"):
        module = importlib.import_module('.'.join(['dgbpy',name]))
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for (_, c) in clsmembers:
          if issubclass(c, TorchUserModel) & (c is not TorchUserModel):
            mlm.append(c())
    
    try:
      py_settings_path = odcommon.get_settings_filename( 'settings_python' )
      pattern = r'^PythonPath\.\d+: (.+)$'
      py_paths = []
      with open(py_settings_path, 'r') as f:
        for line in f.readlines():
          match = re.match(pattern, line)
          if match and os.path.exists(match.group(1)): py_paths.append(match.group(1))
    except FileNotFoundError: pass

    for path in py_paths:
      for root, _, files in os.walk(path):
        for file in files:
          if file.startswith('mlmodel_torch_') and file.endswith('.py'):
            relpath = os.path.relpath(root, path)
            if relpath != '.': name = '.'.join([relpath, file[:-3]]).replace(os.path.sep, '.')  
            else: name = file[:-3]
            module = importlib.import_module(name)
            clsmembers = inspect.getmembers(module, inspect.isclass)
            for (_, c) in clsmembers:
              if issubclass(c, TorchUserModel) & (c is not TorchUserModel):
                mlm.append(c())
    return mlm
  
  @staticmethod
  def findName(modname):
    """Static method that searches the found TorchUserModel's for a match with the
    uiname class variable
    
    Parameters
    ----------
    modname : str
    Name (i.e. uiname) of the TorchUserModel to search for.
    
    Returns
    -------
    an instance of the class with the first matching name in the mlmodels
    list or None if no match is found
    
    """
    return next((model for model in TorchUserModel.mlmodels if model.uiname == modname), None)
  
  @staticmethod
  def getModelsByType(pred_type, out_type, dim_type):
    """Static method that returns a list of the TorchUserModels filtered by the given
    prediction, output and dimension types
    
    Parameters
    ----------
    pred_type: DataPredType enum
    The prediction type of the model to filter by
    out_type: OutputType enum
    The output shape type of the model to filter by
    dim_type: DimType enum
    The dimensions that the model must support
    
    Returns
    -------
    a list of matching model or None if no match found
    
    """
    if isinstance(pred_type, DataPredType) and isinstance(out_type, OutputType) and\
       isinstance(dim_type, DimType) :
       return [model for model in TorchUserModel.mlmodels \
          if (model.predtype == pred_type or pred_type == DataPredType.Any) and\
	     (model.outtype == out_type or out_type == OutputType.Any) and\
               (model.dimtype == dim_type or model.dimtype == DimType.Any)]
    
    return None

  @staticmethod
  def getNamesByType(pred_type, out_type, dim_type):
      models = TorchUserModel.getModelsByType(pred_type, out_type, dim_type)
      model_names = []
      for model in models:
        if model.uiname not in model_names: model_names.append(model.uiname)
      return model_names

  @staticmethod
  def isPredType( modelnm, pred_type ):
      models = TorchUserModel.getModelsByType( pred_type, OutputType.Any, DimType.Any )
      for mod in models:
          if mod.uiname == modelnm:
              return True
      return False
  
  @staticmethod
  def isOutType( modelnm, out_type ):
      models = TorchUserModel.getModelsByType( DataPredType.Any, out_type, DimType.Any )
      for mod in models:
          if mod.uiname == modelnm:
              return True
      return False
  
  @staticmethod
  def isClassifier( modelnm ):
      return TorchUserModel.isPredType( modelnm, DataPredType.Classification )
  
  @staticmethod
  def isRegressor( modelnm ):
      return TorchUserModel.isPredType( modelnm, DataPredType.Continuous )  
  
  @staticmethod
  def isImg2Img( modelnm ):
      return TorchUserModel.isOutType( modelnm, OutputType.Image )
  
  @abstractmethod
  def _make_model(self, model_shape, nroutputs, nrattribs):
    """Abstract static method that defines a machine learning model.
    
    Must be implemented in the user's derived class
    
    Parameters
    ----------
    input_shape : tuple
    nroutputs : int (number of discrete classes for a classification)
    Number of outputs
    learnrate : float
    
    Returns
    -------
    a compiled torch model
    """
    pass

  def model(self, model_shape, nroutputs, nrattribs):
    """Creates/returns a compiled torch model instance
    
    Parameters
    ----------
    nroutputs : int (number of discrete classes for a classification)
    Number of outputs
    
    Returns
    -------
    a pytorch model architecture
    
    """
    if True:
      self._nroutputs = nroutputs
      self.model_shape = model_shape
      self.nrattribs = nrattribs
      self._model = self._make_model(model_shape, nroutputs, nrattribs)
    return self._model

TorchUserModel.mlmodels = TorchUserModel.findModels()