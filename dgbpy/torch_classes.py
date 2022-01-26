#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        Olawale I.
# Date:          Sept 2021
#
# _________________________________________________________________________
# various tools machine learning using PyTorch platform
#

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import Linear, ReLU, Sequential, Conv1d, Conv2d, Conv3d
from torch.nn import MaxPool1d, MaxPool2d, MaxPool3d, Softmax, BatchNorm1d, BatchNorm2d, BatchNorm3d, Dropout
from sklearn.metrics import accuracy_score, f1_score
import dgbpy.keystr as dgbkeys
import odpy.common as odcommon
#import albumentations as A

class Net(nn.Module):   
    def __init__(self, output_classes, dim, nrattribs):
        super(Net, self).__init__()
        
        self.output_classes = output_classes
        self.dim, self.nrattribs = dim, nrattribs
        if output_classes==1:
            self.activation = ReLU()
        else:
            self.activation = Softmax()
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

        self.cnn_layers = Sequential(
            Conv(nrattribs, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm(4),
            ReLU(inplace=True),
            MaxPool(kernel_size=2, stride=2),
        )

        self.linear_layers_3D = Sequential(
            Linear(4096, self.output_classes),
            self.activation,
        )

        self.linear_layers_2D = Sequential(
            Linear(512, self.output_classes),
            self.activation,
        )
        
        self.linear_layers_1D = Sequential(
            Linear(64, self.output_classes),
            self.activation,
        )

        self.linear_layers_D = Sequential(
            Linear(40, self.output_classes),
            self.activation,
        )
 
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        try:
            if self.dim==3:
                x = self.linear_layers_3D(x)
            elif self.dim==2:
                x = self.linear_layers_2D(x)
            elif self.dim==1:
                x = self.linear_layers_1D(x)
        except RuntimeError:
            x = self.linear_layers_D(x)

        return x   

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 earlystopping: int = 5,
                 imgdp = None
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook,
        self.earlystopping = earlystopping
        self.imgdp = imgdp

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.training_accuracy = []
        self.validation_accuracy = []
        self.F1_old = 0.0
        self.MAE = 100 ** 10000

    def run_trainer(self):
        odcommon.log_msg(f'Device is: {self.device}')
        for i in range(self.epochs):
            """Epoch counter"""
            odcommon.log_msg(f'----------------- Epoch {i + 1} ------------------')
            self.epoch += 1
            """Training block"""
            self._train()
            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()
            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])
                else:
                    self.lr_scheduler.batch() 
        classification = self.imgdp[dgbkeys.infodictstr][dgbkeys.classdictstr]
        if classification:
            odcommon.log_msg(f'Best model with validation accuracy {np.round(self.validation_best, 4)} saved.')
        else:
            odcommon.log_msg(f'Best model with validation MAE {np.round(self.validation_best, 4)} saved.')
        return (self.savemodel, self.training_loss, self.validation_loss, self.training_accuracy, 
                self.validation_accuracy, self.learning_rate)

    def _train(self):
        self.model.train()
        train_losses = [] 
        train_accs = []
        classification = self.imgdp[dgbkeys.infodictstr][dgbkeys.classdictstr]
        for input, target in self.training_DataLoader:
            self.optimizer.zero_grad()
            out = self.model(input) 
            if len(self.imgdp[dgbkeys.xtraindictstr].shape)==len(self.imgdp[dgbkeys.ytraindictstr].shape) and len(self.imgdp[dgbkeys.xtraindictstr].shape)==5 and classification:
                target = target.type(torch.LongTensor)
                pred = out.detach().cpu().numpy()
                pred = np.argmax(pred, axis=1)
                acc = accuracy_score(pred.flatten(), target.flatten())
            elif len(self.imgdp[dgbkeys.xtraindictstr].shape)>len(self.imgdp[dgbkeys.ytraindictstr].shape) and classification:
                target = target.type(torch.LongTensor)
                pred = out.detach().numpy()
                pred = np.argmax(pred, axis=1)
                acc = accuracy_score(pred, target)
            elif not classification:
                from sklearn.metrics import mean_absolute_error
                pred = out.detach().cpu().numpy()
                acc = mean_absolute_error(pred.flatten(), target.flatten())
            loss = self.criterion(out, target.squeeze(1))
            loss_value = loss.item()
            train_losses.append(loss_value)
            train_accs.append(acc)
            loss.backward()
            self.optimizer.step()
        self.training_loss.append(np.mean(train_losses))
        self.training_accuracy.append(np.mean(train_accs))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
        odcommon.log_msg(f'Train loss: {np.round(np.mean(train_losses), 4)}')
        if classification:
            odcommon.log_msg(f'Train Accuracy: {np.round(np.mean(train_accs, dtype="float64"), 4)}')
        else:
            odcommon.log_msg(f'Train MAE: {np.round(np.mean(train_accs, dtype="float64"), 4)}')

    def _validate(self):
        self.model.eval() 
        valid_losses = []  
        valid_accs = []
        classification = self.imgdp[dgbkeys.infodictstr][dgbkeys.classdictstr]
        for input, target in self.validation_DataLoader:
            with torch.no_grad():
                out = self.model(input)
                if len(self.imgdp[dgbkeys.xtraindictstr].shape)==len(self.imgdp[dgbkeys.ytraindictstr].shape) and len(self.imgdp[dgbkeys.xtraindictstr].shape)==5 and classification:  #segmentation
                    target = target.type(torch.LongTensor)
                    target = target[:, :, :, :]
                    val_pred = out.detach().cpu().numpy()
                    val_pred = np.argmax(val_pred, axis=1)
                    acc = accuracy_score(val_pred.flatten(), target.flatten())
                elif len(self.imgdp[dgbkeys.xtraindictstr].shape)>len(self.imgdp[dgbkeys.ytraindictstr].shape) and classification:
                    target = target.type(torch.LongTensor)
                    val_pred = out.detach().numpy()
                    val_pred = np.argmax(val_pred, axis=1)
                    acc = accuracy_score(val_pred, target)
                elif not classification:
                    from sklearn.metrics import mean_absolute_error
                    val_pred = out.detach().cpu().numpy()
                    acc = mean_absolute_error(val_pred.flatten(), target.flatten())
                loss = self.criterion(out, target.squeeze(1))
                loss_value = loss.item()
                valid_losses.append(loss_value)
                valid_accs.append(acc)
        self.validation_loss.append(np.mean(valid_losses))
        self.validation_accuracy.append(np.mean(valid_accs))
        odcommon.log_msg(f'Validation loss: {np.round(np.mean(valid_losses), 4)}')
        if classification:
            odcommon.log_msg(f'Validation Accuracy: {np.round(np.mean(valid_accs, dtype="float64"), 4)}')
        else:
            odcommon.log_msg(f'Validation MAE: {np.round(np.mean(valid_accs, dtype="float64"), 4)}')
        if self.F1_old < np.mean(valid_accs) and classification:
            self.F1_old = np.mean(valid_accs)
            self.savemodel = self.model
            self.validation_best = np.mean(valid_accs, dtype="float64")
        elif self.MAE > np.mean(valid_accs) and not classification:
            self.F1_old = np.mean(valid_accs)
            self.MAE = self.F1_old
            self.savemodel = self.model
            self.validation_best = np.mean(valid_accs, dtype="float64")

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
        elif self.ndims==1:
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
            # print(m)
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                '''
                # Do not initialize bias (due to batchnorm)-
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                '''
            
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                # Standard initialization for batch normalization-
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


############ 3D UNET SEGMENTATION START ###############

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
        num_groups = int(normalization.partition('group')[-1])  # get the group size from string
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
                 n_blocks: int = 4,
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

class SeismicTrainDataset:
    def __init__(self, X, y, info,  im_ch, ndims):
        super().__init__()
        self.im_ch = im_ch
        self.ndims = ndims
        self.info = info
        self.X = X.astype('float32')
        self.y = y.astype('float32')
        '''
        self.aug = A.Compose([
            A.ShiftScaleRotate(p=0.35, shift_limit=0, scale_limit=0.30, rotate_limit=30) ,
            A.HorizontalFlip(p=0.5),
#             A.RandomCrop(p=1, height=256, width=256),
        ])
        '''

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,index):
        classification = self.info[dgbkeys.classdictstr]
        if self.ndims == 3:
            if len(self.X.shape)==len(self.y.shape) and len(self.X.shape)==5 and classification:     #segmentation
                data = self.X[index, :, :, :, :]
                label = self.y[index, :, :, :, :]
            elif len(self.X.shape)>len(self.y.shape) and classification:     #supervised
                data = self.X[index, :, :, :]
                label = self.y[index, :]
            elif not self.info[dgbkeys.classdictstr]:
                if len(self.X.shape)==len(self.y.shape):
                    data = self.X[index, :, :, :, :]
                    label = self.y[index, :, :, :, :]
                elif len(self.X.shape)>len(self.y.shape):    #supervised regression
                    data = self.X[index, :, :, :, :]
                    label = self.y[index, :]
        elif self.ndims == 2:
            if len(self.X.shape)==len(self.y.shape) and len(self.X.shape)==5 and classification:     #segmentation
                data = self.X[index, :, 0, :, :]
                label = self.y[index, :, 0, :, :]
            elif len(self.X.shape)>len(self.y.shape) and classification:     #supervised
                data = self.X[index,  :, 0, :, :]
                label = self.y[index, :]
            elif not self.info[dgbkeys.classdictstr]:
                if len(self.X.shape)==len(self.y.shape):
                    data = self.X[index, :, 0, :, :]
                    label = self.y[index, :, 0, :, :]
                elif len(self.X.shape)>len(self.y.shape):    #supervised regression
                    data = self.X[index, :, 0, :, :]
                    label = self.y[index, :]
        elif self.ndims == 1:
            if len(self.X.shape)==len(self.y.shape) and len(self.X.shape)==5 and classification:     #segmentation
                data = self.X[index, :, 0, 0, :]
                label = self.y[index, :, 0, 0, :]
            elif len(self.X.shape)>len(self.y.shape) and classification:     #supervised classification
                data = self.X[index, :, 0, 0, :]
                label = self.y[index, :]
            elif not self.info[dgbkeys.classdictstr]:
                if len(self.X.shape)==len(self.y.shape):
                    data = self.X[index, :, 0, 0, :]
                    label = self.y[index, :, 0, 0, :]
                elif len(self.X.shape)>len(self.y.shape):    #supervised regression
                    data = self.X[index, :, 0, 0, :]
                    label = self.y[index, :]
        else:
            return self.X[index, :, 0, 0, :], self.y[index, :]

        return data, label

class SeismicTestDataset:
    def __init__(self, X, y, info,  im_ch, ndims):
        super().__init__()
        self.im_ch = im_ch
        self.ndims = ndims
        self.info = info
        self.X = X.astype('float32')
        self.y = y.astype('float32')

    def __len__(self):
        return self.X.shape[0]

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

        return data, label


############ 3D UNET SEGMENTATION END ###############


class DatasetApply(Dataset):
    def __init__(self, X, isclassification, im_ch, ndims):
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

  Examples
  --------
    from dgbpy.torch_classes import TorchUserModel, DataPredType, OutputType, DimType
  
    class myModel(TorchUserModel):
      uiname = 'mymodel'
      uidescription = 'short description of model'
      predtype = DataPredType.Classification
      outtype = OutputType.Pixel
      dimtype = DimType.D3
      
      def _make_model(self, input_shape, nroutputs, learnrate, data_format):
        inputs = Input(input_shape)
        conv1 = Conv3D(2, (3,3,3), activation='relu', padding='same')(inputs)
        conv1 = Conv3D(2, (3,3,3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling3D(pool,size=(2,2,2))(conv1)
        ...
        conv8 = Conv3D(1, (1,1,1,), activation='sigmoid')(conv7)
      
        model = Model(inputs=[inputs], outputs=[conv8])
        model.compile(optimizer = Adam(lr = 1e-4), loss = cross_entropy_balanced, metrics = ['accuracy'])
        return model
      
    
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
        
    for _, name, ispkg in pkgutil.iter_modules():
      if name.startswith('mlmodel_torch_'):
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
      return [model.uiname for model in models]
  
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
    Defines input data shape in the torch default data_format for the current backend.
    For the TensorFlow backend the default data_format is 'channels_last'
    nroutputs : int (number of discrete classes for a classification)
    Number of outputs
    learnrate : float
    The step size applied at each iteration to move toward a minimum of the loss function
    
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

