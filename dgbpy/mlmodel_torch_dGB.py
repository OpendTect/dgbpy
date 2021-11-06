#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Olawale Ibrahim
# DATE     : October 2021
#
# dGB PyTorch machine learning models in TorchUserModel format
#

from dgbpy.torch_classes import TorchUserModel, DataPredType, OutputType, DimType

class dGB_UnetSeg(TorchUserModel):
  uiname = 'dGB UNet Torch Segmentation'
  uidescription = 'dGBs Unet image segmentation'
  predtype = DataPredType.Classification
  outtype = OutputType.Image
  dimtype = DimType.D3
  
  def _make_model(self, nroutputs):
    model = UNet(in_channels=1, n_blocks=1, out_channels=nroutputs, dim=3)
    return model

from dgbpy.torch_classes import Net, create_resnet_block, UNet
import torch.nn as nn

class dGB_Simple_Net(TorchUserModel):
    uiname = 'Simple Net Classifier'
    uidescription = 'dGbs Simple Net Classifier Model in TorchUserModel form'
    predtype = DataPredType.Classification
    outtype = OutputType.Pixel
    dimtype = DimType.D3

    def _make_model(self, nroutputs):
        model = Net(nroutputs)
        return model

class dGB_UnetReg(TorchUserModel):
  uiname = 'dGB UNet Regression'
  uidescription = 'dGBs Unet image regression'
  predtype = DataPredType.Continuous
  outtype = OutputType.Image
  dimtype = DimType.D3
  
  def _make_model(self, nroutputs=1):
    model = UNet(in_channels=1, n_blocks=1, out_channels=nroutputs, dim=3)
    return model

class dGB_ResNet18(TorchUserModel):
    uiname = 'ResNet 18 Classifier'
    uidescription = 'dGBs ResNet Classifier Model in TorchUserModel form'
    predtype = DataPredType.Classification
    outtype = OutputType.Pixel
    dimtype = DimType.D3

    def _make_model(self, nroutputs):
        model = ResNet18(nroutputs)
        return model

def ResNet18(nroutputs):
    b0 = nn.Sequential(
    nn.Conv3d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
    nn.BatchNorm3d(num_features = 64),
    nn.ReLU())

    b1 = nn.Sequential(*create_resnet_block(input_filters = 64, output_filters = 64, num_residuals = 2, first_block = True))
    b3 = nn.Sequential(*create_resnet_block(input_filters = 64, output_filters = 128, num_residuals = 2, first_block = True))
    b5 = nn.Sequential(*create_resnet_block(input_filters = 128, output_filters = 256, num_residuals = 2, first_block = True))
    b7 = nn.Sequential(*create_resnet_block(input_filters = 256, output_filters = 512, num_residuals = 2, first_block = True))
    model = nn.Sequential(
    b0, b1, b3, b5, b7,
    nn.AdaptiveAvgPool2d(output_size = (1, 1)),
    nn.Flatten(),
    nn.Linear(in_features = 1024, out_features = nroutputs))
    return model
    