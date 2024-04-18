#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Jan 2019
#
# _________________________________________________________________________
# Standard keys used by the dgb machine learning python modules
#


# Dictionary keys (lower case, mostly alphabetic only):

arrayorderdictstr = 'array_order'
classdictstr = 'classification'
classesdictstr = 'classes'
classnmdictstr = 'classnm'
collectdictstr = 'collection'
componentdictstr = 'component'
confdictstr = 'confidence'
criteriondictstr = 'criterion'
datasetdictstr = 'datasets'
dbkeydictstr = 'dbkey'
decimkeystr = 'decimation'
dtypeconf = 'confdtype'
dtypepred = 'preddtype'
dtypeprob = 'probdtype'
estimatedsizedictstr = 'estimatedsize'
exampledictstr = 'examples'
filedictstr = 'filenm'
foldstr = 'fold'
geomiddictstr = 'geomid'
iddictstr = 'id'
infodictstr = 'info'
inpscalingdictstr = 'inp_scaling'
inpscalingvalsdictstr = 'inp_scalingvals'
inpshapedictstr = 'inp_shape'
inputdictstr = 'input'
interpoldictstr = 'interpolated'
learntypedictstr = 'learntype'
locationdictstr = 'location'
logdictstr = 'log'
namedictstr = 'name'
nroutdictstr = 'nroutputs'
outputunscaledictstr = 'out_unscale'
pathdictstr = 'path'
plfdictstr = 'platform'
preddictstr = 'prediction'
probadictstr = 'probabilities'
rangedictstr = 'range'
scaledictstr = 'scale'
seeddictstr = 'seed'
outshapedictstr = 'out_shape'
segmentdictstr = 'segmentation'
surveydictstr = 'survey'
targetdictstr = 'target'
traindictstr = 'train'
trainconfigdictstr = 'training_config'
trainseldicstr = 'training_selection'
validdictstr = 'validation'
versiondictstr = 'version'
withunlabeleddictstr = 'withunlabeled'
xdatadictstr = 'x_data'
xtraindictstr = 'x_train'
xvaliddictstr = 'x_validate'
ydatadictstr = 'y_data'
ytraindictstr = 'y_train'
yvaliddictstr = 'y_validate'
ypreddictstr = 'y_pred'
zstepdictstr = 'zstep'


# Value keys

averagestr = 'Average'
carrorderstr = 'C-Order'
classdatavalstr = 'Classification Data'
classesvalstr = 'Classes'
classificationvalstr = 'Classification'
confvalstr = 'Confidence'
contentvalstr = 'Content'
continuousvalstr = 'Continuous Data'
crosslinestr = 'Cross-line'
disclaimerstr = 'Disclaimer, IP Rights and Permission-to-Use'
globalstdtypestr = 'Global Standardization'
inlinestr = 'In-line'
inpshapestr = 'Input.Shape'
kerasplfnm = 'keras'
localstdtypestr = 'Local Standardization'
logclustertypestr = 'Log Clustering'
loglogtypestr = 'Log-Log Prediction'
maxstr = "Maximum"
minstr = "Minimum"
minmaxtypestr = 'MinMax'
mlsoftkey = 'OpendTect-ML-Software'
modelnm = 'new model'
modelnmstr = 'Model.Name'
normalizetypestr = 'Normalization'
numpyvalstr = 'numpy'
onnxcudastr = 'CUDAExecutionProvider'
onnxcpustr = 'CPUExecutionProvider'
onnxplfnm = 'onnx'
outshapestr = 'Output.Shape'
prefercpustr = 'prefercpu'
reversestr = 'Reverse'
scikitplfnm = 'scikit'
segmenttypestr = 'Segmentation'
seisclasstypestr = 'Seismic Classification'
seisimgtoimgtypestr = 'Seismic Image Transformation'
seisproptypestr = 'Property Prediction from Seismics'
torchplfnm = 'torch'
typestr = 'Type'
valuestr = 'Value'
versionstr = 'Version'

# Others
ndimstr = 'ndims'
s3bokehmsg = '--S3--:'

def getNames( lst, uinames=True ):
  idx = -1
  if not uinames:
    idx = 0
  ret = list()
  for itm in lst:
    ret.append( itm[idx] )
  return ret

def getNameFromUiName( lst, itmnm ):
  for lstitms in lst:
    if lstitms[1] == itmnm:
      return lstitms[0]
  return None

def getNameFromList( lst, itmnm, uiname ):
  for lstitms in lst:
    if lstitms[0] != itmnm and lstitms[1] != itmnm:
      continue
    if uiname:
      return lstitms[1]
    else:
      return lstitms[0]
  return None

def getDefaultAugmentation():
  import os
  if 'ML_WITH_AUGMENTATION' in os.environ:
    return not ( os.environ['ML_WITH_AUGMENTATION'] == False or \
                 os.environ['ML_WITH_AUGMENTATION'] == 'No' )
  return False

def getDefaultTensorBoard():
  import os
  if 'ML_WITH_TENSORBOARD' in os.environ:
    return not ( os.environ['KERAS_WITH_TENSORBOARD'] == False or \
                 os.environ['KERAS_WITH_TENSORBOARD'] == 'No' )
  return True

def format_time(t):
    "Format `t` (in seconds) to (h):mm:ss"
    t = int(t)
    h,m,s = t//3600, (t//60)%60, t%60
    if h!= 0: return f'{h}:{m:02d}:{s:02d}'
    else:     return f'{m:02d}:{s:02d}'

from typing import *
def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]