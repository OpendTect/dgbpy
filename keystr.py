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

classdictstr = 'classification'
classesdictstr = 'classes'
classnmdictstr = 'classnm'
collectdictstr = 'collection'
componentdictstr = 'component'
confdictstr = 'confidence'
datasetdictstr = 'datasets'
dbkeydictstr = 'dbkey'
decimkeystr = 'decimation'
dtypeconf = 'confdtype'
dtypepred = 'preddtype'
dtypeprob = 'probdtype'
estimatedsizedictstr = 'estimatedsize'
exampledictstr = 'examples'
filedictstr = 'filenm'
geomiddictstr = 'geomid'
iddictstr = 'id'
infodictstr = 'info'
inpshapedictstr = 'inp_shape'
inputdictstr = 'input'
interpoldictstr = 'interpolated'
learntypedictstr = 'learntype'
locationdictstr = 'location'
logdictstr = 'log'
namedictstr = 'name'
nroutdictstr = 'nroutputs'
pathdictstr = 'path'
plfdictstr = 'platform'
preddictstr = 'prediction'
probadictstr = 'probabilities'
rangedictstr = 'range'
scaledictstr = 'scale'
outshapedictstr = 'out_shape'
surveydictstr = 'survey'
targetdictstr = 'target'
traindictstr = 'train'
trainseldicstr = 'training_selection'
validdictstr = 'validation'
versiondictstr = 'version'
xdatadictstr = 'x_data'
xtraindictstr = 'x_train'
xvaliddictstr = 'x_validate'
ydatadictstr = 'y_data'
ytraindictstr = 'y_train'
yvaliddictstr = 'y_validate'
ypreddictstr = 'y_pred'
zstepdictstr = 'zstep'


# Value keys

classdatavalstr = 'Classification Data'
classesvalstr = 'Classes'
classificationvalstr = 'Classification'
continuousvalstr = 'Continuous Data'
confvalstr = 'Confidence'
contentvalstr = 'Content'
inpshapestr = 'Input.Shape'
kerasplfnm = 'keras'
loglogtypestr = 'Log-Log Prediction'
mlsoftkey = 'OpendTect-ML-Software'
modelnm = 'new model'
numpyvalstr = 'numpy'
outshapestr = 'Output.Shape'
scikitplfnm = 'scikit'
seisclasstypestr = 'Seismic Classification'
seisimgtoimgtypestr = 'Seismic Image Transformation'
seisproptypestr = 'Property Prediction from Seismics'
typestr = 'Type'
valuestr = 'Value'
versionstr = 'Version'

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
