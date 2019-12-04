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

attribdictstr = 'attribute'
classdictstr = 'classification'
classesdictstr = 'classes'
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
iddictstr = 'id'
infodictstr = 'info'
inputdictstr = 'input'
interpoldictstr = 'interpolated'
locationdictstr = 'location'
logdictstr = 'log'
namedictstr = 'name'
nroutdictstr = 'nrtargets'
pathdictstr = 'path'
plfdictstr = 'platform'
preddictstr = 'prediction'
probadictstr = 'probabilities'
rangedictstr = 'range'
scaledictstr = 'scale'
shapedictstr = 'shape'
stepoutdictstr = 'stepout'
surveydictstr = 'survey'
targetdictstr = 'target'
traindictstr = 'train'
trainseldicstr = 'training_selection'
typedictstr = 'type'
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

classvalstr = 'Classification'
confvalstr = 'Confidence'
cubeletvalstr = 'Cubelets'
imagevalstr = 'Images'
kerasplfnm = 'keras'
loglogtypestr = 'Log-Log Prediction'
modelnm = 'new model'
numpyvalstr = 'numpy'
scikitplfnm = 'scikit'
seisclasstypestr = 'Seismic Classification'
seisimgtoimgtypestr = 'Seismic Image Transformation'
seispredtypestr = 'Seismic Prediction'
seisproptypestr = 'Property Prediction from Seismics'
shapestr = 'Shape'
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
