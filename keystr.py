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
dbkeydictstr = 'dbkey'
decimkeystr = 'decimation'
dtypeconf = 'confdtype'
dtypepred = 'preddtype'
dtypeprob = 'probdtype'
exampledictstr = 'examples'
iddictstr = 'id'
interpoldictstr = 'interpolated'
locationdictstr = 'location'
logdictstr = 'log'
infodictstr = 'info'
inputdictstr = 'input'
namedictstr = 'name'
nroutdictstr = 'nrtargets'
pathdictstr = 'path'
plfdictstr = 'platform'
preddictstr = 'prediction'
probadictstr = 'probabilities'
rangedictstr = 'range'
stepoutdictstr = 'stepout'
targetdictstr = 'target'
typedictstr = 'type'
versiondictstr = 'version'
xtraindictstr = 'x_train'
ytraindictstr = 'y_train'
ypreddictstr = 'y_pred'
zstepdictstr = 'zstep'


# Value keys

classvalstr = 'Classification'
confvalstr = 'Confidence'
kerasplfnm = 'keras'
loglogtypestr = 'Log-Log Prediction'
modelnm = 'new model'
numpyvalstr = 'numpy'
seisproptypestr = 'Property Prediction from Seismics'
scikitplfnm = 'scikit'
seisclasstypestr = 'Seismic Classification'
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
