import numpy as np
from joblib import load
import json

def model_info( modelfnm ):
    model = load( modelfnm )
    esttype = getattr(model,'_estimator_type','Unknown')
    modtype = getattr(type(model),'__name__', None)
    module = getattr(model,'__module__',None)
    params = json.dumps(model.get_params())
    if hasattr(model,'estimators_'):
      if isinstance(model.estimators_,np.ndarray):
        model = model.estimators_[0][0]
      else:
        model = model.estimators_[0]

    nfeatures = getattr(model,'n_features_',None)
    noutputs = getattr(model,'n_outputs_',None)
    coef = getattr(model,'coef_', None)
    classes = getattr(model,'classes_',None)
    if nfeatures is None:
      if coef is not None:
        nfeatures = coef.shape[1]

    if noutputs is None:
      if coef is not None:
        noutputs = coef.shape[0]

    if classes is not None:
      pass
    elif esttype is 'classifier' and noutputs is not None:
        classes = [i for i in range(noutputs)]

    return (esttype, modtype, module, params, nfeatures, noutputs, classes)
