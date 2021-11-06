import numpy as np
from joblib import load
import json

def model_info( modelfnm ):
    model = load( modelfnm )
    mi = model_info_dict( model )
    return (mi['esttype'], mi['modtype'], mi['module'], mi['params'],
	    mi['nfeatures'], mi['noutputs'], mi['classes'])

def model_info_dict( skl_model ):
    minfo = {}
    minfo['esttype'] = getattr(skl_model,'_estimator_type','Unknown')
    minfo['modtype'] = getattr(type(skl_model),'__name__', None)
    minfo['module']  = getattr(skl_model,'__module__',None)
    minfo['params']  = json.dumps(skl_model.get_params())
    if hasattr(skl_model,'estimators_'):
      if isinstance(skl_model.estimators_,np.ndarray):
        skl_model = skl_model.estimators_[0][0]
      else:
        skl_model = skl_model.estimators_[0]

    if minfo['module']=='xgboost.sklearn':
        minfo['nfeatures'] = skl_model.feature_importances_.shape[0]
    else:
        minfo['nfeatures'] = getattr(skl_model,'n_features_',None)

    minfo['noutputs']  = getattr(skl_model,'n_outputs_',None)
    minfo['coef']      = getattr(skl_model,'coef_', None)
    minfo['classes']   = getattr(skl_model,'classes_',None)
    if minfo['nfeatures'] is None:
      if minfo['coef'] is not None:
        minfo['nfeatures'] = minfo['coef'].shape[1]

    if minfo['noutputs'] is None:
      if minfo['coef'] is not None:
        minfo['noutputs'] = minfo['coef'].shape[0]

    if minfo['classes'] is not None:
      pass
    elif minfo['esttype'] is 'classifier' and minfo['noutputs'] is not None:
        minfo['classes'] = [i for i in range(minfo['noutputs'])]

    return minfo
