#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        Wayne Mogg
# Date:          Oct 2021
#
# _________________________________________________________________________
# convert an sklearn model to ONNX format
#

__version__ = '1.0'
import argparse
import joblib
import sys

parser = argparse.ArgumentParser(
	    description='Convert an sklearn model to ONNX format')
parser.add_argument('-v', '--version', action='version', version=f"%(prog)s {__version__}")
parser.add_argument('-i', '--input', dest='infile',
		    help='Input sklearn joblib file for conversion')
parser.add_argument('-o', '--output', dest='outfile',
		    help='Output ONNX file')
args = parser.parse_args()

try:
    skl_model = joblib.load(args.infile)
except Exception as e:
    print(e)
    sys.exit(1)

import dgbpy.sklearn_classes as skc
mi = skc.model_info_dict( skl_model )

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

nfeatures = mi['nfeatures']
initial_type = [('float_input', FloatTensorType([None, nfeatures]))]
if mi['module']=='xgboost.sklearn':
    from skl2onnx import update_registered_converter
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
    import onnxmltools.convert.common.data_types
    if mi['esttype']=='regressor':
        from xgboost import XGBRegressor
        from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
        update_registered_converter( XGBRegressor, 'XGBoostXGBRegressor',
                     calculate_linear_regressor_output_shapes, convert_xgboost)
    else:
        from xgboost import XGBClassifier
        from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
        update_registered_converter( XGBClassifier, 'XGBoostXGBClassifier',
                     calculate_linear_classifier_output_shapes, convert_xgboost,
                     options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

try:
    onx = convert_sklearn(skl_model, initial_types=initial_type)
    with open(args.outfile, "wb") as f:
        f.write(onx.SerializeToString())
except Exception as e:
    print(e)

import numpy as np
ntest = 100
X = np.random.normal(0.0, 1.0, (ntest, nfeatures))
sk_predict = skl_model.predict(X)

import onnxruntime as rt
sess = rt.InferenceSession(args.outfile)
pred_onx = sess.run(None, {"float_input": X.astype(np.float32)})

if mi['esttype']=='regressor':
    onnx_predict = pred_onx[0].ravel()
    mae = np.sum(np.absolute(sk_predict-onnx_predict))/ntest
    print("Prediction MAE: ", mae)
else:
    onnx_predict = pred_onx[0]
    sk_prob = skl_model.predict_proba(X)
    onnx_prob = pred_onx[1]
    mae = np.sum(np.absolute(sk_prob-onnx_prob))/ntest
    print("Prediction Probability MAE: ", mae)

