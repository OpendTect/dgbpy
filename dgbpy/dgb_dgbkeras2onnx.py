#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        Wayne Mogg
# Date:          Feb 2023
#
# _________________________________________________________________________
# convert a dGB Keras img2img model to ONNX format
#

__version__ = '1.0'
import argparse
import sys
import datetime
import tempfile
import traceback
import onnx
import tensorflow as tf 
import tf2onnx
import dgbpy.mlio as dgbmlio
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'error:{message}\n')
        self.print_help()
        sys.exit(2)

parser = MyParser(description='Convert a dGB Keras Img2Img model to ONNX format')
parser.add_argument('-v', '--version', action='version', version=f"%(prog)s {__version__}")
parser.add_argument('-i', '--input', dest='infile',
		    help='Input dGB Keras hdf5 Img2Img file for conversion')
parser.add_argument('-o', '--output', dest='outfile',
		    help='Output ONNX file')
parser.add_argument('--opset', type=int, default=15, dest='opset',
		    help='Target ONNX opset version')
args = parser.parse_args()
if not args.infile or not args.outfile:
    parser.print_help()
    sys.exit(1)

try:

    info = dgbmlio.getInfo(args.infile)
    numin = dgbhdf5.getNrAttribs(info)
    numout = dgbhdf5.getNrOutputs(info)
    inshape = info[dgbkeys.inpshapedictstr]
    if inshape[0]==1:
        inshape = inshape[1:]

    inshape += [numin]
    inshape.insert(0, 1)
    outshape = info[dgbkeys.outshapedictstr]
    dgb_model = tf.keras.models.load_model(args.infile)
    input_sig = [tf.TensorSpec(inshape, tf.float32, name="input")]
    onnx_model, _ = tf2onnx.convert.from_keras(dgb_model, input_sig, opset=args.opset)
    if info[dgbkeys.namedictstr]:
        onnx_model.doc_string = info[dgbkeys.namedictstr] + " "
    
    onnx_model.doc_string += "exported from OpendTect"
    onnx_model.model_version = int(info[dgbkeys.versiondictstr])
    meta = onnx_model.metadata_props.add()
    meta.key = "creation_date"
    meta.value = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    meta = onnx_model.metadata_props.add()
    meta.key = "scaling_type"
    meta.value = info[dgbkeys.inpscalingdictstr]
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, args.outfile)
    print(f'Converted: {args.infile}\nto ONNX model: {args.outfile}')
except Exception:
    traceback.print_exc()

import onnxruntime as ort
import numpy as np
print(f'Comparing predictions {args.infile} vs {args.outfile}')
ex_indata = np.random.normal(0.0, 1.0, inshape).astype(np.float32)

sess = ort.InferenceSession(args.outfile)
res_onnx = sess.run(None, {"input": ex_indata})

res_keras = dgb_model(ex_indata)

for onnx_res, keras_res in zip(res_onnx, res_keras):
    np.testing.assert_allclose(onnx_res[0], keras_res, rtol=1e-5, atol=1e-5)

print("All OK")