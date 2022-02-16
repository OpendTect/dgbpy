#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:       Wayne Mogg
# Date:         February 2022
#
# _________________________________________________________________________
# various tools for Onnx model files
#
import onnx

def __model_type( onnx_model ):
    prodnm = getattr(onnx_model, 'producer_name', 'unknown')
    domain = getattr(onnx_model.opset_import[0],'domain')
    producer = { 'skl2onnx': 'sklearn' }
    return producer.get(prodnm, domain)

def __input_shape( onnx_model ):
    inp = onnx_model.graph.input
    res = '(' + str(len(inp))
    for d in range(len(inp[0].type.tensor_type.shape.dim)):
        dimvalue = inp[0].type.tensor_type.shape.dim[d].dim_value
        if dimvalue:
            res += ','
            res += str(dimvalue)
    res += ')'
    return res

def __output_shape( onnx_model ):
    out = onnx_model.graph.output
    res = '(' + str(len(out))
    for d in range(len(out[0].type.tensor_type.shape.dim)):
        dimvalue = out[0].type.tensor_type.shape.dim[d].dim_value
        if dimvalue:
            res += ','
            res += str(dimvalue)
    res += ')'
    return res

def model_info( modelfnm ):
    model = onnx.load( modelfnm )
    mi = model_info_dict( model )
    return (mi['model_type'], mi['version'], mi['input_shape'], mi['output_shape'])

def model_info_dict( onnx_model ):
    minfo = {}
    minfo['model_type'] = __model_type(onnx_model)
    minfo['version'] = onnx_model.producer_version
    minfo['input_shape'] = __input_shape(onnx_model)
    minfo['output_shape'] = __output_shape(onnx_model)
    return minfo
