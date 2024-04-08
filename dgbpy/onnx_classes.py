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
import json

def __model_type( onnx_model ):
    prodnm = getattr(onnx_model, 'producer_name', 'unknown')
    domain = getattr(onnx_model.opset_import[0],'domain')
    producer = { 'skl2onnx': 'sklearn' }
    return producer.get(prodnm, domain)

def __input_shape( onnx_model: str ) ->list[int]:
    inp = onnx_model.graph.input
    return [dim.dim_value for dim in inp[0].type.tensor_type.shape.dim]

def __output_shape( onnx_model ):
    out = onnx_model.graph.output
    return [dim.dim_value for dim in out[0].type.tensor_type.shape.dim]

def __num_inputs( onnx_model ):
    return len(onnx_model.graph.input)

def __input_names( onnx_model ):
    return [inp.name for inp in onnx_model.graph.input]

def __output_names( onnx_model ):
    return [out.name for out in onnx_model.graph.output]

def __num_outputs( onnx_model ):
    return len(onnx_model.graph.output)

def __dataformat( onnx_model ):
    inpshape = [int(shp) if shp else 0 for shp in __input_shape(onnx_model)]
    dataformat = 'channels_first'
    if inpshape[1]!=0 and inpshape[-1]!=0:
        dataformat = 'channels_first' if inpshape[1]<inpshape[-1] else 'channels_last'
    elif inpshape[1]==0 or inpshape[-1]==0:
        dataformat = 'channels_first' if inpshape[1]!=0 else 'channels_last'
    return dataformat

def model_info( modelfnm ):
    model = onnx.load( modelfnm )
    mi = model_info_dict( model )
    return json.dumps(mi)

def model_info_dict( onnx_model ):
    minfo = {}
    minfo['model_type'] = __model_type(onnx_model)
    minfo['version'] = onnx_model.producer_version
    minfo['num_inputs'] = __num_inputs(onnx_model)
    minfo['num_outputs'] = __num_outputs(onnx_model)
    minfo['input_names'] = __input_names(onnx_model)
    minfo['output_names'] = __output_names(onnx_model)
    minfo['input_shape'] = [shp if shp else 0 for shp in __input_shape(onnx_model)]
    minfo['output_shape'] = [shp if shp else 0 for shp in __output_shape(onnx_model)]
    minfo['data_format'] = __dataformat(onnx_model)
    return minfo
