#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Date:          Jan 2024
#
# _________________________________________________________________________
# various tools to support import of PyTorch models
# 
import json
import torch

def __model_type( torch_model ):
    return 'Torch Model'

def __model_impl( torch_model ):
    if torch_model.__class__.__name__ == 'RecursiveScriptModule':
        return 'torchscript'
    elif torch_model.__class__.__name__ == 'OrderedDict':
        return 'torch'
    return ''

def __model_classname( torch_model ):
    if hasattr(torch_model, 'original_name'):
        return torch_model.original_name
    return ''

def __input_shape( torch_model: str ) ->list[int]:
    return [None,None]

def __output_shape( torch_model ):
    return [None,None]

def __num_inputs( torch_model ):
    return len(__input_names(torch_model))

def __input_names( torch_model ):
    return []

def __output_names( torch_model ):
    return []

def __num_outputs( torch_model ):
    return len(__output_names(torch_model))

def model_info( modelfnm ):
    try:
        model = torch.jit.load( modelfnm )
    except RuntimeError:
        model = torch.load( modelfnm )
    mi = model_info_dict( model )
    return json.dumps(mi)

def model_info_dict( torch_model ):
    minfo = {}
    minfo['model_type'] = __model_type(torch_model)
    minfo['model_impl'] = __model_impl(torch_model)
    minfo['model_classname'] = __model_classname(torch_model)
    minfo['version'] = 1.0
    minfo['num_inputs'] = __num_inputs(torch_model)
    minfo['num_outputs'] = __num_outputs(torch_model)
    minfo['input_names'] = __input_names(torch_model)
    minfo['output_names'] = __output_names(torch_model)
    minfo['input_shape'] = [shp if shp else 1 for shp in __input_shape(torch_model)]
    minfo['output_shape'] = [shp if shp else 1 for shp in __output_shape(torch_model)]
    minfo['data_format'] = 'channels_first' if minfo['input_shape'][1]<minfo['input_shape'][-1] else 'channels_last'
    return minfo
