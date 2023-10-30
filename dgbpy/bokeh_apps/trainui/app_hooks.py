def on_server_loaded(server_context):
    # If present, this function executes when the server starts.
    try:
        import sklearn
    except ModuleNotFoundError:
        pass
    try:
        import torch
        from dgbpy.dgbtorch import get_torch_infos
        get_torch_infos()
    except ModuleNotFoundError:
        pass
    try:
        import tensorflow
        from dgbpy.dgbkeras import get_keras_infos
        get_keras_infos()
    except ModuleNotFoundError:
        pass
        

