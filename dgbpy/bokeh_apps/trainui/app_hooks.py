def on_server_loaded(server_context):
    # If present, this function executes when the server starts.
    try:
        import sklearn
    except ModuleNotFoundError:
        pass
    try:
        import torch
    except ModuleNotFoundError:
        pass
    try:
        import tensorflow
    except ModuleNotFoundError:
        pass
        

