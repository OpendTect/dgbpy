
def _get_available_gpus():
  """Get a list of available gpu devices (formatted as strings).
  
    Workaround for TensorFlow 2 error when loading TensorFlow 1 models in Keras

    # Returns
      A list of available GPU devices.
  """
  import tensorflow as tf
  try:
    import keras.backend.tensorflow_backend as tfback
  except ModuleNotFoundError:
    return []

  if tfback._LOCAL_DEVICES is None:
    devices = tf.config.list_logical_devices()
    tfback._LOCAL_DEVICES = [x.name for x in devices]
  return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

try:
  import keras.backend.tensorflow_backend as tfback
  tfback._get_available_gpus = _get_available_gpus()
except:
  pass

