#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : Matthew Oke
# DATE     : August 2022
#
# Data Augmentation for Keras and Pytorch
#
#

import numpy as np
from dgbpy import keystr as dgbkeys
from dgbpy import hdf5 as dgbhdf5

class RandomFlip():
    def __init__(self, p=0.3):
        self.p = p
        self.multiplier = 1

    def transformLabel(self, info):
        return dgbhdf5.isImg2Img(info)

    def __call__(self, image=None, label=None, ndims=None):
        if self.p > np.random.uniform(0,1):
            if not isinstance(label, np.ndarray):
                return self.transform(image), label
            return self.transform(image), self.transform(label)
        return image, label

    def transform_pars(self, inp_shape):
        if len(inp_shape) == 3:
            self.aug_dims = (1, 2)
            self.aug_count = 2
            cubesz = inp_shape[1:3]
            if cubesz[0] == cubesz[1]:
                self.aug_count= np.random.randint(low=0, high=4)

    def transform(self, arr):
        arr_shape = arr.shape[1:]
        self.transform_pars(arr_shape)
        flip2d = len(arr_shape) == 2
        if flip2d:
            return np.fliplr(arr)
        else:
            return np.rot90(arr,self.aug_count,self.aug_dims).copy()

class RandomGaussianNoise():
    def __init__(self, p=0.3, std=0.1):
        self.p = p
        self.std = std
        self.multiplier = 1

    def transformLabel(self, info):
        return False
        
    def __call__(self, image=None, label=None, ndims=None):
        if self.p > np.random.uniform(0,1):
            self.noise = np.random.normal(loc = 0, scale = self.std, size = image.shape).astype('float32')
            return self.transform(image), label
        return image, label

    def transform(self, arr):
        arr = self.noise + arr
        return arr

def hasOpenCV():
  try:
    import cv2
  except ModuleNotFoundError:
    return False
  return True

class RandomRotation():
    def __init__(self, p=0.3, angle=15):
        self.p = p
        self.angle = angle
        self.multiplier = 1
        import cv2
        self.cv2 = cv2

    def transformLabel(self, info):
        return dgbhdf5.isImg2Img(info)
   
    def __call__(self, image=None, label=None, ndims=None):
        if self.p > np.random.uniform(0,1):
            self.ndims = ndims
            angle = np.random.choice(range(-self.angle, self.angle))
            # angle = np.random.choice([-self.angle, self.angle])
            if not isinstance(label, np.ndarray):
                return self.transform(image, angle), label
            return self.transform(image, angle), self.transform(label, angle)
        return image, label

    def transform(self, arr, angle):
        if self.ndims == 2:
            return self.transform_2d(arr, angle)
        return self.transform_3d(arr, angle)
    
    def transform_2d(self, arr, angle):
        center = ( (arr.shape[-1])//2, (arr.shape[-2])//2 )
        dst_image = (arr.shape[-1], arr.shape[-2])
        M = self.cv2.getRotationMatrix2D(center, angle, 1)
        _arr = arr.copy()
        for attrib in range(_arr.shape[0]):
            rotated_arr = self.cv2.warpAffine( np.squeeze(_arr[attrib]), M , dst_image , borderMode=self.cv2.BORDER_REFLECT)
            _arr[attrib,:] = rotated_arr[np.newaxis, :]
        return _arr.copy()

    def transform_3d(self, arr, angle):
        center = ( (arr.shape[-2])//2, (arr.shape[-3])//2 )
        dst_image = (arr.shape[-2], arr.shape[-3])
        M = self.cv2.getRotationMatrix2D(center, angle, 1)
        _arr = arr.copy()
        for attrib in range(_arr.shape[0]):
            _arr[attrib,:] = self.cv2.warpAffine( _arr[attrib], M , dst_image , borderMode=self.cv2.BORDER_REFLECT)
        return _arr.copy()

class RandomTranslation():
    def __init__(self, p = 0.15, percent = 20):
        self.p = p
        self.percent = percent / 100
        self.multiplier = 1
        from scipy.ndimage import shift
        self.shift = shift
        self.ndims = None
    
    def transformLabel(self, info):
        return dgbhdf5.isImg2Img(info)

    def __call__(self, image=None, label=None, ndims=None):
        if self.p > np.random.uniform(0,1):
            self.ndims = ndims
            if not isinstance(label, np.ndarray):
                return self.transform(image), label
            return self.transform(image), self.transform(label)
        return image, label

    def transform(self, arr):
        if self.ndims == 2:
            ax = arr.shape[2:]
            ax = map(lambda x: int(x*self.percent), ax)
            transform_axes = (0, 0, *ax)
        elif self.ndims == 3:
            ax = arr.shape[1:]
            ax = map(lambda x: int(x*self.percent), ax)
            transform_axes = (0, *ax)
        return self.shift(arr, transform_axes)

class RandomPolarityFlip():
    def __init__(self, p = 0.25):
        self.p = p
        self.multiplier = 1

    def transformLabel(self, info):
        return dgbhdf5.isRegression(info)

    def __call__(self, image=None, label=None, ndims=None):
        if self.p > np.random.uniform(0,1):
            if not isinstance(label, np.ndarray):
                return self.transform(image), label
            return self.transform(image), self.transform(label)
        return image, label

    def transform(self, arr):
        transfomed_arr = arr * -1.0
        return transfomed_arr

class ScaleTransform():
    def __init__(self):
        from dgbpy.dgbscikit import scale
        self.scale = scale
        self.multiplier = 0

    def transformLabel(self, info):
        return dgbhdf5.doOutputScaling(info)       

    def __call__(self, image=None, label=None, ndims=None):
        if not isinstance(label, np.ndarray):
            return self.transform(image), label
        return self.transform(image), self.transform(label)
        
class Normalization(ScaleTransform):
    def __init__(self):
        super().__init__()
        from dgbpy.dgbscikit import getNewMinMaxScaler
        self.getNewMinMaxScaler = getNewMinMaxScaler

    def transform(self, arr):
        self.scaler = self.getNewMinMaxScaler(arr)
        return self.scale(arr, self.scaler)

class StandardScaler(ScaleTransform):
    def __init__(self):
        super().__init__()
        from dgbpy.dgbscikit import getScaler
        self.getScaler = getScaler

    def transform(self, arr):
        self.scaler = self.getScaler(arr, False)
        return self.scale(arr, self.scaler)

class MinMaxScaler(ScaleTransform):
    def __init__(self):
        super().__init__()
        from dgbpy.dgbscikit import getNewMinMaxScaler
        self.getNewMinMaxScaler = getNewMinMaxScaler
 
    def transform(self, arr):
        self.scaler = self.getNewMinMaxScaler(arr, maxout=255)
        return self.scale(arr, self.scaler)




scale_transforms = {
    dgbkeys.localstdtypestr: StandardScaler,
    dgbkeys.normalizetypestr: Normalization,
    dgbkeys.minmaxtypestr: MinMaxScaler
}
all_transforms = {
    'RandomFlip': RandomFlip,
    'RandomGaussianNoise': RandomGaussianNoise,
    'RandomRotation': RandomRotation,
    'RandomTranslation': RandomTranslation,
    'RandomPolarityFlip': RandomPolarityFlip,
}
all_transforms.update(scale_transforms)

class TransformComposefromList():
    """
        Applies all the transform from a list to the data.
    """
    def __init__(self, transforms, info, ndims, mixed = False):
        if not isinstance(transforms, (list, tuple)):
            transforms = [transforms]
        self.info = info
        self.ndims = ndims
        self.transforms = self._readTransforms(transforms)
        self.do_label = self.transformLabel()
        self.multiplier = 0
        self.mixed = mixed
        if self.mixed:
            self.set_params()

    def set_params(self):
        for transform_i in self.transforms:
            self.multiplier += transform_i.multiplier

    def passModuleCheck(self, transform_i):
        if isinstance(transform_i, RandomRotation) and not hasOpenCV():
            return False
        return True

    def _readTransforms(self, transforms):
        for tr, transform_i in enumerate(transforms):
            if transform_i in all_transforms:
                transforms[tr] = all_transforms[transform_i]()
            if not self.passModuleCheck(transform_i):
                transforms.pop(tr)
        return transforms

    def transformLabel(self):
        doLabelTransform = tuple()
        for transform in self.transforms:
            doLabelTransform += transform.transformLabel(self.info),
        return doLabelTransform

    def __call__(self, image, label, mixed_val = None):
        probs = False
        if self.mixed:
            probs = np.zeros(self.multiplier)
            if mixed_val: probs[mixed_val-1] = 1.0
        for tr_label, transform_i in enumerate(self.transforms):
            if hasattr(transform_i, 'p') and isinstance(probs, np.ndarray):
                transform_i.p = probs[tr_label]
            if self.do_label[tr_label]:
                image, label = transform_i(image=image, label=label, ndims=self.ndims)
            else:
                image, _ = transform_i(image=image, label=None, ndims=self.ndims)
        return image, label
