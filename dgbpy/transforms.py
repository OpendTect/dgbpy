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
from abc import ABC, abstractmethod

class BaseTransform(ABC):
    """
        Base class for all transforms. All transforms should inherit from this class.
        To create a new transform, inherit from this class and implement the transform_label and transform functions.

        Example:
            class MyTransform(BaseTransform):
                def __init__(self, p=0.2):
                    super().__init__()
                    self.p = p
                    self.multiplier = 1

                def transform_label(self, info):
                    return dgbhdf5.isImg2Img(info)

                def transform(self, arr):
                    return arr + 1

        Note:
            self.multiplier is used to determine the number of times the transform should be applied.
            For example, if self.multiplier = 2, then the transform will be applied twice 
              for cases like Flip that can be done in multiple directions.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.uniform_prob = np.random.uniform(0,1)
        self.do_label_transform = False

    def can_apply(self, info):
        """
            Returns True if the transform can be applied to the data.
        """
        return not dgbhdf5.isLogInput(info)

    @abstractmethod
    def transform_label(self, info):
        """
            Returns True if the transform should be applied to the label
        """
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, arr):
        """
            Returns the transformed array. This should hold the logic for the transform.
        """
        raise NotImplementedError

    def __call__(self, image=None, label=None, **kwargs):
        """
            This is the main function that is called when the transform is applied.
        """
        self.ndims = kwargs.get(dgbkeys.ndimstr, None)
        self.create_copy = kwargs.get('create_copy', False)
        if self.p > self.uniform_prob:
            image = self.transform(image)
            if self.do_label_transform:
                label = self.transform(label)
        return image, label



class Flip(BaseTransform):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
        self.multiplier = 1
        self.mult_count = -1

    def set_multiplier(self, info):
        inp_shape = info[dgbkeys.inpshapedictstr]
        if inp_shape[0] == inp_shape[1]:
            self.multiplier = 3

    def transform_label(self, info):
        return dgbhdf5.isImg2Img(info)

    def transform_pars(self, inp_shape):
        if self.ndims == 3:
            self.aug_dims = (1, 2)
            self.aug_axis = 2
            cubesz = inp_shape[1:3]
            if cubesz[0] == cubesz[1]:
                self.mult_count+=1
                self.aug_axis= self.mult_count%3

    def transform(self, arr):
        arr_shape = arr.shape[1:]
        flip2d = self.ndims == 2
        if flip2d:
            return np.fliplr(arr).copy()
        else:
            self.transform_pars(arr_shape)
            return np.rot90(arr,self.aug_axis,self.aug_dims).copy()

class GaussianNoise(BaseTransform):
    def __init__(self, p=0.2, std=0.1):
        super().__init__()
        self.p = p
        self.std = std
        self.multiplier = 1

    def transform_label(self, info):
        return False

    def transform(self, arr):
        noise = np.random.normal(loc = 0, scale = self.std, size = arr.shape).astype('float32')
        return arr + noise

def hasOpenCV():
  try:
    import cv2
  except ModuleNotFoundError:
    return False
  return True

class Rotate(BaseTransform):
    def __init__(self, p=0.2, angle=15):
        super().__init__()
        self.p = p
        self.angle = angle
        self.multiplier = 1
        if hasOpenCV():
            import cv2
            self.cv2 = cv2

    def transform_label(self, info):
        return dgbhdf5.isImg2Img(info)

    def transform(self, arr):
        angle = np.random.choice(range(-self.angle, self.angle))
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

class Translate(BaseTransform):
    def __init__(self, p = 0.15, percent = 20):
        super().__init__()
        self.p = p
        self.percent = percent / 100
        self.multiplier = 1
        from scipy.ndimage import shift
        self.shift = shift
        self.ndims = None
    
    def transform_label(self, info):
        return dgbhdf5.isImg2Img(info)

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

class FlipPolarity(BaseTransform):
    def __init__(self, p = 0.2):
        """
            Flip the polarity of the image
            
            Parameters
            ----------
            p : float
                Probability of applying the transform        
        """
        super().__init__()
        self.p = p
        self.multiplier = 1

    def transform_label(self, info):
        """
        Check if the label should be transformed
        """
        return dgbhdf5.isRegression(info)

    def transform(self, arr):
        """
            FlipPolarity transform logic
        """
        transfomed_arr = arr * -1.0
        return transfomed_arr



class ScaleTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        from dgbpy.dgbscikit import scale
        self.scale = scale
        self.multiplier = 0

    def transform_label(self, info):
        return dgbhdf5.doOutputScaling(info)   
        
class Normalization(ScaleTransform, BaseTransform):
    def __init__(self):
        super().__init__()
        from dgbpy.dgbscikit import getNewMinMaxScaler
        self.getNewMinMaxScaler = getNewMinMaxScaler

    def transform(self, arr):
        self.scaler = self.getNewMinMaxScaler(arr)
        return self.scale(arr, self.scaler)

class StandardScaler(ScaleTransform, BaseTransform):
    def __init__(self):
        super().__init__()
        from dgbpy.dgbscikit import getScaler
        self.getScaler = getScaler

    def transform(self, arr):
        self.scaler = self.getScaler(arr, False)
        return self.scale(arr, self.scaler)

class MinMaxScaler(ScaleTransform, BaseTransform):
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
    'Flip': Flip,
    'GaussianNoise': GaussianNoise,
    'Rotate': Rotate,
    'Translate': Translate,
    'FlipPolarity': FlipPolarity,
}
all_transforms.update(scale_transforms)

class TransformCompose():
    """
        Applies all the transform from a list to the data.
    """
    def __init__(self, transforms, info, ndims, create_copy = False):
        """
            Args:
                transforms (list): list of transforms to be applied.
                info (dict): info of the dataset.
                ndims (int): number of dimensions of the data.
                create_copy (bool): if True, the data will be transformed multiple times.
        """
        if not isinstance(transforms, (list, tuple)):
            transforms = [transforms]

        self.info = info
        self.ndims = ndims
        self.create_copy = create_copy
        self.use_seed = False
        self.transforms = self._readTransforms(transforms)
        self.transformLabel()
        self.multiplier = TransformMultiplier()
        self.set_params()
        

    def set_params(self):
        """
            Helps understand how many times each transform would multiply the data.
        """
        if not self.create_copy:
            return
        for transform_i in self.transforms:
            self.multiplier.add(transform_i.multiplier)

    def set_uniform_generator_seed(self, seed, nsamples):
        """
            Sets the seed sample to receive the same type of tranform for each training.
        """
        self.use_seed = True
        for transform_i in self.transforms:
            if seed: seed+=1 # set different seed for each transform
            self.randomstate = np.random.RandomState(seed=seed)
            transform_i.all_uniform_prob = self.randomstate.uniform(0, 1, nsamples)

    def passModuleCheck(self, transform_i):
        """
            Checks if the transform can be applied.
        """
        if isinstance(transform_i, Rotate) and not hasOpenCV():
            return False
        if isinstance(transform_i, (Translate, Rotate, Flip, GaussianNoise)) and dgbhdf5.isLogInput(self.info):
            return False
        return True
    
    def _readTransforms(self, transforms):
        """
            Read and initialize the transforms and checks if they can be applied.
        """
        valid_transforms = []
        for transform_i in transforms:
            if transform_i in all_transforms:
                transform_i = all_transforms[transform_i]()
            if not isinstance(transform_i, (*all_transforms.values(),)):
                continue
            if not self.passModuleCheck(transform_i):
                continue
            valid_transforms.append(transform_i)
        self.set_multiplier(valid_transforms)
        return valid_transforms
    
    def set_multiplier(self, transforms):
        if not self.create_copy:
            return
        for transform_i in transforms:
            if hasattr(transform_i, 'set_multiplier'):
                transform_i.set_multiplier(self.info)

    def transformLabel(self):
        """
            Checks if the transform can be applied to the label.
        """
        for transform in self.transforms:
            transform.do_label_transform = transform.transform_label(self.info)

    
    def copy_config(self, transform_idx):
        """
            Configures the copy method.
        """
        apply_no_transform_idx = 0
        if self.create_copy:
            copy_prob = np.zeros(len(self.multiplier))
            if transform_idx != apply_no_transform_idx:
                copy_prob[self.multiplier.process_map[transform_idx-1]] = 1.0 # use 1.0 as probability to apply the transform
            return copy_prob
        return None

    def use_copy_method(self, copy_prob):
        """
            Checks if the copy method is being used.
        """
        return isinstance(copy_prob, np.ndarray)

    def __call__(self, image, label, prob_idx, transform_idx = None):
        """
            Applies all the transforms to the data.

            Args:
                image: sample
                label: label
                prob_idx: index of the current sample used to choose the uniform probability when using seed
                transform_idx: value to be used for mixed transforms
        """
        copy_prob = self.copy_config(transform_idx)
        for tr_label, transform_i in enumerate(self.transforms):
            if hasattr(transform_i, 'p') and self.use_copy_method(copy_prob):
                transform_i.p = copy_prob[tr_label]
            if self.use_seed:
                transform_i.uniform_prob = transform_i.all_uniform_prob[prob_idx] #set current uniform probability for each transform
            image, label = transform_i(image=image, label=label, ndims=self.ndims, create_copy=self.create_copy)
        return image, label


class TransformMultiplier:
    """
        Helps understand how many times each transform would multiply the data.
    """
    def __init__(self):
        self.divmod_r = 0
        self.tr_idx = 0
        self.process_map = {self.divmod_r: self.tr_idx}
    
    def add(self, multiplier):
        """
            Adds the multiplier to the process map.
        """
        self.tr_idx += 1
        for id in range(multiplier):
            self.divmod_r+=1
            self.process_map[self.divmod_r] = self.tr_idx

    def __len__(self):
        return self.tr_idx+1