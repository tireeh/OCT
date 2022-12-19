import torch, cv2
import math
import numbers
import random, pdb
import numpy as np

from PIL import Image, ImageOps

class Normalize_divide(object):
    def __init__(self, denominator = 255.0):
        self.denominator = float(denominator)
    
    def __call__(self, image):
        image = np.array(image).astype(np.float32)
        image /= 255.0
        return image

class Split_OCTimage(object):
    """split the specific image into two peices without overlap"""
    def __init__(self):
        pass
        
    
    def __call__(self, image):
        w, h = img.size
        assert w == 512 and h == 1024
        split0 = image.crop((0, 0, 512, 512)) # 4-tuple defining the left, upper, right, and lower pixel coordinate
        split1 = image.crop((512, 0, 512, 1024))
        return [split0, split1]


class NLMDenoising(object):
    def __init__(self, h = 10, hForColorComponents = 10, templateWindowSize =7, searchWindowSize = 21):
        self.h = h
        self.hForColorComponents = hForColorComponents
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize
    
    def __call__(self, image):
        image_arr = np.array(image)
        image_denoised = cv2.fastNlMeansDenoising(image_arr, self.h, self.hForColorComponents, 
                                                  self.templateWindowSize, self.searchWindowSize)
        return image_denoised


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        pass
    
    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.expand_dims(np.array(image).astype(np.float32), -1).transpose((2, 0, 1))
        
        image = torch.from_numpy(image).float()

        return image
