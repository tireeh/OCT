import os, pdb, cv2, random
import numpy as np

class Resize(object):
    """"directly resize the image to a given size. The scale of height and width may change"""
    def __init__(self, output_size, interpolation = cv2.INTER_LINEAR):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.interpolation = cv2.INTER_LINEAR
    
    def __call__(self, image):
        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size
        else:
            new_h, new_w = self.output_size
        
        resized_img = cv2.resize(image, (new_w, new_h), interpolation = self.interpolation)
        return resized_img
    
class Split_h(object):
    """Split image into several square patches from the height"""
    def __init__(self, start_h, random_split, num_split, augmentators = None):
        self.start_h = start_h
        self.random_split = random_split
        self.num_split = num_split
        self.augmentators = augmentators
        
    def __call__(self, image):
        h, w = image.shape[:2]
        start_h = self.start_h
        if self.random_split:
            start_h = np.random.randint(start_h)
        patches = np.empty((self.num_split, w, w))
        overlap = (w * self.num_split - h + start_h) // (self.num_split - 1)
        for i in range(self.num_split):
            patch_image = image[start_h : start_h + w, :]
            if self.augmentators is not None and len(self.augmentators) > 0:
                for augmentator in self.augmentators:
                    patch_image = augmentator(patch_image)
            
            patches[i] = patch_image
            if i < self.num_split - 2:
                start_h = start_h + w - overlap
            else:
                start_h = h - w
        return patches    
    
class Normalize_divide(object):
    def __init__(self, denominator = 255.0):
        self.denominator = float(denominator)
    
    def __call__(self, data):
        return data / self.denominator   

class Random_flip(object):
    '''random flip image
    Args:
        axis (int): the axis to flip randomly
    '''
    def __init__(self, axis = 0):
        self.axis = axis
    
    def __call__(self, image_data):
        if random.random() < 0.5:
            image_data = cv2.flip(image_data, self.axis)
        return image_data

class Rotation(object):
    '''randomly rotate image
    Args:
        angle_range(int): the range of angle in rotation
    '''
    def __init__(self, angle_range = 50):
        self.angle_range = angle_range
    
    def __call__(self, image_data):
        rows,cols = image_data.shape[:2]
        ang_rot = np.random.uniform(self.angle_range) - self.angle_range/2
        Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
        return cv2.warpAffine(image_data, Rot_M, (cols,rows))

class Translation(object):
    '''randomly translate image
    '''
    def __init__(self, trans_range = 50):
        self.trans_range = trans_range
    
    def __call__(self, image_data):
        rows,cols = image_data.shape[:2]
        tr_x = self.trans_range*np.random.uniform() - self.trans_range/2
        tr_y = self.trans_range*np.random.uniform() - self.trans_range/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
        return cv2.warpAffine(image_data, Trans_M, (cols,rows))

class Shear(object):
    '''shear mapping image
    '''
    def __init__(self, shear_range = 2):
        self.shear_range = shear_range
    
    def __call__(self, image_data):
        rows, cols = image_data.shape[:2]
        pts1 = np.float32([[5,5],[20,5],[5,20]])

        pt1 = 5+self.shear_range*np.random.uniform() - self.shear_range/2
        pt2 = 20+self.shear_range*np.random.uniform() - self.shear_range/2

        pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

        shear_M = cv2.getAffineTransform(pts1,pts2)
        return cv2.warpAffine(image_data, shear_M, (cols,rows))    
    
class To_CHW(object):
    '''transpose the images to match torch :[c, h, w]
    
    Args:
        axis_order (list): the target order of axis 
    '''
    def __init__(self, axis_order):
        self.axis_order = axis_order
    
    def __call__(self, img_data):
        return np.transpose(img_data, self.axis_order)

class Expand_channel(object):
    """Expand the channel dimension of image (for the gray image)
    """
    def __init__(self, channel_dim):
        self.channel_dim = channel_dim
    
    def __call__(self, image):
        return np.expand_dims(image, self.channel_dim)