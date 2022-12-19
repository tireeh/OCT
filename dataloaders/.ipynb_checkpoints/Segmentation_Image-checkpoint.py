import os, pdb, re, cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from PIL import Image

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

class OCT_image_segmentation(Dataset):
    def __init__(self, data_root, 
                 label_root, 
                 included_pixels,
                 label_dict,
                 aug_dict,
                 denoising,
                 transform = None):
        self.transform = transform
        
        self.images, self.labels = [], []
        for sample_folder in tqdm(sorted(os.listdir(data_root))):
            image_names = os.listdir(os.path.join(data_root, sample_folder))
            image_names.sort(key=natural_keys)
            for image_name in image_names:
                label_image = Image.open(os.path.join(label_root, sample_folder.replace(".img", "_labelMark"), image_name))
                image = Image.open(os.path.join(data_root, sample_folder, image_name))
                label_image_arr = np.array(label_image)
                
                if denoising:
                    image_arr = np.array(image)
                    image_denoised = cv2.fastNlMeansDenoising(image_arr, 10, 10, 7, 21)
                    image = Image.fromarray(image_denoised, "L")
                
                passed = False
                for included_pixel in included_pixels:
                    if included_pixel in label_image_arr:
                        passed = True
                        
                if not passed: continue
                
                # convert the target pixel in label image into label
                for target_pixel in label_dict.keys():
                    np.place(label_image_arr, label_image_arr==target_pixel, label_dict[target_pixel])
                label_image = Image.fromarray(np.uint8(label_image_arr))
                
                aug_times = 0
                if aug_dict is not None:
                    for target_pixel in aug_dict.keys():
                        if target_pixel in label_image_arr:
                            aug_times = aug_dict[target_pixel]
                
                if aug_times > 0:
                    for i in range(aug_times):
                        self.images.append(image.copy())
                        self.labels.append(label_image.copy())
                else:
                    self.images.append(image)
                    self.labels.append(label_image)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        sample = {'image': self.images[idx], 'label': self.labels[idx]}
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample    

class OCT_segmentation_persample(Dataset):
    def __init__(self, sample_path, 
                 transform = None):
        self.transform = transform
        self.images = []
        image_names = os.listdir(sample_path)
        image_names.sort(key=natural_keys)
        for image_name in image_names:
            image = Image.open(os.path.join(sample_path, image_name))
            self.images.append(image)
    
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]        
        if self.transform is not None:
            image = self.transform(image)
        
        return image


