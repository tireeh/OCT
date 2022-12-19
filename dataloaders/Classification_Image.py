import os, pdb, cv2, re
import numpy as np
import torch
from torch.utils.data import Dataset

from tqdm import tqdm

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

class OCT_image_classification(Dataset):
    def __init__(self, data_root, label_root, 
                 included_pixels, 
                 label_dict, 
                 aug_label_dict,
                 transform = None):
        self.transform = transform
        
        self.images, self.labels = [], []
        for sample_folder in tqdm(sorted(os.listdir(data_root))):
            image_names = os.listdir(os.path.join(data_root, sample_folder))
            image_names.sort(key=natural_keys)
            for image_name in image_names:
                passed = False
                
                label_image = cv2.imread(os.path.join(label_root, sample_folder.replace(".img", "_labelMark"), image_name))[:,:,0]
                for included_pixel in included_pixels:
                    if included_pixel in label_image:
                        passed = True
                        
                if passed:
                    image = cv2.imread(os.path.join(data_root, sample_folder, image_name))[:,:,0] # the three channels are the same
                    for target_pixel in label_dict.keys():
                        if target_pixel in label_image:
                            label = label_dict[target_pixel]
                    
                    if (aug_label_dict is not None) and (label in aug_label_dict.keys()):
                        for i in range(aug_label_dict[label]):
                            self.images.append(np.copy(image))
                            self.labels.append(label)
                    else:
                        self.images.append(image)
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return [image, label]

class OCT_image_multi_classification(Dataset):
    def __init__(self, data_root, label_root, 
                 included_pixels, 
                 label_dict, 
                 aug_dict,
                 num_classes,
                 transform = None):
        self.transform = transform
        
        self.images, self.labels = [], []
        for sample_folder in tqdm(sorted(os.listdir(data_root))):
            image_names = os.listdir(os.path.join(data_root, sample_folder))
            image_names.sort(key=natural_keys)
            for image_name in image_names:
                passed = False
                
                label_image = cv2.imread(os.path.join(label_root, sample_folder.replace(".img", "_labelMark"), image_name))[:,:,0]
                for included_pixel in included_pixels:
                    if included_pixel in label_image:
                        passed = True
                        
                if passed:
                    image = cv2.imread(os.path.join(data_root, sample_folder, image_name))[:,:,0] # the three channels are the same
                    label_multi_class = np.zeros(num_classes)
                    
                    for target_pixel in label_dict.keys():
                        if target_pixel in label_image:
                            label_multi_class[label_dict[target_pixel]] = 1
                    
                    aug_times = 0
                    if aug_dict is not None:
                        for target_pixel in aug_dict.keys():
                            if target_pixel in label_image:
                                aug_times = aug_dict[target_pixel]
                        
                    if aug_times > 0:
                        for i in range(aug_times):
                            self.images.append(np.copy(image))
                            self.labels.append(np.copy(label_multi_class))
                    else:
                        self.images.append(np.copy(image))
                        self.labels.append(np.copy(label_multi_class))
                    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return [image, label]
    
    
class OCT_classification_persample(Dataset):
    def __init__(self, sample_path,
                 transform = None):
        self.transform = transform
        self.images = []
        image_names = os.listdir(sample_path)
        image_names.sort(key=natural_keys)
        for image_name in image_names:
            image = cv2.imread(os.path.join(sample_path, image_name))[:,:,0] # the three channels are the same
            self.images.append(image)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image    

class Imageset(object):
    def __init__(self, txt_dicts, transform):
        """Construct image dataset from txts. Each txt contains only one class
        params:
            txt_dicts (dict): key is txt path, value is a list composed of [root, label, ratio] 
        """
        self.transform = transform
        self.image_paths, self.labels = [], []
        for txt_path, [root, label, ratio] in txt_dicts.items():
            image_names, labels = [], []
            for line in open(txt_path):
                if line.strip().startswith("#") or line.strip()=="":
                    continue
                else:
                    image_names.append(os.path.join(root, line.strip()))
                    labels.append(label)
            self.image_paths += image_names[: int(ratio*len(image_names))]
            self.labels += labels[: int(ratio*len(image_names))]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])[:, :, 0] # the image is in gray color and R==G==B
        label = self.labels[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return [image, label]