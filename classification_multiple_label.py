
# coding: utf-8

# In[1]:


import os, pdb, cv2, random, datetime, shutil, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from pprint import pprint, pformat
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

from dataloaders.Classification_Image import OCT_image_classification, OCT_image_multi_classification
from dataloaders.Image_transforms import Resize, Split_h, Normalize_divide, Random_flip, Rotation, Translation, Shear    

import utils, metrics
from networks.classification.ResNet import ResNet34,ResNet50,ResNet101
from networks.classification.ResNet_original import ResNet18_original, ResNet34_original, ResNet50_original
from networks.classification.DenseNet_original import DenseNet121_original, DenseNet169_original
from utils import aic_fundus_lesion_classification

def log_best_metric(metric_list, cur_epoch_idx, logger, state, save_path, save_model=True, metric = "AUC"):
    if len(metric_list) == 0:
        return
    else:
        if save_model:
            dir_path = os.path.dirname(save_path)  # get parent path
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            torch.save(state, save_path)
            logger.info("Model saved in file: %s"%(save_path))
        best_idx = np.argmax(metric_list)
        best_metric = metric_list[best_idx]
        if best_idx == cur_epoch_idx:
            logger.info("Epoch: %d, Validation %s improved to %.4f"%(cur_epoch_idx, metric, best_metric))
        else:
            logger.info("Epoch: %d, Validation %s didn't improve. Best is %.4f in epoch %d"%(cur_epoch_idx, metric, best_metric, best_idx))


# In[2]:


class Config(object):
    def __init__(self):
        
        self.multiple_label = {"name": "multiple_label",
                        "included_pixels": [0, 255, 191, 128],
                        "label_dict": OrderedDict([(255, 0), (191, 1), (128, 2)]), # the value represent the position of 1. OrderedDict([(255, 0), (191, 1), (128, 2)])
                        "aug_label_dict": OrderedDict([(128, 10)])}
                        
        self.batch_size = 64
        self.num_classes = 3
        
        self.num_split = 3
        self.start_h = 50
        
        self.target_h = 224
        self.target_w = 224
        
        self.task = self.multiple_label
        self.network = "DenseNet169_original"
        self.net_config = None
        self.lr = 1.0
        self.suffix = "aug_multipleLabel3" # aug_oversample_includeNormal
        self.checkpoint = None
        
        self.gpus = "3"
        self.num_workers = 1
        self.nepoch = 200
        self.manualSeed = None
        

config = Config()        


# In[3]:


log_path = os.path.join('logs', config.task['name'], config.network, '{}.log'.format(config.suffix))
if os.path.exists(log_path):
    delete_log = raw_input("The log file %s exist, delete it or not (y/n) \n"%(log_path))
    if delete_log in ['y', 'Y']:
        os.remove(log_path)
    else:
        log_path = os.path.join('logs', config.task['name'], config.network, '{}_{}.log'.format(config.suffix, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))

checkpoint_path = os.path.join('checkpoint', config.task['name'], config.network, config.suffix)
if os.path.exists(checkpoint_path):
    delete_checkpoint_path = raw_input("The checkpoint folder %s exist, delete it or not (y/n) \n"%(checkpoint_path))
    if delete_checkpoint_path in ['y', 'Y']:
        shutil.rmtree(checkpoint_path)
    else:
        checkpoint_path = os.path.join("checkpoint", config.task['name'], config.network, config.suffix+"_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
else:
    os.makedirs(checkpoint_path)

summary_path = os.path.join("summaries", config.task['name'], config.network, config.suffix)
if os.path.exists(summary_path):
    delete_summary = raw_input("The tf_summary folder %s exist, delete it or not (y/n) \n"%(summary_path))
    if delete_summary in ['y', 'Y']:
        shutil.rmtree(summary_path)
    else:
        summary_path = os.path.join("summaries", config.task['name'], config.network, config.suffix+"_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
else:
    os.makedirs(summary_path)
    

logger = utils.get_logger(log_path)
writer = SummaryWriter(summary_path)
logger.info(config.__dict__)

if config.manualSeed is None:
    config.manualSeed = random.randint(1, 10000)
logger.info("Random Seed: {}".format(config.manualSeed))
np.random.seed(config.manualSeed)
random.seed(config.manualSeed)
torch.manual_seed(config.manualSeed)


# In[4]:

train_tr = transforms.Compose([
                     Resize((config.target_h, config.target_w)),
                     Random_flip(1),
                     Rotation(20),
                     Translation(50),
                     Shear(2),
                     Normalize_divide(255.0)])
val_tr = transforms.Compose([
                     Resize((config.target_h, config.target_w)),
                     Normalize_divide(255.0)])
    

trainset = OCT_image_multi_classification("./data/Edema_trainingset/original_images", 
                     "./data/Edema_trainingset/label_images",
                     included_pixels = config.task["included_pixels"],
                     label_dict = config.task["label_dict"], 
                     aug_dict = config.task["aug_label_dict"],
                     num_classes = config.num_classes,
                     transform = train_tr)

valset = OCT_image_multi_classification("./data/Edema_validationset/original_images", 
                     "./data/Edema_validationset/label_images",
                     included_pixels = config.task["included_pixels"], 
                     label_dict = config.task["label_dict"], 
                     aug_dict = None,
                     num_classes = config.num_classes,
                     transform = val_tr)

trainset_loader = torch.utils.data.DataLoader(trainset, batch_size = config.batch_size,
                                         shuffle=True, num_workers=config.num_workers)

valset_loader = torch.utils.data.DataLoader(valset, batch_size = config.batch_size,
                                         shuffle=False, num_workers=config.num_workers)


# In[5]:


'''get recall, precision and f1
'''
def statistic(pred_logits, y_logits, label_size=2):
    confusion_mat = np.zeros((label_size, label_size))
    for i, sample in enumerate(pred_logits):
        confusion_mat[y_logits[i], pred_logits[i]] += 1
    c = metrics.Confusion(confusion_mat)
    acc = c.accuracy()
    recall_list = c.recall()
    precision_list = c.precision()
    f1_list = c.f1()      
    return confusion_mat, recall_list, precision_list, f1_list

def train(model, device, data_loader, criterion, optimizer, epoch, writer):
    model.train()
    correct, total = 0, 0
    epoch_loss, epoch_outputs, epoch_labels = [], [], []
    predicted_correct = np.zeros(3)
    with tqdm(len(data_loader)) as pbar:
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            
            predicted_bool = (torch.sigmoid(outputs) >= 0.5).float() == labels
            predicted_correct = predicted_correct + torch.sum(predicted_bool, 0).cpu().numpy()
            
            epoch_outputs.append(outputs.detach().cpu().numpy())
            epoch_labels.append(labels.detach().cpu().numpy())
            
            total += labels.size(0)
            pbar.update(1)
            pbar.set_description("Epoch %d, Batch %d/%d, Train loss: %.4f, Train acc: %s"%(epoch, 
                                                                                               batch_idx+1, len(data_loader),                                                                                                np.mean(epoch_loss), predicted_correct.astype(np.float32)/total))
    
    loss = np.mean(epoch_loss)
    epoch_outputs = np.concatenate(epoch_outputs, 0)
    epoch_labels = np.concatenate(epoch_labels, 0)
    class_aucs = []
    for i in range(3):
        fpr, tpr, thresholds = roc_curve(y_true = epoch_labels[:, i], y_score = epoch_outputs[:, i])
        auc_value = auc(fpr, tpr)
        class_aucs.append(auc_value)
    writer.add_scalar('train/epoch_loss', loss, epoch)
    return class_aucs, loss

def validate(model, device, data_loader, num_classes, criterion, epoch, writer):
    model.eval()
    correct, total = 0, 0
    epoch_loss, epoch_outputs, epoch_labels, outputs_softmax = [], [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(data_loader)):
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            epoch_loss.append(loss.detach())
                    
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            epoch_outputs.append(outputs.detach().cpu().numpy())
            epoch_labels.append(labels.detach().cpu().numpy())
    epoch_outputs = np.concatenate(epoch_outputs, 0)
    epoch_labels = np.concatenate(epoch_labels, 0)
    class_mean_aucs = mean_auc_persample(epoch_outputs, epoch_labels)
    loss = np.mean(epoch_loss)
    writer.add_scalar('validation/epoch_loss', loss, epoch)
    
    return class_mean_aucs, loss

def mean_auc_persample(all_outputs, all_labels, num_image = 128):
    sample_aucs = []
    outputs, labels = [], []
    for i in range(len(all_outputs)):
        outputs.append(all_outputs[i])
        labels.append(all_labels[i])
        if (i+1) % 128 == 0:
            sample_aucs.append(aic_fundus_lesion_classification(np.array(labels), np.array(outputs)))
            outputs, labels = [], []
    valid_aucs = [[], [], []]
    for sample_auc in sample_aucs:
        for i, auc_value in enumerate(sample_auc):
            if not math.isnan(auc_value):
                valid_aucs[i].append(auc_value)
    return [np.mean(auc_values) for auc_values in valid_aucs]
        
        

# In[6]:


model = globals()[config.network](config.net_config, config.num_classes)

if config.checkpoint != None and config.checkpoint != "":
    logger.info("load checkpoint: {}".format(config.checkpoint))
    model.load_state_dict(torch.load(config.checkpoint))

gpus = map(int, config.gpus.split(","))
if len(gpus) > 1:
    model = nn.DataParallel(model, gpus)
device = torch.device("cuda:{}".format(gpus[0]))
model.to(device)

# In[7]:

metric_list = []
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adadelta(model.parameters(), lr = config.lr)
for epoch in range(config.nepoch):
    class_aucs, train_loss = train(model, device, trainset_loader, criterion, optimizer, epoch, writer)
    logger.info("Epoch: %d, Train Loss: %.4f, Train AUC: %s"%(epoch, train_loss, str(class_aucs)))
    class_mean_aucs, val_loss = validate(model, device, valset_loader, int(config.num_classes), criterion, epoch, writer)
    metric_list.append(np.mean(class_mean_aucs))
    logger.info("Epoch: %d, Validation Loss: %.4f, Validation AUCs: %s"%(epoch, val_loss, str(class_mean_aucs)))
    
    log_best_metric(metric_list, epoch, logger, 
                  model.state_dict(),
                  "{}/epoch{}.pth".format(checkpoint_path, epoch),
                  save_model=True,
                  metric = "AUC")

