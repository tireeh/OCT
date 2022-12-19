
# coding: utf-8

# In[1]:


import os, pdb, cv2, random, datetime, shutil
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

from dataloaders.Classification_Image import OCT_image_classification, Imageset
from dataloaders.Image_transforms import Resize, Split_h, Normalize_divide, Random_flip, Rotation, Translation, Shear, Expand_channel    

import utils, metrics
from networks.classification.ResNet import ResNet34,ResNet50
from networks.classification.ResNet_original import ResNet34_original, ResNet50_original
from networks.classification.DenseNet_original import DenseNet121_original
from networks.classification.ResNet_pretrain import ResNet101_pretrain, ResNet50_pretrain
from networks.classification.Xception_pretrain import Xception_pretrain


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
        self.normal_edema = {"name": "normal_edema",
                             "included_pixels": [0, 255, 191, 128], 
                             "label_dict": OrderedDict([(0, 0), (255, 1), (191, 1), (128, 1)]),
                             "aug_label_dict": None}
        
        self.edema_SRF = {"name": "edema_SRF",
                          "included_pixels": [0, 255, 191, 128], 
                          "label_dict": OrderedDict([(0, 0), (255, 0), (128, 0), (191, 1)]),
                          "aug_label_dict": None}
        
        self.edema_PED = {"name": "edema_PED",
                          "included_pixels": [0, 255, 191, 128], 
                          "label_dict": OrderedDict([(0, 0), (255, 0), (191, 0), (128, 1)]), 
                          "aug_label_dict": OrderedDict([(1, 10)])}
        
        self.resnet_aggregate_config = {"ensemble_idx": 7, "feature_size": 7}
        
        self.batch_size = 256
        self.num_classes = 2
        
        self.num_split = 3
        self.start_h = 50
        
        self.target_h = 224
        self.target_w = 224
        
        self.root_task = "OCT_pretrain"
        
        self.task = self.edema_PED
        self.network = "ResNet50_pretrain"
        self.pretrain_checkpoint = "/root/.torch/models/resnet50-19c8e357.pth"
        self.optimizer = "sgd" # sgd | adadelta
        self.lr = 0.001
        
        self.net_config = self.resnet_aggregate_config
        
        self.suffix = "aug_sgd" # aug_oversample_includeNormal
        self.checkpoint = None
        
        self.gpus = "0, 1, 2, 3"
        self.num_workers = 6
        self.nepoch = 200
        self.manualSeed = None
        

config = Config()        


# In[3]:


log_path = os.path.join('logs', config.root_task, config.network, '{}.log'.format(config.suffix))
if os.path.exists(log_path):
    delete_log = raw_input("The log file %s exist, delete it or not (y/n) \n"%(log_path))
    if delete_log in ['y', 'Y']:
        os.remove(log_path)
    else:
        log_path = os.path.join('logs', config.root_task, config.network, '{}_{}.log'.format(config.suffix, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))

checkpoint_path = os.path.join('checkpoint', config.root_task, config.network, config.suffix)
if os.path.exists(checkpoint_path):
    delete_checkpoint_path = raw_input("The checkpoint folder %s exist, delete it or not (y/n) \n"%(checkpoint_path))
    if delete_checkpoint_path in ['y', 'Y']:
        shutil.rmtree(checkpoint_path)
    else:
        checkpoint_path = os.path.join("checkpoint", config.root_task, config.network, config.suffix+"_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
else:
    os.makedirs(checkpoint_path)

summary_path = os.path.join("summaries", config.root_task, config.network, config.suffix)
if os.path.exists(summary_path):
    delete_summary = raw_input("The tf_summary folder %s exist, delete it or not (y/n) \n"%(summary_path))
    if delete_summary in ['y', 'Y']:
        shutil.rmtree(summary_path)
    else:
        summary_path = os.path.join("summaries", config.root_task, config.network, config.suffix+"_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
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


if config.root_task == "OCT_pretrain":
    train_tr = transforms.Compose([
                     Resize((config.target_h, config.target_w)),
                     Random_flip(1),
                     Rotation(20),
                     Translation(50),
                     Shear(2),
                     Expand_channel(0),
                     Normalize_divide(255.0)])

    val_tr = transforms.Compose([
                         Resize((config.target_h, config.target_w)),
                         Expand_channel(0),
                         Normalize_divide(255.0)])

    trainset = Imageset({"./data_list/Cell_OCT/train_NORMAL.txt": ["/root/workspace/datasets/OCT/train/NORMAL/", 0, 0.22], 
                         "./data_list/Cell_OCT/train_DME.txt": ["/root/workspace/datasets/OCT/train/DME/", 1, 1.0],
                         "./data_list/AIChallenger/train_normal.txt": ["/root/workspace/AIChallenger_OCT/data/Edema_trainingset/original_images", 0, 1.0],
                         "./data_list/AIChallenger/train_abnormal.txt": ["/root/workspace/AIChallenger_OCT/data/Edema_trainingset/original_images", 1, 1.0]
                        }, train_tr)

    valset = Imageset({"./data_list/Cell_OCT/test_NORMAL.txt": ["/root/workspace/datasets/OCT/test/NORMAL/", 0, 1.0], 
                         "./data_list/Cell_OCT/test_DME.txt": ["/root/workspace/datasets/OCT/test/DME/", 1, 1.0],
                         "./data_list/AIChallenger/val_normal.txt": ["/root/workspace/AIChallenger_OCT/data/Edema_validationset/original_images", 0, 1.0],
                         "./data_list/AIChallenger/val_abnormal.txt": ["/root/workspace/AIChallenger_OCT/data/Edema_validationset/original_images", 1, 1.0]
                        }, val_tr)

else:
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

    trainset = OCT_image_classification("./data/Edema_trainingset/original_images", 
                         "./data/Edema_trainingset/label_images",
                         included_pixels = config.task["included_pixels"],
                         label_dict = config.task["label_dict"], 
                         aug_label_dict = config.task["aug_label_dict"],
                         transform = train_tr)

    valset = OCT_image_classification("./data/Edema_validationset/original_images", 
                         "./data/Edema_validationset/label_images",
                         included_pixels = config.task["included_pixels"], 
                         label_dict = config.task["label_dict"], 
                         aug_label_dict = None,
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
    epoch_loss = []
    with tqdm(len(data_loader)) as pbar:
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            _, predicted = torch.max(outputs.detach(), 1)
            total += labels.size(0)
            
            correct += torch.sum(predicted.detach() == labels.detach())
            pbar.update(1)
            pbar.set_description("Epoch %d, Batch %d/%d, Train loss: %.4f, Train acc: %.4f"%(epoch, 
                                                                                               batch_idx+1, len(data_loader), 
                                                                                               np.mean(epoch_loss), float(correct)/total))
    accuracy = float(correct) / float(total)
    loss = np.mean(epoch_loss)
    writer.add_scalar('train/epoch_accuracy', accuracy, epoch)
    writer.add_scalar('train/epoch_loss', loss, epoch)
    return accuracy, loss

def validate(model, device, data_loader, num_classes, criterion, epoch, writer):
    model.eval()
    correct, total = 0, 0
    epoch_loss, epoch_logits, y_logits, outputs_softmax = [], [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(data_loader)):
            inputs = inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            outputs_softmax = outputs_softmax + [softmax[1] for softmax in F.softmax(outputs, dim=1).detach().cpu().numpy()]
            
            epoch_loss.append(loss.detach())
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += torch.sum(predicted == labels)

            epoch_logits = epoch_logits + [idx for idx in predicted]
            y_logits = y_logits + [idx for idx in labels]
    
    fpr, tpr, thresholds = roc_curve(y_true = y_logits, y_score = outputs_softmax)
    auc_value = auc(fpr, tpr)
    confusion_mat, recall, precision, f1 = statistic(epoch_logits, y_logits, num_classes)
    logger.info("Confusion matrix: \n %s \n Recall: %s \n Precision: %s \n F1: %s, \n AUC: %.4f"%(str(confusion_mat),                                                                         str(recall),str(precision), str(f1), float(auc_value)))
    accuracy = float(correct) / float(total)
    loss = np.mean(epoch_loss)
    
    writer.add_scalar('validation/epoch_accuracy', accuracy, epoch)
    writer.add_scalar('validation/epoch_loss', loss, epoch)
    
    return accuracy, auc_value, loss


# In[6]:


model = globals()[config.network](num_classes = config.num_classes, checkpoint = config.pretrain_checkpoint)

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
criterion = nn.CrossEntropyLoss()

if config.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0001)
elif config.optimizer == "adadelta":
    optimizer = optim.Adadelta(model.parameters(), lr = config.lr)
else:
    raise("Unknown optimizer: {}".format(config.optimizer))
    
for epoch in range(config.nepoch):
    train_acc, train_loss = train(model, device, trainset_loader, criterion, optimizer, epoch, writer)
    logger.info("Epoch: %d, Train Loss: %.4f, Train Acc: %.4f"%(epoch, train_loss, train_acc))
    val_acc, val_auc, val_loss = validate(model, device, valset_loader, int(config.num_classes), criterion, epoch, writer)
    metric_list.append(val_auc)
    logger.info("Epoch: %d, Validation Loss: %.4f, Validation Acc: %.4f, Validation AUC: %.4f"%(epoch, val_loss, val_acc, val_auc))
    
    log_best_metric(metric_list, epoch, logger, 
                  model.state_dict(),
                  "{}/epoch{}.pth".format(checkpoint_path, epoch),
                  save_model=True,
                  metric = "AUC")

