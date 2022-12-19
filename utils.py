import os, logging, traceback
import torch
import torch.nn as nn
from sklearn import metrics

def get_logger(log_path):
    parent_path = os.path.dirname(log_path)  # get parent path
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(level=logging.INFO,
                    filename=log_path,
                    format='%(levelname)s:%(name)s:%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(console)
    return logger


def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)

def get_aucs(ground_truths, predictions):
    """get auc value per sample"""
    aucs, valid_aucs = [], []
    for i in range(len(ground_truths)):
        type_aucs = aic_fundus_lesion_classification(ground_truths[i], predictions[i])
        aucs.append(type_aucs)
        for type_auc in type_aucs:
            if not math.isnan(type_auc):
                valid_aucs.append(type_auc)  
    return aucs, np.mean(valid_aucs)


def aic_fundus_lesion_classification(ground_truth, prediction, num_samples=128):
    """
    Classification task auc metrics.
    :param ground_truth: numpy matrix, (num_samples, 3)
    :param prediction: numpy matrix, (num_samples, 3)
    :param num_samples: int, default 128
    :return list:[AUC_1, AUC_2, AUC_3]
    """
    assert (ground_truth.shape == (num_samples, 3))
    assert (prediction.shape == (num_samples, 3))

    try:
        ret = [0.5, 0.5, 0.5]
        for i in range(3):
            fpr, tpr, thresholds = metrics.roc_curve(ground_truth[:, i], prediction[:, i], pos_label=1)
            ret[i] = metrics.auc(fpr, tpr)
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret

def get_dice_score(ground_truths, predictions):
    """get dice score per sample"""
    dice_scores, valid_scores = [], []
    for i in range(len(ground_truths)):
        type_scores = aic_fundus_lesion_segmentation(ground_truths[i], predictions[i])
        dice_scores.append(type_scores)
        for type_score in type_scores:
            if not math.isnan(type_score):
                valid_scores.append(type_score)  
    return dice_scores, np.mean(valid_scores)

def aic_fundus_lesion_segmentation(ground_truth, prediction, num_samples=128):
    """
    Detection task auc metrics.
    :param ground_truth: numpy matrix, (num_samples, 1024, 512)
    :param prediction: numpy matrix, (num_samples, 1024, 512)
    :param num_samples: int, default 128
    :return list:[Dice_0, Dice_1, Dice_2, Dice_3][background,REA,SRF,PED]
    """
#     assert (ground_truth.shape == (num_samples, 1024, 512))
#     assert (prediction.shape == (num_samples, 1024, 512))

    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    try:
        ret = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            mask1 = (ground_truth == i)
            mask2 = (prediction == i)
            if mask1.sum() != 0:
                ret[i] = float(2 * ((mask1 * (ground_truth == prediction)).sum())) / (mask1.sum() + mask2.sum())
            else:
                ret[i] = float('nan')
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret
