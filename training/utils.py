import os
import torch
import numpy as np
from imgaug import augmenters as iaa
import torchvision
import random
import PIL.Image as Image
import cv2
import math

from blindMatch import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from os import path
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Grayscale
from sklearn.metrics import roc_curve, auc, make_scorer
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch.nn.functional as F
import torch.nn as nn


##########################################################
class TestPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairlist, degree=0, size=224):
        self.pairlist=pairlist
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(225),
            torchvision.transforms.Resize((size, size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation(degree),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
        ])

    def __len__(self):
        return len(self.pairlist)

    def __getitem__(self, idx):
        data1= self.pairlist[idx]
        pil1 = Image.fromarray(cv2.imread(data1[0]))
        file1 = self.transform(pil1)
        
        pil2 = Image.fromarray(cv2.imread(data1[1]))
        file2 = self.transform(pil2)
        
        y = float(data1[2])
        return file1, file2, y
  
    
class TrainPairDataset(torch.utils.data.Dataset):
    def __init__(self, pospairlist, negpairlist, degree=0, size=224):
        self.pospairlist=pospairlist
        
        len_pos = len(pospairlist)
        self.negpairlist = negpairlist[:len_pos]
        self.pairlist = np.concatenate((self.pospairlist, self.negpairlist))
        # np.random.shuffle(self.pairlist)
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(225),
            torchvision.transforms.Resize((size, size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation(degree),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
        ])


    def __len__(self):
        return len(self.pairlist)

    def __getitem__(self, idx):
        data1= self.pairlist[idx]
        pil1 = Image.fromarray(cv2.imread(data1[0]))
        file1 = self.transform(pil1)
        
        pil2 = Image.fromarray(cv2.imread(data1[1]))
        file2 = self.transform(pil2)
        
        y = float(data1[2])
        return file1, file2, y
    
class FingerprintDataset(torch.utils.data.Dataset):
    def __init__(self, trainlist, degree=0, size=224):
        self.trainlist=trainlist

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(225),
            torchvision.transforms.Resize((size, size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation(degree),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
        ])

    def __len__(self):
        return len(self.trainlist)

    def __getitem__(self, idx):
        entry = self.trainlist[idx]
        file = Image.fromarray(cv2.imread(entry[0]))
        file = self.transform(file)

        n_id = entry[1]

        return file, np.array(n_id, dtype=np.int64)

##########################################################
#
#
#
#
#
##########################################################
class CustomBlindTouch(torch.nn.Module):
    def __init__(self, num_identities):
        """
        Please refer to Blind Touch
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU(True),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(2),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(True),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(2),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=2),
            nn.BatchNorm2d(128),
            nn.SiLU(True),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(2),
            )        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=2),
            nn.BatchNorm2d(256),
            nn.SiLU(True),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(2),
            )        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=2),
            nn.BatchNorm2d(512),
            nn.SiLU(True),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(2),
            )        
        
        self.fc_identity1 = nn.Linear(32768, 16)
        self.fc_identity2 = nn.Linear(16, num_identities)

        # Define the softmax layer
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x1_net = self.conv1(x)
        x1_net = self.conv2(x1_net)
        x1_net = self.conv3(x1_net)
        x1_net = self.conv4(x1_net)
        x1_net = self.conv5(x1_net)
        
        x = torch.flatten(x1_net, 1)
        x = torch.nn.functional.normalize(x)
        x = self.fc_identity1(x)
        # net = torch.square(net)
        # net = self.fc2(net)
        # return torch.sigmoid(net)
        return x

##########################################################
#
#
#
#
#
##########################################################
def find_best_threshold(scores, labels, thresholds, pos_label=1):
    labels = np.array(labels)
    scores = np.array(scores)
    
    best_acc, best_ms = 0, [0,0,0]
    best_thresh = None
    for i, thresh in enumerate(thresholds):
        # print(thresh)
        # compute accuracy for this threshold
        pred_labels = [1 if s >= thresh else 0 for s in scores]
        acc = sum([1 if pred_labels[j] == labels[j] else 0 for j in range(len(labels))]) / len(labels)
        true_positives = sum([1 if (pred_labels[j] == labels[j])&(labels[j] == pos_label) else 0 for j in range(len(labels))])
        predicted_positives = sum([pos_label if pred_labels[j] == labels[j] else 1-pos_label for j in range(len(labels))])
        possible_positives = sum(labels == pos_label)
        precision = true_positives / (predicted_positives + 1e-8)  
        recall = true_positives / (possible_positives + 1e-8)      
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)  

        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            best_ms = [precision, recall, f1_score]

    return best_thresh, best_acc, best_ms


# 1-FPIR ~= FA = FM  = False Match     - same but predicted as diff
# FNIR ~= FR = FNM = False Non-match - diff but predicted as same
def eer_from_fpr_tpr(fpr, tpr, thresholds):
    '''
    Equal-Error Rate (EER)  : it denotes the error rate at the threshold t 
                            for which both false match rate and false non-match rate are identical
                            : FMR(t) = FNMR(t)
                            
    Although EER is an important indicator, a fingerprint system is
    rarely used at the operating point corresponding to EER (because the corresponding
    FMR is often not sufficiently low for security requirements), and often a more stringent
    threshold is set corresponding to a pre-specified value of FMR.
    =================================================================From FVC2004 paper.
    '''
    eer = 1.0 
    for i in range(len(thresholds)):
        if fpr[i] >= 1 - tpr[i]:
            eer = min(eer, (fpr[i] + (1 - tpr[i])) / 2)
    return eer


def eval_state(probs, labels, thr):
    labels = np.array(labels)
    probs = np.array(probs)

    predict = probs >= thr
    labels = np.array(labels)
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP


def auc_loop(scores, labels, pos_label=1.0):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=pos_label)
    # print(thresholds)
    th, acc, metrics = find_best_threshold(scores, labels, thresholds=thresholds, pos_label=pos_label)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    auc_score = auc(fpr, tpr)
    TN, FN, FP, TP = eval_state(scores, labels, th)
    TPR = TP / float(TP + FN + 1e-8)
    FRR = FN / float(FN + TP + 1e-8)
    print(f"At threshold = {th}", TN, FN, FP, TP, len(labels))
    return thresholds, th, auc_score, eer, acc, metrics, FRR, TPR


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances, labels):
        loss_contrastive = torch.mean((1-labels) * torch.pow(distances, 2) +
                                      (labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2))
        return loss_contrastive
    
def contrastive_loss(y_true, distances, margin=1):
    """
    Calculates the contrastive loss.

    Arguments:
        y_true: List of labels, each label is of type float32.
        y_pred: List of predictions of same length as of y_true,
                each label is of type float32.

    Returns:
        A tensor containing contrastive loss as floating point value.
    """
    square_pred = torch.square(distances)
    margin_square = torch.square(torch.clamp(margin - distances, min=0.0))
    return torch.mean(
        (1 - y_true) * square_pred + (y_true) * margin_square
    )
