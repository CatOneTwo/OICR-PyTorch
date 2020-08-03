# OICR output module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tasks.config import cfg

import numpy as np
from pdb import set_trace as pause

from torchvision.ops.boxes import box_iou

def OICR(boxes, cls_prob, im_labels, lambda_gt=0.5):


    im_labels    = im_labels.long()

    cls_prob     = cls_prob.clone().detach()

    # if cls_prob have the background dimenssion, we cut it out
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    

    #  avoiding NaNs.
    eps = 1e-9
    cls_prob = cls_prob.clamp(eps, 1 - eps)

    num_images, num_classes = im_labels.shape

    #
    max_values, max_indexes = cls_prob.max(dim=0) # select the proposal with highest score from (k-1)th OICR

    gt_boxes   = boxes[max_indexes, :][im_labels[0]==1,:] # treat the proposal with highest score as ground truth
    gt_classes = torch.arange(num_classes)[im_labels[0]==1].view(-1,1) + 1
    gt_scores  = max_values[im_labels[0]==1].view(-1,1)

    overlaps = box_iou(boxes, gt_boxes)  # calculate IOU

    max_overlaps, gt_assignment = overlaps.max(dim=1)  # max IOU and Index

    labels = gt_classes[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    
    bg_inds = torch.where(max_overlaps < lambda_gt)[0]
    labels[bg_inds] = 0
    gt_assignment[bg_inds] = -1


    return {'labels' : labels.reshape(1, -1),
            'cls_loss_weights' : cls_loss_weights.reshape(1, -1),
            'gt_assignment' : gt_assignment.reshape(1, -1),
            'im_labels_real' : torch.cat((torch.tensor([[1]]).cuda(), im_labels), dim=1)}

