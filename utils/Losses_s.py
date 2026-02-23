from __future__ import absolute_import

from argparse import Namespace

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import init

import math
from torch import nn

from torch.nn.functional import normalize

def cos_distance(source, target):
    cos_sim = F.cosine_similarity(source.unsqueeze(1), target, dim=-1)
    distances = torch.clamp(1 - cos_sim, 0)
    return distances


def get_triplet_mask(s_labels, t_labels, beta, gamma):
    flag = (beta - 0.1) * gamma
    batch_size = s_labels.shape[0]
    sim_origin = s_labels.mm(t_labels.t())
    sim = (sim_origin > 0).float()
    ideal_list = torch.sort(sim_origin, dim=1, descending=True)[0]
    ph = torch.arange(0., batch_size) + 2
    ph = ph.repeat(1, batch_size).reshape(batch_size, batch_size)
    th = torch.log2(ph).to(1)
    Z = (((2 ** ideal_list - 1) / th).sum(axis=1)).reshape(-1, 1)
    sim_origin = 2 ** sim_origin - 1
    sim_origin = sim_origin / Z

    i_equal_j = sim.unsqueeze(2)
    i_equal_k = sim.unsqueeze(1)
    sim_pos = sim_origin.unsqueeze(2)
    sim_neg = sim_origin.unsqueeze(1)
    weight = (sim_pos - sim_neg) * (flag + 0.1)
    mask = i_equal_j * (1 - i_equal_k) * (flag + 0.1)

    return mask, weight


class TripletLoss3(nn.Module):
    def __init__(self, opt, reduction='mean'):
        super(TripletLoss3, self).__init__()
        self.reduction = reduction
        self.opt = opt
        self.beta = 0.1
        self.gamma = 10
        self.beta_2 = 0.1
        self.mu = 0.00001
        self.lamb = 1
        self.alpha = 10

    def forward(self, source, s_labels, target=None, t_labels=None, margin=0.05):
        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels
        pairwise_dist = cos_distance(source, target)

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        mask, weight = get_triplet_mask(s_labels, t_labels, self.beta, self.gamma)
        if self.alpha == 10:
            triplet_loss = 10 * weight * mask * triplet_loss
        else:
            triplet_loss = mask * triplet_loss

        triplet_loss = triplet_loss.clamp(0)

        valid_triplets = triplet_loss.gt(1e-16).float()
        num_positive_triplets = valid_triplets.sum()

        if self.reduction == 'mean':
            triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        elif self.reduction == 'sum':
            triplet_loss = triplet_loss.sum()

        return triplet_loss
