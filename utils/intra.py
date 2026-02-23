import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

BIG_NUMBER = 1e12

class Estimate_Covariance(nn.Module):

    def __init__(self, opts, class_num, feature_num=512, momentum=0.8, device=torch.device('cuda'), diag=True):
        super().__init__()

        self.distance_cache = {
            'sorted_indices': [],
            'normalized_dist': [],
            'raw_dist': [],
            'epoch': []
        }

        self.class_num = class_num
        self.feature_num = feature_num
        self.momentum = momentum
        self.device = device
        self.diag = diag

        self.register_buffer('covariance_i', torch.zeros(class_num, feature_num))
        self.register_buffer('mean_i', torch.zeros(class_num, feature_num))
        self.register_buffer('covariance_t', torch.zeros(class_num, feature_num))
        self.register_buffer('mean_t', torch.zeros(class_num, feature_num))

        self.register_buffer('amount', torch.zeros(class_num))

        self.register_buffer('global_cov_i', torch.zeros(feature_num))
        self.register_buffer('global_cov_t', torch.zeros(feature_num))

        self.register_buffer('neighbor_cov_i', torch.zeros(class_num, feature_num))
        self.register_buffer('neighbor_cov_t', torch.zeros(class_num, feature_num))

        self.register_buffer('class_weight', torch.zeros(class_num))
        self.gamma = opts.gamma
        self.num_neighbor = opts.num_neighbor
        self.beta = opts.beta

        self.reference_class_distances = {
            'class_idx': None,
            'mean_dist': [],
            'cov_dist': [],
            'class_order': []
        }

        self.distances_l2 = {
            'class_idx': None,
            'mean_dist': None,
            'cov_dist': None,
            'class_order': None
        }
        self.sigma_m = 1.0
        self.sigma_cv = 4.0

    def update_diag(self, feat_i, feat_t, labels):

        feat_i = feat_i.to(self.device)
        feat_t = feat_t.to(self.device)
        labels = labels.to(self.device)

        self.mean_i = self.mean_i.to(self.device)
        self.mean_t = self.mean_t.to(self.device)
        self.covariance_i = self.covariance_i.to(self.device)
        self.covariance_t = self.covariance_t.to(self.device)
        self.amount = self.amount.to(self.device)

        feat_i, feat_t = feat_i.detach(), feat_t.detach()

        N = feat_i.size(0)
        C = self.class_num
        A = feat_i.size(1)

        NxCxFeatures_i = feat_i.view(N, 1, A).expand(N, C, A)
        NxCxFeatures_t = feat_t.view(N, 1, A).expand(N, C, A)

        onehot = labels.to(self.device)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort_i = NxCxFeatures_i.mul(NxCxA_onehot)
        features_by_sort_t = NxCxFeatures_t.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA_i = features_by_sort_i.sum(0) / Amount_CxA
        ave_CxA_t = features_by_sort_t.sum(0) / Amount_CxA

        var_temp_i = features_by_sort_i - ave_CxA_i.expand(N, C, A).mul(NxCxA_onehot)
        var_temp_t = features_by_sort_t - ave_CxA_t.expand(N, C, A).mul(NxCxA_onehot)

        var_temp_i = var_temp_i.pow(2).sum(0).div(Amount_CxA)
        var_temp_t = var_temp_t.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        self.amount = self.amount.to(self.device)

        weight_CV = sum_weight_CV.div(sum_weight_CV + self.amount.view(C, 1).expand(C, A))
        weight_CV[weight_CV != weight_CV] = 0

        weight_CV = torch.clamp(weight_CV, min=1-self.momentum).mul((weight_CV > 0).float())

        additional_CV_i = weight_CV.mul(1 - weight_CV).mul((self.mean_i - ave_CxA_i).pow(2))
        additional_CV_t = weight_CV.mul(1 - weight_CV).mul((self.mean_t - ave_CxA_t).pow(2))

        self.covariance_i = self.covariance_i.mul(1 - weight_CV) + var_temp_i.mul(weight_CV) + additional_CV_i
        self.mean_i = self.mean_i.mul(1 - weight_CV) + ave_CxA_i.mul(weight_CV)
        self.covariance_t = self.covariance_t.mul(1 - weight_CV) + var_temp_t.mul(weight_CV) + additional_CV_t
        self.mean_t = self.mean_t.mul(1 - weight_CV) + ave_CxA_t.mul(weight_CV)
        self.amount += onehot.sum(0)

        with torch.no_grad():
            mean_dist_mat = torch.cdist(self.mean_i.unsqueeze(0), self.mean_i.unsqueeze(0), p=2).squeeze(0)
            mean_dist_mat.fill_diagonal_(float('inf'))

            min_mean_dist = mean_dist_mat.min(dim=1).values

            sorted_indices = torch.argsort(min_mean_dist, descending=True)
            sorted_dist = min_mean_dist[sorted_indices]

            min_val = sorted_dist.min()
            max_val = sorted_dist.max()
            normalized_dist = (sorted_dist - min_val) / (max_val - min_val + 1e-8)

            self.distance_cache['sorted_indices'].append(sorted_indices.cpu())
            self.distance_cache['normalized_dist'].append(normalized_dist.cpu())
            self.distance_cache['raw_dist'].append(sorted_dist.cpu())
            self.distance_cache['epoch'].append(len(self.distance_cache['epoch']))

    def update(self, feat_i, feat_t, labels):
        self.update_diag(feat_i, feat_t, labels)

    def update_each_class(self, feat_i, feat_t, labels):
        assert self.diag == True
        assert labels.shape == (len(feat_i), self.class_num)

        labels = labels.transpose(0, 1).contiguous()

        counts = labels.sum(dim=1).float()
        class_set = torch.arange(self.class_num, device=labels.device)

        valid_mask = counts > 0

        valid_classes = class_set[valid_mask]

        valid_counts = counts[valid_mask]

        class_iter = tqdm(valid_classes, ncols=60, desc='Class Estimation')
        class_alone = (valid_counts == 1).sum().item()

        def weight_functions(cnt):
            weighting = 1 / (1 + torch.log(1 + self.beta * (cnt - 1 + 1e-8)))
            return weighting * (cnt < 40).float()

        self.class_weight = weight_functions(valid_counts)

        self.mean_i = self.mean_i.to(self.device)
        self.mean_t = self.mean_t.to(self.device)
        self.covariance_i = self.covariance_i.to(self.device)
        self.covariance_t = self.covariance_t.to(self.device)
        self.amount = self.amount.to(self.device)

        for cls_idx in class_iter:
            mask = labels[cls_idx].bool()

            weights = labels[cls_idx][mask]

            cls_count = weights.sum()

            if cls_count <= 1:
                continue

            cls_feat_i = feat_i[mask]
            cls_feat_t = feat_t[mask]

            sum_w = weights.sum()

            mean_i = (cls_feat_i * weights.unsqueeze(1)).sum(0) / sum_w
            mean_t = (cls_feat_t * weights.unsqueeze(1)).sum(0) / sum_w

            var_i = ((cls_feat_i - mean_i).pow(2) * weights.unsqueeze(1)).sum(0) / sum_w
            var_t = ((cls_feat_t - mean_t).pow(2) * weights.unsqueeze(1)).sum(0) / sum_w

            alpha = cls_count / (self.amount[cls_idx] + cls_count)
            alpha = torch.clamp(alpha, min=1 - self.momentum)

            delta_mean_i = self.mean_i[cls_idx] - mean_i
            self.mean_i[cls_idx] = self.mean_i[cls_idx] * (1 - alpha) + mean_i * alpha

            self.covariance_i[cls_idx] = (self.covariance_i[cls_idx] * (1 - alpha) + var_i * alpha + alpha * (
                        1 - alpha) * delta_mean_i.pow(2))

            delta_mean_t = self.mean_t[cls_idx] - mean_t
            self.mean_t[cls_idx] = self.mean_t[cls_idx] * (1 - alpha) + mean_t * alpha
            self.covariance_t[cls_idx] = (
                    self.covariance_t[cls_idx] * (1 - alpha) +
                    var_t * alpha +
                    alpha * (1 - alpha) * delta_mean_t.pow(2))

            self.amount[cls_idx] += cls_count

        global_mask = (valid_counts > 1)
        global_weights = valid_counts[global_mask]

        self.global_covariance_i = (
                                           self.covariance_i[valid_classes][global_mask] * global_weights.unsqueeze(1)
                                   ).sum(0) / global_weights.sum()

        self.global_covariance_t = (
                                           self.covariance_t[valid_classes][global_mask] * global_weights.unsqueeze(1)
                                   ).sum(0) / global_weights.sum()

        valid_means_i = self.mean_i[valid_classes]
        valid_means_t = self.mean_t[valid_classes]
        valid_covs_i = self.covariance_i[valid_classes]
        valid_covs_t = self.covariance_t[valid_classes]

        mean_dist = torch.cdist(valid_means_i.unsqueeze(0), valid_means_t.unsqueeze(0)).squeeze()
        cov_dist = torch.cdist(valid_covs_i.unsqueeze(0), valid_covs_t.unsqueeze(0)).squeeze()

        mean_dist = (mean_dist - mean_dist.min()) / (mean_dist.max() - mean_dist.min() + 1e-8)
        cov_dist = (cov_dist - cov_dist.min()) / (cov_dist.max() - cov_dist.min() + 1e-8)

        adj_matrix = mean_dist + cov_dist

        adj_matrix[~global_mask] = float('inf')
        adj_matrix.fill_diagonal_(float('inf'))

        _, neighbors = torch.topk(
            adj_matrix,
            k=min(self.num_neighbor, len(valid_classes) - 1),
            dim=1,
            largest=False
        )

        sigma_m_tensor = torch.tensor(self.sigma_m, device=valid_means_i.device)
        sigma_cv_tensor = torch.tensor(self.sigma_cv, device=valid_means_i.device)

        for i, cls_idx in enumerate(valid_classes):
            if global_mask[i]:
                mean_dist_part = - (mean_dist[i][neighbors[i]].pow(2)) / (2 * sigma_m_tensor.pow(2))
                cov_dist_part = - (cov_dist[i][neighbors[i]].pow(2)) / (2 * sigma_cv_tensor.pow(2))
                w = valid_counts[neighbors[i]] * torch.exp(mean_dist_part + cov_dist_part)

                self.neighbor_cov_i[cls_idx] = (
                                                       valid_covs_i[neighbors[i]] * w.unsqueeze(1)
                                               ).sum(0) / w.sum()

                self.neighbor_cov_t[cls_idx] = (
                                                       valid_covs_t[neighbors[i]] * w.unsqueeze(1)
                                               ).sum(0) / w.sum()

        self.covariance_i = self.distribution_calibration_i(self.covariance_i, class_set)
        self.covariance_t = self.distribution_calibration_t(self.covariance_t, class_set)

        torch.cuda.empty_cache()

    def distribution_calibration_i(self, cv_temp, labels):

        labels = labels.long() if labels.dtype != torch.long else labels

        device = self.global_covariance_i.device

        labels = labels.to(device)
        cv_temp = cv_temp.to(device)

        gamma = self.gamma

        self.class_weight = self.class_weight.to(device)
        weight_temp = self.class_weight[labels].unsqueeze(1)

        self.neighbor_cov_i = self.neighbor_cov_i.to(device)
        cv_temp = (1-weight_temp) * cv_temp  +  weight_temp * gamma * self.global_covariance_i.expand(labels.shape[0], -1)  +  weight_temp * (1-gamma) * self.neighbor_cov_i[labels]

        return cv_temp

    def distribution_calibration_t(self, cv_temp, labels):

        labels = labels.long() if labels.dtype != torch.long else labels

        gamma = self.gamma
        weight_temp = self.class_weight[labels].unsqueeze(1)

        self.neighbor_cov_t = self.neighbor_cov_t.to(self.device)
        cv_temp = (1-weight_temp) * cv_temp  +  weight_temp * gamma * self.global_covariance_t.expand(labels.shape[0], -1)  +  weight_temp * (1-gamma) * self.neighbor_cov_t[labels]

        return cv_temp

    def global_update(self, model, loader_train_eval):

        img_hash_all, text_hash_all, labels_all = [], [], []
        test_iter = tqdm(loader_train_eval, ncols=40)
        test_iter.set_description('Global Estimation')
        model = model.to(self.device)

        with torch.no_grad():
            for image, text, label, index in test_iter:
                image, text, label = image.to(self.device), text.to(self.device), label.to(self.device)
                img_hash, text_hash = model(image, text)
                img_hash_all.append(img_hash.data)
                text_hash_all.append(text_hash.data)
                labels_all.append(label.data)

            img_hash_all = torch.cat(img_hash_all)
            text_hash_all = torch.cat(text_hash_all)
            labels_all = torch.cat(labels_all)

            self.update_each_class(img_hash_all, text_hash_all, labels_all)
            # self.update(img_hash_all, text_hash_all, labels_all)

def intra_synthetsis(feat_i, feat_t, labels, estimator, lamda=0.2, aug_num=1, detach=False, diag=True):

    if detach:
        feat_i = feat_i.detach()
        feat_t = feat_t.detach()

    features_aug_i, features_aug_t = [], []
    B, C = labels.shape

    device = estimator.covariance_i.device
    labels = labels.to(device)
    feat_i = feat_i.to(device)
    feat_t = feat_t.to(device)

    cv_temp_i, cv_temp_t = [], []
    for i in range(B):
        active_classes = torch.where(labels[i])[0]

        if len(active_classes) == 0:
            cv_i = estimator.global_cov_i
            cv_t = estimator.global_cov_t
        else:
            cv_i = estimator.covariance_i[active_classes].mean(dim=0)
            cv_t = estimator.covariance_t[active_classes].mean(dim=0)

        cv_temp_i.append(cv_i)
        cv_temp_t.append(cv_t)

    cv_temp_i = torch.stack(cv_temp_i, dim=0)
    cv_temp_t = torch.stack(cv_temp_t, dim=0)

    lamda = torch.tensor(lamda, device=cv_temp_i.device).sqrt()

    for i in range(aug_num):
        cv_temp_i = cv_temp_i.sqrt()
        aug_i = feat_i + lamda * cv_temp_i.mul(torch.randn_like(feat_i))
        features_aug_i.append(aug_i)

        cv_temp_t = cv_temp_t.sqrt()
        aug_t = feat_t + lamda * cv_temp_t.mul(torch.randn_like(feat_t))
        features_aug_t.append(aug_t)

    features_aug_i = torch.cat(features_aug_i, dim=0)
    features_aug_t = torch.cat(features_aug_t, dim=0)
    labels_aug = labels.repeat(aug_num, 1)

    return features_aug_i, features_aug_t, labels_aug

def lamda_epoch(lamda, epochs, ep):
    
    x1 = 0.4
    x2 = 0.6
    x3 = 0.8    

    y1 = 1.0
    y2 = 0.9
    y3 = 0.75
    y4 = 0.6

    radio = ep / epochs

    if radio <= x1:
        weight = y1
    elif radio > x1 and radio <= x2:
        weight = y2
    elif radio > x2 and radio <= x3:
        weight = y3
    else:
        weight = y4
    
    return weight * lamda


def maxmin_norm(x):

    return (x - x.min()) / (x.max() - x.min())

