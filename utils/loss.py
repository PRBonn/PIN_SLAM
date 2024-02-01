#!/usr/bin/env python3
# @file      loss.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import torch
import torch.nn as nn

def sdf_diff_loss(pred, label, weight, scale=1.0, l2_loss=True):
    count = pred.shape[0]
    diff = pred - label
    diff_m = diff / scale # so it's still in m unit
    if l2_loss:
        loss = (weight * (diff_m**2)).sum() / count  # l2 loss
    else:
        loss = (weight * torch.abs(diff_m)).sum() / count  # l1 loss
    return loss

def sdf_l1_loss(pred, label):
    loss = torch.abs(pred-label)
    return loss.mean()

def sdf_l2_loss(pred, label):
    loss = (pred-label)**2
    return loss.mean()

def color_diff_loss(pred, label, weight, weighted=False, l2_loss=False):
    diff = pred - label
    if not weighted:
        weight = 1.
    else:
        weight = weight.unsqueeze(1)
    if l2_loss:
        loss = (weight * (diff**2)).mean()
    else:
        loss = (weight * torch.abs(diff)).mean()
    return loss

# used by our approach
def sdf_bce_loss(pred, label, sigma, weight, weighted=False, bce_reduction = "mean"):
    if weighted:
        loss_bce = nn.BCEWithLogitsLoss(reduction=bce_reduction, weight=weight)
    else:
        loss_bce = nn.BCEWithLogitsLoss(reduction=bce_reduction)
    label_op = torch.sigmoid(label / sigma)  # occupancy prob
    loss = loss_bce(pred / sigma, label_op)
    return loss

# the loss divised by Starry Zhong
def sdf_zhong_loss(pred, label, trunc_dist=None, weight=None, weighted = False):
    if not weighted:
        weight = 1.
    else:
        weight = weight
    loss = torch.zeros_like(label, dtype = label.dtype, device = label.device)
    middle_point = label/2.0
    middle_point_abs = torch.abs(middle_point)
    shift_difference_abs = torch.abs(pred-middle_point)
    mask = shift_difference_abs > middle_point_abs
    loss[mask] = (shift_difference_abs - middle_point_abs)[mask] # not masked region simply has a loss of zero, masked region L1 loss
    if trunc_dist is not None:
        surface_mask = (torch.abs(label) < trunc_dist)
        loss[surface_mask] = torch.abs(pred-label)[surface_mask]
    loss *= weight
    return loss.mean()

# not used
def smooth_sdf_loss(pred, label, delta = 20., weight=None, weighted = False):
    if not weighted:
        weight = 1.
    else:
        weight = weight
    sign_factors = torch.ones_like(label, dtype = label.dtype, device = label.device)
    sign_factors[label<0.0] = -1.0
    sign_loss = -sign_factors*delta*pred/2.
    no_loss = torch.zeros_like(pred, dtype = pred.dtype, device = pred.device)
    truncated_loss = sign_factors*delta*(pred/2.-label)
    losses = torch.stack((sign_loss, no_loss, truncated_loss),dim=0)
    final_loss = torch.logsumexp(losses, dim=0)
    final_loss = ((2.0/delta)*final_loss*weight).mean()
    return final_loss

def ray_estimation_loss(x, y, d_meas):  # for each ray
    # x as depth
    # y as sdf prediction
    # d_meas as measured depth

    # regard each sample as a ray
    mat_A = torch.vstack((x, torch.ones_like(x))).transpose(0, 1)
    vec_b = y.view(-1, 1)
    least_square_estimate = torch.linalg.lstsq(mat_A, vec_b).solution

    a = least_square_estimate[0]  # -> -1 (added in ekional loss term)
    b = least_square_estimate[1]

    d_estimate = torch.clamp(-b / a, min=1.0, max=40.0)  # -> d

    d_error = torch.abs(d_estimate - d_meas)

    return d_error


def ray_rendering_loss(x, y, d_meas):  # for each ray [should run in batch]
    # x as depth
    # y as occ.prob. prediction
    x = x.squeeze(1)
    sort_x, indices = torch.sort(x)
    sort_y = y[indices]

    w = torch.ones_like(y)
    for i in range(sort_x.shape[0]):
        w[i] = sort_y[i]
        for j in range(i):
            w[i] *= 1.0 - sort_y[j]

    d_render = (w * x).sum()

    d_error = torch.abs(d_render - d_meas)

    return d_error


def batch_ray_rendering_loss(x, y, d_meas, neus_on=True):  # for all rays in a batch
    # x as depth [ray number * sample number]
    # y as prediction (the alpha in volume rendering) [ray number * sample number]
    # d_meas as measured depth [ray number]
    # w as the raywise weight [ray number]
    # neus_on determine if using the loss defined in NEUS

    sort_x, indices = torch.sort(x, 1)  # for each row
    sort_y = torch.gather(y, 1, indices)  # for each row

    if neus_on:
        neus_alpha = (sort_y[:, 1:] - sort_y[:, 0:-1]) / ( 1. - sort_y[:, 0:-1] + 1e-10)
        # avoid dividing by 0 (nan)
        # print(neus_alpha)
        alpha = torch.clamp(neus_alpha, min=0.0, max=1.0)
    else:
        alpha = sort_y

    one_minus_alpha = torch.ones_like(alpha) - alpha + 1e-10

    cum_mat = torch.cumprod(one_minus_alpha, 1)

    weights = cum_mat / one_minus_alpha * alpha

    weights_x = weights * sort_x[:, 0 : alpha.shape[1]]

    d_render = torch.sum(weights_x, 1)

    d_error = torch.abs(d_render - d_meas)

    d_error_mean = torch.mean(d_error)

    return d_error_mean
