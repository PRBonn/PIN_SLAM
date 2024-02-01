#!/usr/bin/env python3
# @file      decoder.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import grad

from utils.config import Config


class Decoder(nn.Module):
    def __init__(self, config: Config, hidden_dim, hidden_level, out_dim, is_time_conditioned = False): 
        
        super().__init__()
    
        self.out_dim = out_dim

        bias_on = config.mlp_bias_on

        self.use_leaky_relu = False

        self.num_bands = config.pos_encoding_band
        self.dimensionality = config.pos_input_dim

        if config.use_gaussian_pe:
            position_dim = config.pos_input_dim + 2 * config.pos_encoding_band
        else:
            position_dim = config.pos_input_dim * (2 * config.pos_encoding_band + 1)
            
        feature_dim = config.feature_dim
        input_layer_count = feature_dim + position_dim
        
        if is_time_conditioned:
            input_layer_count += 1

        # predict sdf (now it anyway only predict sdf without further sigmoid
        # Initializa the structure of shared MLP
        layers = []
        for i in range(hidden_level):
            if i == 0:
                layers.append(nn.Linear(input_layer_count, hidden_dim, bias_on))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias_on))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(hidden_dim, out_dim, bias_on)

        if config.main_loss_type == 'bce':
            self.sdf_scale = config.logistic_gaussian_ratio*config.sigma_sigmoid_m
        else: # l1, l2 or zhong loss
            self.sdf_scale = 1.

        self.to(config.device)
        # torch.cuda.empty_cache()

    def forward(self, feature):
        # If we use BCEwithLogits loss, do not need to do sigmoid mannually
        output = self.sdf(feature)
        return output

    # predict the sdf (opposite sign to the actual sdf)
    # unit is already m
    def sdf(self, features):
        for k, l in enumerate(self.layers):
            if k == 0:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(features))
                else:
                    h = F.relu(l(features))
            else:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(h))
                else:
                    h = F.relu(l(h))

        out = self.lout(h).squeeze(1)
        out *= self.sdf_scale
        # linear (feature_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out
    
    
    def time_conditionded_sdf(self, features, ts):

        # print(ts.shape)
        nn_k = features.shape[1]
        ts_nn_k = ts.repeat(nn_k).view(-1, nn_k, 1)
        time_conditioned_feature = torch.cat((features, ts_nn_k), dim=-1)

        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(time_conditioned_feature))
            else:
                h = F.relu(l(h))

        out = self.lout(h).squeeze(1)
        out *= self.sdf_scale
        # linear (feature_dim + 1 -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out

    # predict the occupancy probability
    def occupancy(self, features):
        out = torch.sigmoid(self.sdf(features)/-self.sdf_scale)  # to [0, 1]
        return out

    # predict the probabilty of each semantic label
    def sem_label_prob(self, features):
        for k, l in enumerate(self.layers):
            if k == 0:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(features))
                else:
                    h = F.relu(l(features))
            else:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(h))
                else:
                    h = F.relu(l(h))

        out = F.log_softmax(self.lout(h), dim=-1)
        return out

    def sem_label(self, features):
        out = torch.argmax(self.sem_label_prob(features), dim=1)
        return out
    
    def regress_color(self, features):
        for k, l in enumerate(self.layers):
            if k == 0:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(features))
                else:
                    h = F.relu(l(features))
            else:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(h))
                else:
                    h = F.relu(l(h))

        out = torch.clamp(self.lout(h), 0., 1.)
        # out = torch.sigmoid(self.lout(h))
        # print(out)
        return out
