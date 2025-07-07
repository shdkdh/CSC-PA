import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import math
import time

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

def get_prototype(x, ss_map):
    B, _, H, W = x.size()

    ss_map = ss_map.view(B, -1, H * W)
    x = x.view(B, -1, H * W)

    prototype_block = torch.bmm(ss_map, x.transpose(1, 2))
    return prototype_block

def get_correlation_map(x, prototype_block):
    B, C, H, W = x.size()

    n_p = prototype_block / prototype_block.norm(dim=2, keepdim=True)
    n_x = x.view(B, C, -1) / x.view(B, C, -1).norm(dim=1, keepdim=True)
    corr = torch.bmm(n_p, n_x).view(B, -1, H, W)
    return corr

def get_ocr_vector(x):
    b, c, h, w = x.size()
    probs = x.view(b, c, -1)
    ss_map = F.softmax(probs, dim=2)
    ss_map = ss_map.view(b, c, h, w)
    pb = get_prototype(x, ss_map.clone().detach())
    return pb

class Transformer(nn.Module):
    def __init__(self, in_channels):
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.theta = nn.Linear(self.in_channels, self.inter_channels)
        self.phi = nn.Linear(self.in_channels, self.inter_channels)
        self.g = nn.Linear(self.in_channels, self.inter_channels)
        self.W = nn.Linear(self.inter_channels, self.in_channels)

    def forward(self, ori_feature):
        ori_feature = ori_feature.permute(0, 2, 1)
        feature = self.bn_relu(ori_feature)
        feature = feature.permute(0, 2, 1)
        B, N, C = feature.size()

        x_theta = self.theta(feature)
        x_phi = self.phi(feature)
        x_phi = x_phi.permute(0, 2, 1)
        attention = torch.matmul(x_theta, x_phi)

        f_div_C = F.softmax(attention, dim=-1)
        g_x = self.g(feature)
        y = torch.matmul(f_div_C, g_x)
        W_y = self.W(y).contiguous().view(B, N, C)
        att_fea = ori_feature.permute(0, 2, 1) + W_y
        return att_fea

class Graph_Attention_Network(nn.Module):
    def __init__(self, in_channels):
        super(Graph_Attention_Network, self).__init__()
        self.transformer = Transformer(in_channels)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, prototype_block):
        att_prototype_block = self.transformer(prototype_block)
        prototype_for_graph = att_prototype_block.permute(0, 2, 1)
        graph_prototype = get_graph_feature(prototype_for_graph, k=10)
        graph_prototype = self.conv1(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]

        graph_prototype = get_graph_feature(graph_prototype, k=10)
        graph_prototype = self.conv2(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]

        graph_prototype = get_graph_feature(graph_prototype, k=10)
        graph_prototype = self.conv3(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]
        graph_prototype_block = graph_prototype.permute(0, 2, 1)
        return graph_prototype_block

class Cross_Attention_Network(nn.Module):
    def __init__(self, in_channels):
        super(Cross_Attention_Network, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.theta = nn.Linear(self.in_channels, self.inter_channels)
        self.phi = nn.Linear(self.in_channels, self.inter_channels)
        self.g = nn.Linear(self.in_channels, self.inter_channels)

        self.W = nn.Linear(self.inter_channels, self.in_channels)

    def forward(self, A, B):
        A = self.bn_relu(A)
        A = A.permute(0, 2, 1)

        B = self.bn_relu(B)
        B = B.permute(0, 2, 1)

        N, num, c = A.size()

        x_theta = self.theta(A)
        x_phi = self.phi(B)
        x_phi = x_phi.permute(0, 2, 1)
        attention = torch.matmul(x_theta, x_phi)

        f_div_C = F.softmax(attention, dim=-1)
        g_x = self.g(A)
        y = torch.matmul(f_div_C, g_x)

        W_y = self.W(y).contiguous().view(N, num, c)

        att_fea = A + W_y
        att_fea = att_fea.permute(0, 2, 1)

        return att_fea

class Mutual_Feature_Refinement(nn.Module):
    def __init__(self, in_channels):
        super(Mutual_Feature_Refinement, self).__init__()
        self.CA1 = Cross_Attention_Network(in_channels)
        self.CA2 = Cross_Attention_Network(in_channels)

    def forward(self, label_map, unlabeled_map):
        B, C, H, W = label_map.size()
        label_f = label_map.view(B, C, -1)
        unlebeled_f = unlabeled_map.view(B, C, -1)
        att_l = self.CA1(label_f, unlebeled_f)
        att_u = self.CA2(unlebeled_f, label_f)
        fu_l = att_l.view(B, C, H, W)
        fu_u = att_u.view(B, C, H, W)
        return fu_l, fu_u

class Prototype_Correlation_Generation(nn.Module):
    def __init__(self, in_channels):
        super(Prototype_Correlation_Generation, self).__init__()
        self.GAN = Graph_Attention_Network(in_channels)
        self.out = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1),
                                 nn.BatchNorm2d(in_channels),
                                 nn.ReLU(inplace=True),
                                 )

    def forward(self, x):
        pb = get_ocr_vector(x)
        graph_pb = self.GAN(pb)
        map = get_correlation_map(x, graph_pb)
        return map

class Foreground_Prototype_Module(nn.Module):
    def __init__(self, in_channels):
        super(Foreground_Prototype_Module, self).__init__()

        self.MFR = Mutual_Feature_Refinement(in_channels)

    def forward(self, feat_l, feat_u):
        pb1 = get_ocr_vector(feat_l)
        pb2 = get_ocr_vector(feat_u)
        label_map = get_correlation_map(feat_l, pb1)
        unlabeled_map = get_correlation_map(feat_u, pb2)

        fu_l, fu_u = self.MFR(label_map, unlabeled_map)

        return fu_l, fu_u