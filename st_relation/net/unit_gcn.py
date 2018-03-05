# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .net import conv_init


class unit_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 use_local_bn=False,
                 kernel_size=1,
                 stride=1,
                 mask_learning=False):
        super(unit_gcn, self).__init__()

        # ==========================================
        # number of nodes
        self.V = A.size()[-1]

        # the adjacency matrixes of the graph
        self.A = Variable(
            A.clone(), requires_grad=False).view(-1, self.V, self.V)

        # number of input channels
        self.in_channels = in_channels

        # number of output channels
        self.out_channels = out_channels

        # if true, use mask matrix to reweight the adjacency matrix
        self.mask_learning = mask_learning

        # number of adjacency matrix (number of partitions)
        self.num_A = self.A.size()[0]

        # if true, each node have specific parameters of batch normalizaion layer.
        # if false, all nodes share parameters.
        self.use_local_bn = use_local_bn
        # ==========================================

        self.conv_list = nn.ModuleList([
            nn.Conv2d(
                self.in_channels,
                self.in_channels,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1)) for i in range(2)
        ])

        self.MLP = nn.ModuleList([
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1))
        ])

        if mask_learning:
            self.mask = nn.Parameter(torch.ones(self.A.size()))
        if use_local_bn:
            self.bn_1 = nn.BatchNorm1d(self.in_channels * self.V)
            self.bn_2 = nn.BatchNorm1d(self.out_channels * self.V)
        else:
            self.bn_1 = nn.BatchNorm2d(self.in_channels)
            self.bn_2 = nn.BatchNorm2d(self.out_channels)

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

        # initialize
        for conv in self.conv_list:
            conv_init(conv)
        for conv in self.MLP:
            conv_init(conv)


    def forward(self, x):
        N, C, T, V = x.size()
        self.A = self.A.cuda(x.get_device())
        A = self.A

        # reweight adjacency matrix
        if self.mask_learning:
            A = A * self.mask

        # graph convolution
        a = A[0]
        xa = x.view(-1, V).mm(a).view(N, C, T, V)
        y_center = self.conv_list[0](xa)

        # graph convolution
        for i, a in enumerate(A[1:]):
            xa = x.view(-1, V).mm(a).view(N, C, T, V)
            y_pair = self.conv_list[1](xa) + y_center

            if self.use_local_bn:
                y_pair = y_pair.permute(0, 1, 3, 2).contiguous().view(
                N, self.out_channels * V, T)
                y_pair = self.bn_1(y_pair)
                y_pair = y_pair.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
            else:
                y_pair = self.bn_1(y_pair)

            # nonlinear
            y_pair = self.relu_1(y_pair)
            if i == 0:
                y = self.MLP[0](y_pair)
            else:
                y += self.MLP[0](y_pair)

        # batch normalization
        if self.use_local_bn:
            y = y.permute(0, 1, 3, 2).contiguous().view(
                N, self.out_channels * V, T)
            y = self.bn_2(y)
            y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
        else:
            y = self.bn_2(y)###32*64*300*25
        y = y.sum(dim=3).view(N, self.out_channels, T, 1)  ###32*64*300
        # nonlinear
        y = self.relu_2(y)
        return y