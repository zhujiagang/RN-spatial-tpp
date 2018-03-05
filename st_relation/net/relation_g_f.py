# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .net import conv_init


class relation_g_f(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 use_local_bn=False,
                 kernel_size=1,
                 stride=1,
                 num_class=60):
        super(relation_g_f, self).__init__()

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

        # number of adjacency matrix (number of partitions)
        self.num_A = self.A.size()[0]

        # if true, each node have specific parameters of batch normalizaion layer.
        # if false, all nodes share parameters.
        self.use_local_bn = use_local_bn
        # ==========================================
        # ==========================================
        # relation_g    4 layers
        self.relation_g_1 = nn.ModuleList([
            nn.Conv2d(
                self.in_channels,
                256,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1)) for i in range(2)
        ])

        self.relation_g_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(kernel_size, 1), padding=(int((kernel_size - 1) / 2), 0),
                      stride=(stride, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(kernel_size, 1), padding=(int((kernel_size - 1) / 2), 0),
                      stride=(stride, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(kernel_size, 1), padding=(int((kernel_size - 1) / 2), 0),
                      stride=(stride, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # relation_f 3 layers
        self.relation_f = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(kernel_size, 1), padding=(int((kernel_size - 1) / 2), 0),
                      stride=(stride, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(kernel_size, 1), padding=(int((kernel_size - 1) / 2), 0),
                      stride=(stride, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=(kernel_size, 1), padding=(int((kernel_size - 1) / 2), 0),
                  stride=(stride, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        if use_local_bn:
            self.bn_1 = nn.BatchNorm1d(self.out_channels * self.V)
            self.bn_2 = nn.BatchNorm1d(self.out_channels * self.V)
        else:
            self.bn_1 = nn.BatchNorm2d(self.out_channels)
            self.bn_2 = nn.BatchNorm2d(self.out_channels)

        self.relu_1 = nn.ReLU()

        # initialize
        for conv in self.relation_g_1:
            conv_init(conv)
        for conv in self.relation_g_2:
            if isinstance(conv, nn.Conv2d):
                conv_init(conv)

        for conv in self.relation_f:
            if isinstance(conv, nn.Conv2d):
                conv_init(conv)


    def forward(self, x):
        N, C, T, V = x.size()
        self.A = self.A.cuda(x.get_device())
        A = self.A
        a = A[0]
        xa = x.view(-1, V).mm(a).view(N, C, T, V)
        y_center = self.relation_g_1[0](xa)

        # Do Pairwise Object G_Theta Propagation
        for i, a in enumerate(A[1:]):
            xa = x.view(-1, V).mm(a).view(N, C, T, V)
            y_pair = self.relation_g_1[1](xa) + y_center
            y_pair = self.bn_1(y_pair)
            # nonlinear
            y_pair = self.relu_1(y_pair)
            if i == 0:
                y = self.relation_g_2[0](y_pair)
            else:
                y += self.relation_g_2[0](y_pair)

        y = y.sum(dim=3).view(N, self.out_channels, T, 1)  ###32*256*60*1
        y = self.relation_f(y)   ###32*60*60*1
        return y