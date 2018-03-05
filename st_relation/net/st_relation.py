import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from .net import Unit2D, conv_init, import_class
from .unit_gcn import unit_gcn
from .relation_g_f import relation_g_f
from torch.nn.init import normal, constant

# default_backbone = [(64, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128,
#                                                             2), (128, 128, 1),
#                     (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 256, 1)]

default_backbone = [(256, 256, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2),(256, 256, 1)]


class Model(nn.Module):
    """ Spatial temporal graph convolutional networks
                        for skeleton-based action recognition.

    Input shape:
        Input shape should be (N, C, T, V, M)
        where N is the number of samples,
              C is the number of input channels,
              T is the length of the sequence,
              V is the number of joints or graph nodes,
          and M is the number of people.
    
    Arguments:
        About shape:
            channel (int): Number of channels in the input data
            num_class (int): Number of classes for classification
            window_size (int): Length of input sequence
            num_point (int): Number of joints or graph nodes
            num_person (int): Number of people
        About net:
            use_data_bn: If true, the data will first input to a batch normalization layer
            backbone_config: The structure of backbone networks
        About graph convolution:
            graph: The graph of skeleton, represtented by a adjacency matrix
            graph_args: The arguments of graph
            use_local_bn: If true, each node in the graph have specific parameters of batch normalzation layer
        About temporal convolution:
            temporal_kernel_size: The kernel size of temporal convolution
            dropout: The drop out rate of the dropout layer in front of each temporal convolution layer

    """

    def __init__(self,
                 channel,
                 T_dim,
                 num_class,
                 window_size,
                 num_point,
                 num_person=1,
                 use_data_bn=False,
                 backbone_config=None,
                 graph=None,
                 graph_args=dict(),
                 use_local_bn=False,
                 temporal_kernel_size=9,
                 dropout=0.8):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
            self.A = torch.from_numpy(self.graph.A).float().cuda(0)
            self.Identity = self.graph.Identity

        self.num_class = num_class
        self.use_data_bn = use_data_bn

        # Different bodies share batchNorma parameters or not
        self.M_dim_bn = True
        self.T_dim = T_dim

        if self.M_dim_bn:
            self.data_bn = nn.BatchNorm1d(channel * num_point * num_person)
        else:
            self.data_bn = nn.BatchNorm1d(channel * num_point)

        # kwargs = dict(
        #     A=self.A,
        #     use_local_bn=use_local_bn,
        #     dropout=dropout,
        #     kernel_size=temporal_kernel_size)

        # backbone
        if backbone_config is None:
            backbone_config = default_backbone

        backbone_in_c = backbone_config[0][0]
        backbone_out_c = backbone_config[-1][1]
        backbone_out_t = window_size

        # backbone = []
        # for in_c, out_c, stride in backbone_config:
        #     backbone.append(unit(in_c, out_c, stride=stride, **kwargs))
        #     if backbone_out_t % stride == 0:
        #         backbone_out_t = backbone_out_t // stride
        #     else:
        #         backbone_out_t = backbone_out_t // stride + 1
        # self.backbone = nn.ModuleList(backbone)

        # head
        self.relation = relation_g_f(
            channel * T_dim,
            backbone_in_c,
            self.A,
            use_local_bn=use_local_bn,
            num_class=num_class)
        # self.tcn0 = Unit2D(60, num_class, kernel_size=9)
        tpp_level = 3
        self.TPP = TPPLayer(tpp_level)

        self.drop = nn.Dropout(dropout)
        # tail
        # self.person_bn = nn.BatchNorm1d(backbone_out_c)
        self.gap_size = backbone_out_t
        num_t_pooling = 0
        for ii in range(tpp_level):
            num_t_pooling += 2**ii

        std = 0.001
        self.fc =  nn.Linear(256 * num_t_pooling, num_class)
        normal(self.fc.weight, 0, std)
        constant(self.fc.bias, 0)


    def forward(self, x):
        # N, C, T, V, M = x.size()
        # a = torch.from_numpy(np.tile(self.Identity, (N, 300, 2, 1, 1))).float().cuda(0)
        # a = Variable(a, requires_grad=False)
        # a = a.permute(0, 3, 1, 4, 2)
        # x = torch.cat([x, a], 1)
        N, C, T, V, M = x.size()
        # data bn
        if self.use_data_bn:
            if self.M_dim_bn:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            else:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
            ### 8*150*300
            x = self.data_bn(x)### 8*150*300
            # to (N*M, C, T, V)
            x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V)
        else:
            # from (N, C, T, V, M) to (N*M, C, T, V)
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        ### 16*3*300*25
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N * M, V, C * self.T_dim, -1)
        x = x.permute(0, 2, 3, 1).contiguous()
        # model ###32*15*60*25
        x = self.relation(x) ### 32*60*60*1    num_class = 20   32*20*60*1

        # M pooling
        x = x.view(N, M, x.size(1), x.size(2))###16*2*60*30
        x = x.mean(dim=1)###16*60*60

        x = x.view(N, x.size(1), x.size(2), 1)
        # TPP pooling
        x = self.TPP(x) ###32*60*30*1
        # x = self.drop(x)
        x = self.fc(x) ###16*60

        # x = F.avg_pool1d(x, kernel_size=x.size()[2])
        return x

class TPPLayer(nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        super(TPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            # kernel_size = h // (2 ** i)
            level = 2 ** i
            xx = (math.ceil(1.0 * h / level), 1)
            kernel_size = (int(math.ceil(1.0 * h / level)), 1)
            stride = (int(math.ceil(1.0 * h / level)), 1)
            padding = (int((kernel_size[0] * level - h + 1) / 2), 0)

            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                      stride=stride, padding=padding).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=stride, padding=padding).view(bs, -1)

            # if self.pool_type == 'max_pool':
            #     tensor = F.max_pool2d(x, kernel_size=kernel_size,
            #                           stride=stride, padding=padding).view(bs, -1)
            # else:
            #     tensor = F.avg_pool2d(x, kernel_size=kernel_size,
            #                           stride=stride, padding=padding).view(bs, -1)

            if (i == 0):
                tpp = tensor.view(bs, -1)
            else:
                tpp = torch.cat((tpp, tensor.view(bs, -1)), 1)

        # pooling_layers.append(tensor)
        # x = torch.cat(pooling_layers, pooling_layers.dim()-1)
        return tpp