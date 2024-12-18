import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv, global_add_pool, GATv2Con
# class ResDilaCNNBlock(nn.Module):
#     def __init__(self, dilaSize, filterSize, dropout=0.15, name='ResDilaCNNBlock'):
#         super(ResDilaCNNBlock, self).__init__()
#         self.layers = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
#             nn.ReLU(),
#             nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
#         )
#         self.name = name
#
#     def forward(self, x):
#         # x: batchSize × filterSize × seqLen
#         return x + self.layers(x)
#
#
# class ResDilaCNNBlocks(nn.Module):
#     # def __init__(self, feaSize, filterSize, blockNum=5, dropout=0.35, name='ResDilaCNNBlocks'):
#     def __init__(self, feaSize, filterSize, blockNum=5, dilaSizeList=[1, 2, 4, 8, 16], dropout=0.5,
#                  name='ResDilaCNNBlocks'):
#         super(ResDilaCNNBlocks, self).__init__()  #
#         self.blockLayers = nn.Sequential()
#         self.linear = nn.Linear(feaSize, filterSize)
#         for i in range(blockNum):
#             self.blockLayers.add_module(f"ResDilaCNNBlock{i}",
#                                         ResDilaCNNBlock(dilaSizeList[i % len(dilaSizeList)], filterSize,
#                                                         dropout=dropout))
#             # self.blockLayers.add_module(f"ResDilaCNNBlock{i}", ResDilaCNNBlock(filterSize,dropout=dropout))
#         self.name = name
#         self.act = nn.ReLU()
#
#     def forward(self, x):
#         # x: batchSize × seqLen × feaSize
#         x = self.linear(x)  # => batchSize × seqLen × filterSize
#         x = self.blockLayers(x.transpose(1, 2))  # => batchSize × seqLen × filterSize
#         x = self.act(x)  # => batchSize × seqLen × filterSize
#
#         # x = self.pool(x.transpose(1, 2))
#         x = torch.max(x.transpose(1, 2), dim=2)[0]
#         return x

class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
    
    def forward(self, x):

        return self.inc(x)

class LinearReLU(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        
        return self.inc(x)

class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):

        return self.inc(x).squeeze(-1)

class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx+1, embedding_num, 96, 3)
            )

        self.linear = nn.Linear(block_num * 96, 96)
        
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x

class GATv2(torch.nn.Module):
    def __init__(self,n_output=1, num_feature_xd=78, num_features_xt=25,
                 n_filters=32, filter_num=32, embed_dim=128, output_dim=96, dropout=0.2):

        super(GATv2, self).__init__()
        self.protein_encoder = TargetRepresentation(block_num=3, vocab_size=26, embedding_num=128)
        self.n_output = n_output
        self.Conv1 = GCNConv(22, num_feature_xd)
        self.Conv2 = GCNConv(num_feature_xd, num_feature_xd * 2)
        self.Conv3 = GCNConv(num_feature_xd * 2, num_feature_xd * 4)
        self.relu = nn.ReLU()
        self.fc_g1 = nn.Linear(312, 1024)
        self.fc_g2 = nn.Linear(1024, 96)
        self.dropout = nn.Dropout(dropout)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.Conv1(x, edge_index)
        x = self.relu(x)
        x = self.Conv2(x, edge_index)
        x = self.relu(x)
        x = self.Conv3(x, edge_index)
        x = self.relu(x)

        x = gmp(x, batch)  # global_max_pooling
        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)
        return x

class GETDTA(nn.Module):
    def __init__(self, block_num, vocab_size,num_layer=1, embedding_num=128, filter_num=32, out_dim=1,):
        super().__init__()
        self.num_layer = num_layer
        self.embedding_num=embedding_num
        self.protein_encoder = TargetRepresentation(block_num, vocab_size, embedding_num)
        self.ligand_encoder = GATv2()
        self.embed_smile = nn.Embedding(65, embedding_num)
        self.classifier = nn.Sequential(
            nn.Linear(filter_num * 3 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )
        self.fc2 = nn.Linear(2048, 1024)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(256, 96)

        # self.fc = nn.Linear(85, 96)             #davis
        self.fc = nn.Linear(2048, 96)              #kiba
        self.fc3 = nn.Linear(192, 96)
        self.att = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32), requires_grad=True)

    def forward(self, data):
        target = data.target
        fp = data.ecfp
        fp = torch.tensor(fp, dtype=torch.float32)
        protein_x = self.protein_encoder(target)
        ligand_x0 = self.ligand_encoder(data)
        # ligand_x1 = self.embed_smile(fp)
        # ligand_x2 = self.onehot_smi_net(ligand_x1)
        # ligand_x3 = self.fc(ligand_x2)
        ligand_x3 = self.fc(fp)
        ligand_x4 = ligand_x0 * self.att[0]
        ligand_x5 = ligand_x3 * self.att[1]
        ligand_x = torch.cat([ligand_x4, ligand_x5], dim=-1)
        ligand_x = self.fc3(ligand_x)
        # x1 = torch.cat([ligand_x, ligand_x2], dim=-1)
        # ligand_x = self.fc(x1)
        x = torch.cat([protein_x, ligand_x], dim=-1)
        x = self.classifier(x)

        return x


