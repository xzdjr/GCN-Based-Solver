# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv, to_hetero

class Base_GCN(nn.Module):
    def __init__(self, hidden_dim, out_dim=1):
        super(Base_GCN, self).__init__()
        # GENConv 支持边特征与 softmax 聚合 P14 消息传递层
        self.conv1 = GENConv(in_channels=-1, out_channels=hidden_dim, aggr='softmax', edge_dim=2)
        self.conv2 = GENConv(in_channels=hidden_dim, out_channels=hidden_dim, aggr='softmax', edge_dim=2)
        
        self.dropout = nn.Dropout(0.5)
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        #传递消息并激活
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        #dropout层
        x = self.dropout(x)
        #第二层传递消息
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        #概率转换
        out = torch.sigmoid(self.out_layer(x))
        return out

def create_hetero_model(metadata):
    # 将同构 GNN 转换为能处理我们定义的 Customer, Product, Constraint 异构图的模型
    base_model = Base_GCN(hidden_dim=32)
    model = to_hetero(base_model, metadata, aggr='sum')
    return model