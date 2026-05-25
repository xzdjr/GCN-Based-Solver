import torch
from torch_geometric.data import HeteroData
import numpy as np

def build_hetero_graph(alpha, r, v, A, b, x_opt=None):
    data = HeteroData()#PyTorch Geometric (PyG) 库中专门用于处理“包含多种节点和边类型”的异构图的数据结构
    K, N = v.shape
    M = len(b)
    
    # 1. Node features
    data['customer'].x = torch.tensor([[0.0, alpha[k], 0.0] for k in range(K)], dtype=torch.float)
    data['product'].x = torch.tensor([[r[j], 0.0, 0.0] for j in range(N)], dtype=torch.float)
    data['constraint'].x = torch.tensor([[0.0, 0.0, b[i]] for i in range(M)], dtype=torch.float)
    
    # 2. Edges: Customer <-> Product 维度为2
    edge_index_cp = []#客户-产品边
    edge_attr_cp = []#产品-约束边
    for k in range(K):
        for j in range(N):
            edge_index_cp.append([k, j]) #P12
            edge_attr_cp.append([v[k, j], 0.0])
    
    # Convert to tensors
    edge_index_cp = torch.tensor(edge_index_cp, dtype=torch.long).t().contiguous()
    edge_attr_cp = torch.tensor(edge_attr_cp, dtype=torch.float)
    

    #消息传递的实现
    # Forward edges (Customer -> Product)
    data['customer', 'values', 'product'].edge_index = edge_index_cp
    data['customer', 'values', 'product'].edge_attr = edge_attr_cp
    
    # Reverse edges (Product -> Customer)
    data['product', 'rev_values', 'customer'].edge_index = edge_index_cp.flip([0])
    data['product', 'rev_values', 'customer'].edge_attr = edge_attr_cp
    
    # 3. Edges: Product <-> Constraint
    edge_index_pc = []
    edge_attr_pc = []
    for j in range(N):
        for i in range(M):
            if A[i, j] != 0:
                edge_index_pc.append([j, i])
                edge_attr_pc.append([0.0, A[i, j]])
                
    # Handle the case where there might be no product-constraint edges
    if len(edge_index_pc) > 0:
        edge_index_pc = torch.tensor(edge_index_pc, dtype=torch.long).t().contiguous()
        edge_attr_pc = torch.tensor(edge_attr_pc, dtype=torch.float)
    else:
        edge_index_pc = torch.empty((2, 0), dtype=torch.long)
        edge_attr_pc = torch.empty((0, 2), dtype=torch.float)
                
    # Forward edges (Product -> Constraint)
    data['product', 'restricted_by', 'constraint'].edge_index = edge_index_pc
    data['product', 'restricted_by', 'constraint'].edge_attr = edge_attr_pc
    
    # Reverse edges (Constraint -> Product)
    data['constraint', 'rev_restricted_by', 'product'].edge_index = edge_index_pc.flip([0])
    data['constraint', 'rev_restricted_by', 'product'].edge_attr = edge_attr_pc
    
    # 4. Labels
    if x_opt is not None:
        data['product'].y = torch.tensor(x_opt, dtype=torch.float)
        
    return data