import numpy as np
import gurobipy as gp
from gurobipy import GRB

def expected_revenue(x, alpha, r, v):
    #P7
    revenue = 0
    N = len(r)
    for k in range(len(alpha)):
        denom = 1 + np.sum(v[k] * x)
        num = np.sum(alpha[k] * x * v[k] * r)#x*v
        revenue += num / denom
    return revenue

def gi_policy(p, r, alpha, v, A, b):
    #Algorithm 2 P18 基于概率的排名
    N = len(p)
    sorted_indices = np.argsort(p)[::-1]
    best_x = np.zeros(N)
    best_rev = 0
    
    #依次测试上榜第一名，前两名，前三名
    for i in range(1, N + 1):
        x_candidate = np.zeros(N)
        x_candidate[sorted_indices[:i]] = 1
        
        if np.all(A @ x_candidate <= b):
            rev = expected_revenue(x_candidate, alpha, r, v)
            if rev > best_rev:
                best_rev = rev
                best_x = x_candidate.copy()
                
    return best_x, best_rev

def ls_policy(x_init, alpha, r, v, A, b):
    #Algorithm 3 GILS P19
    x = x_init.copy()
    N = len(r)
    removal_count = np.zeros(N)
    improved = True
    
    while improved:
        improved = False
        best_op_rev = expected_revenue(x, alpha, r, v)
        best_x = x.copy()
        
        for i in range(N):
            if x[i] == 0:
                x_temp = x.copy()
                x_temp[i] = 1
                if np.all(A @ x_temp <= b):
                    rev = expected_revenue(x_temp, alpha, r, v)
                    if rev > best_op_rev * 1.001:
                        best_op_rev = rev
                        best_x = x_temp
                        improved = True
        
        if improved:
            x = best_x
            continue
            
        for i in range(N):
            if x[i] == 1 and removal_count[i] < 1:
                x_temp = x.copy()
                x_temp[i] = 0
                rev = expected_revenue(x_temp, alpha, r, v)
                if rev > best_op_rev * 1.001:
                    best_op_rev = rev
                    best_x = x_temp
                    removal_count[i] += 1
                    improved = True

        for i in range(N):
            if x[i] == 1:
                for j in range(N):
                    if x[j] == 0:
                        x_temp = x.copy()
                        x_temp[i] = 0
                        x_temp[j] = 1
                        if np.all(A @ x_temp <= b):
                            rev = expected_revenue(x_temp, alpha, r, v)
                            if rev > best_op_rev * 1.001:
                                best_op_rev = rev
                                best_x = x_temp
                                improved = True

        if improved:
            x = best_x

    return x, expected_revenue(x, alpha, r, v)

def gils_policy(p, r, alpha, v, A, b):
    x_gi, _ = gi_policy(p, r, alpha, v, A, b)
    return ls_policy(x_gi, alpha, r, v, A, b)#在GI的基础上局部搜索

def gip_policy(p, A, b):
    #Algorithm 4 P19
    #直接用GCN算出的Pj代替函数，变成线性问题
    N = len(p)
    M = A.shape[0]
    
    model = gp.Model("GIP")
    model.setParam('OutputFlag', 0)
    
    x = model.addVars(N, vtype=GRB.BINARY, name="x")
    
    model.setObjective(gp.quicksum(p[j] * x[j] for j in range(N)), GRB.MAXIMIZE)
    
    for i in range(M):
        model.addConstr(gp.quicksum(A[i, j] * x[j] for j in range(N)) <= b[i])
        
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return np.array([x[j].X for j in range(N)])
    return np.zeros(N)