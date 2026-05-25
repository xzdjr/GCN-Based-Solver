# data_generator.py
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog
from config import *

def generate_instance(N, K, M):
    # Customer proportions P7
    z = np.random.uniform(0, 1, K)#[0,1]中随机生成k个
    alpha = z / np.sum(z)#归一化
    
    # Revenues and Utilities P7
    r = np.random.uniform(1, 2, N)#产品价格
    u = np.random.uniform(0, 1, (K, N))#K*N矩阵
    v = np.exp(u - PRICE_SENSITIVITY_ETA * r)#吸引权重
    
    # Constraints (A * x <= b)
    A = np.zeros((M, N))
    b = np.zeros(M)
    
    num_capacity = M // 2
    num_precedence = M - num_capacity
    #P21
    # Capacity constraints 一半容量约束
    for i in range(num_capacity):
        A[i, :] = np.random.uniform(0, 1, N)
        b[i] = np.random.uniform(5, 10)
        
    # Precedence constraints 一半优先约束
    for i in range(num_capacity, M):
        j1, j2 = np.random.choice(N, 2, replace=False)
        A[i, j1] = 1
        A[i, j2] = -1
        b[i] = 0
        
    return alpha, r, v, A, b

#Algorithm 9
def compute_mccormick_estimators(V, A, b, K, M, N):
    F = np.zeros((K, N, 2))
    for k in range(K):
        for j in range(N):
            for xi in [0, 1]:
                #  目标函数: max \sum v_{kj} x_j
                c = -V[k, :]
                bounds = []
                for index in range(N):
                    if index == j:
                        bounds.append((xi, xi)) 
                    else:
                        bounds.append((0.0, 1.0))
                
                res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
                if res.success:
                    F[k, j, xi] = -res.fun
                else:
                    F[k, j, xi] = 0.0
    return F

#Algorithm 8
def solve_conic_algorithm_8(alpha, r, v, A, b, F):
    K, N = v.shape
    M = len(b)
    r_bar = np.max(r)
    
    #  Gurobi 模型
    model = gp.Model("Conic_IP_Algorithm_8") 
    model.setParam('OutputFlag', 0) # 关闭控制台输出
    
    # 变量 
    x = model.addVars(N, vtype=GRB.BINARY, name="x")#x_j \in {0, 1}
    y = model.addVars(K, vtype=GRB.CONTINUOUS, lb=0.0, name="y")# y_k >= 0
    z = model.addVars(K, N, vtype=GRB.CONTINUOUS, lb=0.0, name="z")# z_kj >= 0
    w = model.addVars(K, vtype=GRB.CONTINUOUS, lb=0.0, name="w")
    
    # 目标函数 max r_bar - sum(alpha_k * r_bar * y_k) - sum(alpha_k * v_kj * (r_bar - r_j) * z_kj)
    obj = r_bar \
          - gp.quicksum(alpha[k] * r_bar * y[k] for k in range(K)) \
          - gp.quicksum(alpha[k] * v[k, j] * (r_bar - r[j]) * z[k, j] for k in range(K) for j in range(N))
    model.setObjective(obj, GRB.MAXIMIZE)

    # s.t.
    for i in range(M):
        model.addConstr(gp.quicksum(A[i, j] * x[j] for j in range(N)) <= b[i], name=f"capacity_{i}")# Ax <= b
    for k in range(K):
        model.addConstr(w[k] == 1 + gp.quicksum(v[k, j] * x[j] for j in range(N)), name=f"w_def_{k}")# w_k = 1 + sum(v_kj * x_j)
        model.addQConstr(y[k] * w[k] >= 1.0, name=f"socp_y_{k}") # y_k * w_k >= 1 
        model.addConstr(y[k] + gp.quicksum(v[k, j] * z[k, j] for j in range(N)) >= 1.0, name=f"y_z_{k}")# y_k + sum(v_kj * z_kj) >= 1
        for j in range(N):
            model.addQConstr(z[k, j] * w[k] >= x[j] * x[j], name=f"socp_z_{k}_{j}")# z_kj * w_k >= x_j^2
            model.addConstr(z[k, j] <= x[j] / (1 + v[k, j]), name=f"mc1_{k}_{j}")# z_kj <= x_j / (1 + v_kj)
            model.addConstr(z[k, j] >= x[j] / (1 + F[k, j, 1]), name=f"mc2_{k}_{j}")# z_kj >= x_j / (1 + f_{k|x_j=1})
            model.addConstr(z[k, j] <= y[k] - (1 - x[j]) / (1 + F[k, j, 0]), name=f"mc3_{k}_{j}")#z_kj <= y_k - (1 - x_j) / (1 + f_{k|x_j=0})
            model.addConstr(z[k, j] >= y[k] - (1 - x[j]), name=f"mc4_{k}_{j}")# z_kj >= y_k - (1 - x_j)
            
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        x_opt = np.array([x[j].X for j in range(N)])
        return np.round(x_opt).astype(int)
    else:
        print("Warning: Gurobi could not find an optimal solution.")
        return np.zeros(N, dtype=int)

def solve_optimal_assortment(alpha, r, v, A, b):
    K, N = v.shape
    M = len(b)
    F = compute_mccormick_estimators(v, A, b, K, M, N)
    x_opt = solve_conic_algorithm_8(alpha, r, v, A, b, F)
    
    return x_opt