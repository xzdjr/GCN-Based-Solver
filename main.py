import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
from config import *
from data_generator import generate_instance, solve_optimal_assortment
from graph_builder import build_hetero_graph
from model import create_hetero_model
from policies import expected_revenue, gi_policy, gils_policy, gip_policy

def print_final_performance(model, test_data_tuples):
    model.eval()
    gi_results = []
    gils_results = []
    gip_results = []
    
    print("\n" + "="*50)
    print(f"{'Policy':<20} | {'Avg Ratio':<20}")
    print("-" * 50)
    
    with torch.no_grad():
        for graph, alpha, r, v, A, b, x_opt in test_data_tuples:
            batch = graph.to(DEVICE)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            probs = out['product'].squeeze().cpu().numpy()
            
            rev_opt = expected_revenue(x_opt, alpha, r, v)
            
            _, rev_gi = gi_policy(probs, r, alpha, v, A, b)
            _, rev_gils = gils_policy(probs, r, alpha, v, A, b)
            
            x_gip = gip_policy(probs, A, b)
            rev_gip = expected_revenue(x_gip, alpha, r, v)
            
            if rev_opt > 0:
                gi_results.append(rev_gi / rev_opt)
                gils_results.append(rev_gils / rev_opt)
                gip_results.append(rev_gip / rev_opt)

    print(f"{'GI':<20} | {np.mean(gi_results):>18.2%}")
    print(f"{'GILS':<20} | {np.mean(gils_results):>18.2%}")
    print(f"{'GIP':<20} | {np.mean(gip_results):>18.2%}")
    print("="*50)

def main():
    #生成训练数据
    all_data = []
    for _ in range(NUM_TRAIN_SAMPLES):
        alpha, r, v, A, b = generate_instance(NUM_PRODUCTS_TRAIN, NUM_CUSTOMERS_TRAIN, NUM_CONSTRAINTS_TRAIN)
        x_opt = solve_optimal_assortment(alpha, r, v, A, b)
        graph = build_hetero_graph(alpha, r, v, A, b, x_opt)
        all_data.append(graph)
    #80%训练集，20%测试
    split = int(0.8 * len(all_data))
    train_data = all_data[:split]
    val_data = all_data[split:]
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    #模型初始化
    metadata = all_data[0].metadata()
    model = create_hetero_model(metadata).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #训练 100次
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            pred = out['product'].squeeze()
            target = batch['product'].y
            loss = F.binary_cross_entropy(pred, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                val_loss = F.binary_cross_entropy(out['product'].squeeze(), batch['product'].y)
                total_val_loss += val_loss.item()
        
        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
    #测试100个
    test_data_tuples = []
    for _ in range(100):
        alpha, r, v, A, b = generate_instance(NUM_PRODUCTS_TEST, NUM_CUSTOMERS_TEST, NUM_CONSTRAINTS_TEST)
        x_opt = solve_optimal_assortment(alpha, r, v, A, b)
        graph = build_hetero_graph(alpha, r, v, A, b, x_opt)
        test_data_tuples.append((graph, alpha, r, v, A, b, x_opt))
    
    print_final_performance(model, test_data_tuples)

if __name__ == "__main__":
    main()