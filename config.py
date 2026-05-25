import torch

NUM_CUSTOMERS_TRAIN = 10 #客户种类数
NUM_PRODUCTS_TRAIN = 5 #训练时产品数量
NUM_CONSTRAINTS_TRAIN = 10 #约束数量

NUM_CUSTOMERS_TEST = 10
NUM_PRODUCTS_TEST = 10
NUM_CONSTRAINTS_TEST = 10

NUM_TRAIN_SAMPLES = 100 #训练数据集大小
PRICE_SENSITIVITY_ETA = 3.0 #价格敏感度系数
#生成吸引力权重公式 v  中使用

HIDDEN_DIM = 32 #隐藏层维度
DROPOUT_RATE = 0.5 #随机失活率
LEARNING_RATE = 1e-4 #learning rate
BATCH_SIZE = 128 
EPOCHS = 100 #训练轮次
PATIENCE = 50 #早停机制，连续50个没有提升则停止

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')