# GCN-Based-Solver

这是一个基于图神经网络（GCN/GNN）的组合优化实验项目，用于学习受约束 assortment optimization（产品组合选择）问题中的最优产品选择策略。项目会随机生成训练样本，通过 Gurobi 求解器得到最优标签，将实例构造成异构图，然后训练 PyTorch Geometric 模型预测每个产品被选择的概率，并使用多种启发式策略评估预测效果。

## 项目功能

- 随机生成客户、产品和约束数据。
- 使用 Gurobi 求解受约束的产品组合优化问题，生成监督学习标签。
- 将问题实例转换为包含客户、产品、约束三类节点的异构图。
- 使用 PyTorch Geometric 的 `GENConv` 和 `to_hetero` 构建异构 GCN 模型。
- 训练模型预测产品选择概率。
- 使用 GI、GILS、GIP 三种策略将模型概率转化为可行解，并与最优解收益进行对比。

## 文件结构

```text
.
|-- config.py          # 实验参数、训练参数和设备配置
|-- data_generator.py  # 随机实例生成、McCormick 估计、Gurobi 最优解求解
|-- graph_builder.py   # 将优化实例构造成 PyTorch Geometric 异构图
|-- main.py            # 数据生成、模型训练、测试评估主入口
|-- model.py           # GCN 基础模型和异构图模型构建
|-- policies.py        # 收益计算与 GI、GILS、GIP 策略
`-- README.md
```

## 环境依赖

建议使用 Python 3.10 或以上版本。项目主要依赖：

- `numpy`
- `scipy`
- `torch`
- `torch-geometric`
- `gurobipy`

其中 `gurobipy` 需要本机已安装 Gurobi，并配置可用许可证。否则最优解生成和 GIP 策略无法正常运行。

可参考以下命令安装 Python 依赖：

```bash
pip install numpy scipy torch torch-geometric gurobipy
```

如果 `torch-geometric` 安装失败，建议根据本机的 PyTorch、CUDA 和 Python 版本，参考 PyTorch Geometric 官方安装方式安装对应 wheel。

## 运行方法

在项目根目录下执行：

```bash
python main.py
```

程序流程如下：

1. 根据 `config.py` 中的参数随机生成训练样本。
2. 对每个样本调用 Gurobi 求解最优产品选择向量 `x_opt`。
3. 将样本转换为异构图数据。
4. 训练异构 GCN 模型。
5. 随机生成测试样本，并输出 GI、GILS、GIP 策略相对于最优解的平均收益比例。

训练过程中每 5 个 epoch 会打印一次训练损失和验证损失，最后会输出类似结果：

```text
==================================================
Policy               | Avg Ratio
--------------------------------------------------
GI                   |             xx.xx%
GILS                 |             xx.xx%
GIP                  |             xx.xx%
==================================================
```

## 主要配置

配置集中在 `config.py` 中：

```python
NUM_CUSTOMERS_TRAIN = 10
NUM_PRODUCTS_TRAIN = 5
NUM_CONSTRAINTS_TRAIN = 10

NUM_CUSTOMERS_TEST = 10
NUM_PRODUCTS_TEST = 10
NUM_CONSTRAINTS_TEST = 10

NUM_TRAIN_SAMPLES = 100
PRICE_SENSITIVITY_ETA = 3.0

HIDDEN_DIM = 32
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 50
```

常用调整项：

- `NUM_TRAIN_SAMPLES`：训练样本数量，越大训练越充分，但生成标签耗时也越长。
- `NUM_PRODUCTS_TRAIN` / `NUM_PRODUCTS_TEST`：训练和测试阶段的产品数量。
- `NUM_CUSTOMERS_TRAIN` / `NUM_CUSTOMERS_TEST`：客户类型数量。
- `NUM_CONSTRAINTS_TRAIN` / `NUM_CONSTRAINTS_TEST`：约束数量。
- `EPOCHS`：训练轮数。
- `LEARNING_RATE`：学习率。
- `DEVICE`：自动选择 CUDA 或 CPU。

## 模型与图结构

项目将每个优化实例构造成异构图，包含三类节点：

- `customer`：客户类型节点，特征包含客户占比 `alpha`。
- `product`：产品节点，特征包含产品收益 `r`。
- `constraint`：约束节点，特征包含约束右端项 `b`。

边类型包括：

- `customer -> product`：表示客户对产品的吸引力权重 `v`。
- `product -> customer`：反向消息传递边。
- `product -> constraint`：表示产品在约束矩阵 `A` 中的系数。
- `constraint -> product`：反向消息传递边。

模型使用 `GENConv` 作为基础消息传递层，并通过 `to_hetero` 转换为异构图模型。最终输出每个产品被选中的概率。

## 策略说明

- `GI`：根据模型输出概率排序，依次选择概率最高的产品组合，并保留满足约束且收益最高的方案。
- `GILS`：以 GI 结果为初始解，进一步进行局部搜索，包括加入、删除和替换产品。
- `GIP`：将模型输出概率作为线性目标系数，调用 Gurobi 求解一个 0-1 线性规划。

最终评估指标为：

```text
策略收益 / Gurobi 最优收益
```

该比例越接近 100%，说明策略效果越接近最优解。

## 实验结果

在当前默认配置下运行 `python main.py`，训练 100 个 epoch 后得到如下结果：

```text
Restricted license - for non-production use only - expires 2027-11-29
Epoch [5/100] | Train Loss: 0.5809 | Val Loss: 0.6916
Epoch [10/100] | Train Loss: 0.5633 | Val Loss: 0.6796
Epoch [15/100] | Train Loss: 0.5421 | Val Loss: 0.6543
Epoch [20/100] | Train Loss: 0.5270 | Val Loss: 0.6239
Epoch [25/100] | Train Loss: 0.5073 | Val Loss: 0.5973
Epoch [30/100] | Train Loss: 0.4891 | Val Loss: 0.5728
Epoch [35/100] | Train Loss: 0.4731 | Val Loss: 0.5516
Epoch [40/100] | Train Loss: 0.4567 | Val Loss: 0.5335
Epoch [45/100] | Train Loss: 0.4414 | Val Loss: 0.5212
Epoch [50/100] | Train Loss: 0.4283 | Val Loss: 0.5091
Epoch [55/100] | Train Loss: 0.4114 | Val Loss: 0.4958
Epoch [60/100] | Train Loss: 0.3956 | Val Loss: 0.4811
Epoch [65/100] | Train Loss: 0.3823 | Val Loss: 0.4677
Epoch [70/100] | Train Loss: 0.3703 | Val Loss: 0.4543
Epoch [75/100] | Train Loss: 0.3548 | Val Loss: 0.4412
Epoch [80/100] | Train Loss: 0.3414 | Val Loss: 0.4268
Epoch [85/100] | Train Loss: 0.3293 | Val Loss: 0.4164
Epoch [90/100] | Train Loss: 0.3170 | Val Loss: 0.4050
Epoch [95/100] | Train Loss: 0.3060 | Val Loss: 0.3934
Epoch [100/100] | Train Loss: 0.2946 | Val Loss: 0.3821
```

最终策略评估结果如下：

```text
==================================================
Policy               | Avg Ratio
--------------------------------------------------
GI                   |             84.57%
GILS                 |             98.02%
GIP                  |             96.75%
==================================================
```

从结果来看，模型训练过程中训练损失和验证损失均持续下降，说明模型能够学习到产品选择标签中的有效模式。在三种策略中，`GILS` 的平均收益比例最高，达到 `98.02%`，最接近 Gurobi 最优解；`GIP` 达到 `96.75%`，同样具有较好的近似效果；`GI` 为 `84.57%`，作为基础排序策略效果相对较弱。

## 注意事项

- 项目运行时间主要受 Gurobi 求解标签的影响，增加样本数或产品数会显著增加耗时。
- 当前代码没有固定随机种子，每次运行生成的数据和训练结果可能不同。
- `PATIENCE` 参数目前在代码中尚未实际用于早停逻辑。
- 如果没有 GPU，程序会自动使用 CPU 运行，但训练速度会变慢。

