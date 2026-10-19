# SAMF 部分模型模块库

本仓库提供与论文 **SAMF: A Structure-Adaptive Modular Framework for Remaining
Useful Life Prediction of Lithium-Ion Batteries** 相关的部分编码器和预测器实现，
以及使用合成输入演示模块注册、模型组合、维度对齐和前向计算的代码。

这是部分模型库的公开版本，不是完整 SAMF 框架，也不是论文实验结果的完整复现包。
由于实验室知识产权限制，本仓库不提供完整框架、训练和测试代码，也不包含真实电池数据、
实验输出、训练权重或内部实验配置。PG-EPM 匹配机制、七维画像计算、特征提取、
物理一致性和单调一致性损失、配置搜索、数据划分及图形界面均不在本次公开范围内。

## 公开内容

| 类型 | 模块或文件 |
| --- | --- |
| 4 种编码器 | `MLPEncoder`、`CNNEncoder`、`LSTMEncoder`、`SCCNNLSTMEncoder` |
| 3 种预测器 | `MLPPredictor`、`TransformerPredictor`、`AttentivePoolingPredictor` |
| 模块连接 | 独立注册表、显式配置构建、线性投影或恒等映射 |
| 演示 | `demo.py`，仅使用随机生成的张量和随机初始化参数 |
| 配置 | `examples/` 中 4 份独立 JSON 示例 |
| 细节 | `docs/MODULES.md` 中的接口和默认参数，`docs/PROVENANCE.md` 中的代码来源与改动 |

这 4 种编码器和 3 种预测器仅是论文模型库的一个子集，不等同于论文中的 12 种编码器、
10 种预测器及 120 种配置。示例不是论文的特定数据集配置，也不对应其配置编号。

## 安装与演示

使用 Python 3.9 或更高版本，以及与该 Python 版本和操作系统兼容的 PyTorch 2.x。
演示使用 CPU，不依赖 CUDA，也不依赖原始 BatteryML 项目。以下命令供使用者执行；
整理本仓库时没有执行安装、导入、演示或测试。

在仓库根目录创建环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS 的环境激活命令为 `source .venv/bin/activate`。
若需在仓库外部导入，可额外使用 `python -m pip install -e .`。

```powershell
python demo.py --list-modules
python demo.py --config examples/mlp_mlp.json
python demo.py --config examples/cnn_transformer.json
python demo.py --config examples/lstm_attention.json
python demo.py --config examples/fusion_mlp.json
```

程序按 JSON 配置选择模块，不执行 PG-EPM 推荐。默认生成形状为 `[4, 16, 8]` 的合成输入，
设计输出形状为 `[4, 1]`，并打印实际输出形状、所选模块、各阶段维度、对齐层及参数总量。
这是接口演示，不进行训练，也不读取测试集或计算 RMSE、MAE、MAPE、配置搜索耗时。
未训练的标量输出不能解释为具有预测意义的 RUL。

`model.eval()` 只切换 Dropout 等层的行为，并不表示运行了论文测试流程。
随机种子用于明确演示输入和初始化，不代表已获得跨版本一致的数值结果。

## 数据流与配置约定

```text
合成特征 [B, T, C]
    -> 展平 [B, T*C]
    -> 编码器 [B, E]
    -> 线性投影或恒等映射 [B, A]
    -> 预测器 [B, 1]
```

- `T`、`C` 分别来自 `input.sequence_length`、`input.feature_channels`。
- `E` 是编码器的 `output_dim`，`A` 是配置中的 `alignment_dim`。
- 编码器和预测器的 `input_dim` 由构建器推导，不能在 `params` 内重复指定。
- 已公开时序编码器的 `seq_len` 必须与 `T` 一致。
- Transformer 和注意力池化预测器的 `seq_len` 表示对编码后向量进行分组的组数，
  必须整除 `A`，不自动等同于原始循环数。
- 参数取值用于说明模块结构和运行接口，不是对论文训练超参数或性能的公开承诺。

特别说明：所提供的 `TransformerPredictor` 实际使用 PyTorch `TransformerEncoder`
层加池化和回归头，不是自回归解码器。`SCCNNLSTMEncoder` 保留来源中的类名，
当前实现将完整特征序列输入 CNN 和 LSTM 两个分支，没有另设容量专用通道。

## 来源、许可与核验状态

编码器和预测器从用户提供的 BatteryML 衍生代码快照中抽取，保留所选模块的网络结构，
移除了原项目注册表、训练框架及训练专用后端控制依赖，并增加独立的演示封装。
README 和 `docs/` 记录了全部保留模块及主要调整，未复制原项目的整套工程。

原文件中的 Microsoft Corporation 版权声明及 MIT 许可证已保留，具体见 `LICENSE`、
`NOTICE.md`。该声明不表示 Microsoft 创作或认可了 SAMF。

按本次整理要求，没有运行代码或执行测试；本仓库不声称已通过运行验证或完成论文结果复现。
