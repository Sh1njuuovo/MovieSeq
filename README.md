# MovieSeq

## 中文

MovieSeq 是一个面向电影推荐的轻量级序列模型。输入用户看过的电影和一个候选电影，模型输出该候选的相关性分数。仓库包含 MovieLens-1M 数据转换、PyTorch 模型、训练入口和测试。

### 方法

1. **全局序列建模。** 将电影 ID 和位置编码相加，再用多头自注意力读取完整历史。
2. **局部序列建模。** 用窗口大小为 3 的逐通道卷积提取相邻交互的变化，并用可学习门控融合全局与局部表示。
3. **候选感知打分。** 候选电影的向量对历史表示计算注意力权重。汇总后的历史向量与候选向量一起进入 MLP，得到预测分数。

训练使用正样本和采样负样本，损失函数为二元交叉熵。ID 0 用于填充，注意力和汇总操作会跳过填充位置。模型代码位于 [`src/mixformer/model.py`](src/mixformer/model.py)。

### 数据流程

1. 从 `ratings.dat` 中保留评分不低于 4 的交互，并按时间排列每位用户的正反馈。
2. 将电影 ID 映射为从 1 开始的整数。对首条之后的每次正反馈，使用更早的交互构造历史。
3. 从用户没有给出正反馈的电影中抽取负样本，写入 JSONL。默认每个正样本配一个负样本，历史最多保留 50 项。
4. 训练脚本读取 JSONL，保存模型权重和每轮训练损失。

每行数据的格式如下：

```json
{"history": [1, 8, 23], "candidate": 42, "label": 1}
```

### 快速开始

需要 Python 3.9 或更新版本。在仓库根目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

PYTHONPATH=src python -m mixformer.train examples/demo.jsonl \
  --epochs 1 --dim 16 --heads 2 --output runs/demo
```

示例数据是合成数据，只用于检查训练流程。运行后，`runs/demo/` 中会生成 `model.pt` 和 `metrics.json`。

使用 MovieLens-1M 时，先将 `ratings.dat` 放在 `data/ml-1m/ratings.dat`，再运行：

```bash
PYTHONPATH=src python -m mixformer.prepare \
  data/ml-1m/ratings.dat data/train.jsonl
PYTHONPATH=src python -m mixformer.train \
  data/train.jsonl --epochs 5 --output runs/ml1m
```

原始数据、转换后的 JSONL 和模型权重不会提交到 Git。当前训练入口只报告训练损失，尚未划分独立测试集或计算排序指标。

### 目录说明

| 路径 | 内容 |
| --- | --- |
| `src/mixformer/prepare.py` | 转换 MovieLens 评分数据并构造训练样本 |
| `src/mixformer/model.py` | 全局与局部序列融合、候选感知打分 |
| `src/mixformer/train.py` | 训练模型并保存权重和损失报告 |
| `analysis/` | 分析先前实验输出的离线脚本 |
| `results/` | 一份历史消融实验摘要 |
| `tests/` | 填充处理、模型、数据转换和训练测试 |

`results/` 中的历史指标尚未用当前代码重新得到。本项目实现了小规模流程，没有覆盖论文中的工业特征体系和在线服务方案。

### 论文

[MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders](https://arxiv.org/abs/2602.14110)

## English

MovieSeq is a compact movie recommendation project built around candidate-aware sequence modeling. Given a user's viewing history and a candidate movie, it predicts a relevance score for that candidate. The repository includes MovieLens-1M preprocessing, a PyTorch model, a training command, and small tests.

### Method

The model combines three views of the interaction history.

1. **Global context.** Item and position embeddings enter multi-head self-attention so each history item can use information from the full sequence.
2. **Local context.** A depthwise convolution with a three-item window captures nearby transitions. A learned gate combines local and global representations at each position.
3. **Candidate-aware pooling.** The candidate embedding attends to the mixed history. An MLP scores the candidate together with the resulting history vector.

Training uses binary cross-entropy on positive and sampled negative user-candidate pairs. Zero is reserved for padding, and padding positions are masked during attention and pooling. The core implementation is in [`src/mixformer/model.py`](src/mixformer/model.py).

### Data flow

`ratings.dat` → chronological positive interactions → history/candidate pairs → negative sampling → JSONL → model training

The MovieLens converter keeps ratings of at least 4, maps movie IDs to positive integers, and creates a training pair for each positive interaction after a user's first one. Each pair uses only earlier positive interactions as its history. Negatives are sampled from movies the user never rated positively. The converter defaults to one negative per positive pair and a maximum history of 50 items.

Each JSONL record has this shape:

```json
{"history": [1, 8, 23], "candidate": 42, "label": 1}
```

### Quick start

Use Python 3.9 or newer. The commands below run from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

PYTHONPATH=src python -m mixformer.train examples/demo.jsonl \
  --epochs 1 --dim 16 --heads 2 --output runs/demo
```

The demo data is synthetic and only checks that the training path runs. `runs/demo/` contains `model.pt` and `metrics.json`.

To prepare MovieLens-1M, place its `ratings.dat` at `data/ml-1m/ratings.dat`, then run:

```bash
PYTHONPATH=src python -m mixformer.prepare \
  data/ml-1m/ratings.dat data/train.jsonl
PYTHONPATH=src python -m mixformer.train \
  data/train.jsonl --epochs 5 --output runs/ml1m
```

Raw data, generated JSONL, and checkpoints are excluded from Git. The training command currently reports training loss. It does not create a held-out split or compute ranking metrics.

### Repository layout

| Path | Purpose |
| --- | --- |
| `src/mixformer/prepare.py` | Convert MovieLens ratings to implicit-feedback examples |
| `src/mixformer/model.py` | Global and local history mixing with candidate-aware scoring |
| `src/mixformer/train.py` | Train the model and save a checkpoint and loss report |
| `analysis/` | Offline scripts for inspecting earlier experiment outputs |
| `results/` | A small historical ablation summary |
| `tests/` | Padding, model, preprocessing, and training checks |

The historical metrics in `results/` came from an earlier experiment and have not been reproduced with the current code. This repository is a small-scale implementation; it does not reproduce the paper's industrial feature set or serving system.

### Paper

[MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders](https://arxiv.org/abs/2602.14110)
