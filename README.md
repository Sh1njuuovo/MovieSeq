# MixFormer 序列推荐复现

这个仓库提供一个独立编写的轻量实现，用隐式反馈预测用户是否会选择候选物品。模型把全局自注意力、局部卷积和候选物品注意力结合起来。输入是历史物品 ID 和候选物品 ID，输出为一个分数。

## 运行

需要 Python 3.9 及以上版本和 PyTorch。安装依赖后，可以先跑合成数据，确认训练流程可用。

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m mixformer.train examples/demo.jsonl --epochs 1 --dim 16 --heads 2
PYTHONPATH=src python -m unittest discover -s tests -v
```

使用 MovieLens-1M 时，先取得 `ratings.dat`，再生成训练样本。原始数据和模型权重均不进入版本库。

```bash
PYTHONPATH=src python -m mixformer.prepare data/ml-1m/ratings.dat data/train.jsonl
PYTHONPATH=src python -m mixformer.train data/train.jsonl --output runs/ml1m
```

`prepare` 保留评分不低于 4 的交互，按时间构造历史与正样本，并从用户未交互物品中抽取负样本。物品 ID 会重新映射为从 1 开始的整数，0 用于填充。训练入口接受每行一个 JSON 对象，字段为 `history`、`candidate` 和 `label`。

## 范围

这个版本用于验证模型和数据流程。它没有复现工业规模训练设置，也没有把旧实验的数值当作当前代码的结果。训练输出包含损失和权重，正式评估需要在独立测试集上另行执行。

`results/historical_ablation.json` 仅保存先前实验的指标摘要。该文件中的指标尚未用本仓库代码重新得到。

`analysis/` 保存两份独立编写的旧实验统计脚本。它们读取既有实验输出，其中 `summarize_evidence.py` 需要 NumPy。
