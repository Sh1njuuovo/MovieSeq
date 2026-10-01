# MovieSeq

MovieSeq is a compact movie recommendation project built around candidate-aware sequence modeling. Given a user's viewing history and a candidate movie, it predicts a relevance score for that candidate. The repository includes MovieLens-1M preprocessing, a PyTorch model, a training command, and small tests.

## Method

The model combines three views of the interaction history.

1. **Global context.** Item and position embeddings enter multi-head self-attention so each history item can use information from the full sequence.
2. **Local context.** A depthwise convolution with a three-item window captures nearby transitions. A learned gate combines local and global representations at each position.
3. **Candidate-aware pooling.** The candidate embedding attends to the mixed history. An MLP scores the candidate together with the resulting history vector.

Training uses binary cross-entropy on positive and sampled negative user-candidate pairs. Zero is reserved for padding, and padding positions are masked during attention and pooling. The core implementation is in [`src/mixformer/model.py`](src/mixformer/model.py).

## Data flow

`ratings.dat` → chronological positive interactions → history/candidate pairs → negative sampling → JSONL → model training

The MovieLens converter keeps ratings of at least 4, maps movie IDs to positive integers, and creates a training pair for each positive interaction after a user's first one. Each pair uses only earlier positive interactions as its history. Negatives are sampled from movies the user never rated positively. The converter defaults to one negative per positive pair and a maximum history of 50 items.

Each JSONL record has this shape:

```json
{"history": [1, 8, 23], "candidate": 42, "label": 1}
```

## Quick start

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

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/mixformer/prepare.py` | Convert MovieLens ratings to implicit-feedback examples |
| `src/mixformer/model.py` | Global and local history mixing with candidate-aware scoring |
| `src/mixformer/train.py` | Train the model and save a checkpoint and loss report |
| `analysis/` | Offline scripts for inspecting earlier experiment outputs |
| `results/` | A small historical ablation summary |
| `tests/` | Padding, model, preprocessing, and training checks |

The historical metrics in `results/` came from an earlier experiment and have not been reproduced with the current code. This repository is a small-scale implementation; it does not reproduce the paper's industrial feature set or serving system.

## Paper

[MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders](https://arxiv.org/abs/2602.14110)
