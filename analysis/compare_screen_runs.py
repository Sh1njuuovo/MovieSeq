"""比较同一组筛查运行，核对协议一致性并按预先设定的筛查线给出工程判断。

用法
    python repro/compare_screen_runs.py --out <输出目录> <run_dir> <run_dir> [<run_dir> ...]

第一个 run 视为对照，其余为候选。阈值是执行前固定的资源决策规则，不是显著性结论。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NDCG_MIN_DELTA = 0.003
HR_MAX_DROP = 0.005
TRAIN_TIME_MAX_INCREASE = 0.10


def load(run_dir: Path) -> dict:
    with open(run_dir / "metrics.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def paired_consistency(baseline: dict, candidate: dict) -> dict:
    fields = [
        "train_seed",
        "candidate_seed",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "eval_negatives",
        "top_k",
        "max_seq_len",
        "train_samples",
        "val_samples",
        "data_sha256",
        "seq_ffn_multiplier",
    ]
    mismatches = {field: {"baseline": baseline.get(field), "candidate": candidate.get(field)}
                  for field in fields if baseline.get(field) != candidate.get(field)}
    same_train_order = baseline.get("training_batch_fingerprints") == candidate.get("training_batch_fingerprints")
    same_candidates = baseline.get("val_candidate_fingerprint") == candidate.get("val_candidate_fingerprint")
    return {
        "protocol_field_mismatches": mismatches,
        "protocol_fields_match": not mismatches,
        "training_batch_fingerprints_match": same_train_order,
        "val_candidate_fingerprint_match": same_candidates,
        "identical_protocol": (not mismatches) and same_train_order and same_candidates,
    }


def summarize(baseline: dict, candidate: dict) -> dict:
    base_train = sum(record["train_seconds"] for record in baseline["history"])
    cand_train = sum(record["train_seconds"] for record in candidate["history"])
    base_total = sum(record["epoch_seconds"] for record in baseline["history"])
    cand_total = sum(record["epoch_seconds"] for record in candidate["history"])
    best = baseline["best_record"]
    cand_best = candidate["best_record"]
    top_k = baseline["top_k"]
    ndcg_delta = cand_best[f"ndcg@{top_k}"] - best[f"ndcg@{top_k}"]
    hr_delta = cand_best[f"hr@{top_k}"] - best[f"hr@{top_k}"]
    train_increase = (cand_train - base_train) / base_train if base_train else float("nan")
    total_increase = (cand_total - base_total) / base_total if base_total else float("nan")
    passed = (
        ndcg_delta >= NDCG_MIN_DELTA
        and hr_delta >= -HR_MAX_DROP
        and train_increase <= TRAIN_TIME_MAX_INCREASE
    )
    return {
        "top_k": top_k,
        "baseline_best_epoch": baseline["best_epoch"],
        "baseline_best_val_ndcg": best[f"ndcg@{top_k}"],
        "baseline_best_val_hr": best[f"hr@{top_k}"],
        "candidate_best_epoch": candidate["best_epoch"],
        "candidate_best_val_ndcg": cand_best[f"ndcg@{top_k}"],
        "candidate_best_val_hr": cand_best[f"hr@{top_k}"],
        "ndcg_delta": ndcg_delta,
        "hr_delta": hr_delta,
        "baseline_train_seconds_total": base_train,
        "candidate_train_seconds_total": cand_train,
        "train_time_relative_increase": train_increase,
        "baseline_epoch_seconds_total": base_total,
        "candidate_epoch_seconds_total": cand_total,
        "epoch_time_relative_increase": total_increase,
        "baseline_params_m": baseline["params_m"],
        "candidate_params_m": candidate["params_m"],
        "baseline_reference_latency_median_ms": baseline["reference_latency"]["median_ms"],
        "candidate_reference_latency_median_ms": candidate["reference_latency"]["median_ms"],
        "baseline_peak_allocated_bytes": baseline["peak_allocated_bytes"],
        "candidate_peak_allocated_bytes": candidate["peak_allocated_bytes"],
        "baseline_peak_reserved_bytes": baseline["peak_reserved_bytes"],
        "candidate_peak_reserved_bytes": candidate["peak_reserved_bytes"],
        "thresholds": {
            "ndcg_min_delta": NDCG_MIN_DELTA,
            "hr_max_drop": HR_MAX_DROP,
            "train_time_max_relative_increase": TRAIN_TIME_MAX_INCREASE,
        },
        "screening_passed": passed,
        "ndcg_rule_met": ndcg_delta >= NDCG_MIN_DELTA,
        "hr_rule_met": hr_delta >= -HR_MAX_DROP,
        "train_time_rule_met": train_increase <= TRAIN_TIME_MAX_INCREASE,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()

    metric_map = {run: load(Path(run)) for run in args.runs}
    baseline_run = args.runs[0]
    baseline = metric_map[baseline_run]
    payload = {"baseline_run": baseline_run, "runs": {}, "comparisons": {}}
    for run, metrics in metric_map.items():
        payload["runs"][run] = {
            "mode": metrics["mode"],
            "train_seed": metrics["train_seed"],
            "best_epoch": metrics["best_epoch"],
            "best_val_ndcg": metrics["best_record"][f"ndcg@{metrics['top_k']}"],
            "best_val_hr": metrics["best_record"][f"hr@{metrics['top_k']}"],
            "params_m": metrics["params_m"],
            "approx_gflops_per_batch": metrics["approx_gflops_per_batch"],
            "history": metrics["history"],
        }
        payload["comparisons"][run] = {
            "consistency": paired_consistency(baseline, metrics),
            "summary": summarize(baseline, metrics),
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)

    lines = ["| 运行 | 模式 | 最佳轮 | 验证 NDCG@10 | 验证 HR@10 | 参数量(M) | 训练耗时增加 | 筛查线 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for run in args.runs:
        entry = payload["runs"][run]
        comparison = payload["comparisons"][run]["summary"]
        lines.append(
            f"| {Path(run).name} | {entry['mode']} | {entry['best_epoch']} | "
            f"{entry['best_val_ndcg']:.6f} | {entry['best_val_hr']:.6f} | {entry['params_m']:.6f} | "
            f"{comparison['train_time_relative_increase'] * 100:.2f}% | "
            f"{'达到' if comparison['screening_passed'] else '未达到'} |"
        )
    table = "\n".join(lines)
    (out_dir / "comparison.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    for run in args.runs[1:]:
        consistency = payload["comparisons"][run]["consistency"]
        print(f"{Path(run).name} 协议一致: {consistency['identical_protocol']} "
              f"(样本顺序={consistency['training_batch_fingerprints_match']}, "
              f"验证候选={consistency['val_candidate_fingerprint_match']}, "
              f"字段不一致={consistency['protocol_field_mismatches']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
