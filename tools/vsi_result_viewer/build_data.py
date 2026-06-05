#!/usr/bin/env python3
"""Build static data files for the VSI-Bench result viewer."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


META_KEYS = {
    "id",
    "dataset",
    "scene_name",
    "question_type",
    "question",
    "ground_truth",
    "options",
    "pruned",
    "prediction",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def flatten_response(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return ""
        if len(value) == 1:
            return flatten_response(value[0])
        return " | ".join(flatten_response(v) for v in value)
    return str(value)


def score_payload(score_obj: dict[str, Any]) -> tuple[str, float | None]:
    for key, value in score_obj.items():
        if key in META_KEYS:
            continue
        if isinstance(value, (int, float)):
            return key, float(value)
    return "", None


def status_for(prediction: str, score: float | None) -> str:
    if not prediction.strip():
        return "empty"
    if score is None:
        return "unknown"
    if score >= 0.999999:
        return "correct"
    if score <= 0.000001:
        return "wrong"
    return "partial"


def run_label(path: Path, root: Path, data: dict[str, Any], sample_count: int) -> str:
    args = data.get("args", {})
    suffix = args.get("log_samples_suffix")
    model_args = args.get("model_args", "")
    model = ""
    for part in str(model_args).split(","):
        if part.startswith("model_version="):
            model = part.split("=", 1)[1]
            break
    pieces = [p for p in [suffix, model] if p]
    if pieces:
        return " / ".join(dict.fromkeys(pieces))
    try:
        return str(path.parent.relative_to(root))
    except ValueError:
        return path.parent.name or f"{sample_count} samples"


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_type: dict[str, dict[str, Any]] = {}
    by_dataset: dict[str, int] = {}
    for row in samples:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1
        item = by_type.setdefault(
            row["question_type"],
            {"total": 0, "correct": 0, "partial": 0, "wrong": 0, "empty": 0, "score_sum": 0.0},
        )
        item["total"] += 1
        item[row["status"]] = item.get(row["status"], 0) + 1
        if isinstance(row["score"], (int, float)):
            item["score_sum"] += row["score"]

    type_rows = []
    for name, item in sorted(by_type.items()):
        total = item["total"] or 1
        type_rows.append(
            {
                "name": name,
                "total": item["total"],
                "correct": item.get("correct", 0),
                "partial": item.get("partial", 0),
                "wrong": item.get("wrong", 0),
                "empty": item.get("empty", 0),
                "avg_score": item["score_sum"] / total,
            }
        )

    return {
        "total": len(samples),
        "by_status": by_status,
        "by_type": type_rows,
        "by_dataset": dict(sorted(by_dataset.items())),
    }


def normalize_run(path: Path, root: Path, output_dir: Path) -> dict[str, Any] | None:
    raw = load_json(path)
    logs = raw.get("logs") if isinstance(raw, dict) else None
    if not isinstance(logs, list):
        return None

    samples: list[dict[str, Any]] = []
    for idx, item in enumerate(logs):
        doc = item.get("doc") or {}
        score_obj = item.get("vsibench_score") or {}
        metric, score = score_payload(score_obj)
        prediction = flatten_response(score_obj.get("prediction", item.get("filtered_resps")))
        ground_truth = flatten_response(score_obj.get("ground_truth", doc.get("ground_truth", item.get("target"))))
        question_type = score_obj.get("question_type", doc.get("question_type", "unknown"))
        sample = {
            "row": idx,
            "id": score_obj.get("id", doc.get("id", item.get("doc_id", idx))),
            "doc_id": item.get("doc_id", idx),
            "dataset": score_obj.get("dataset", doc.get("dataset", "")),
            "scene_name": score_obj.get("scene_name", doc.get("scene_name", "")),
            "question_type": question_type,
            "question_family": question_type.replace("_hard", "").replace("_medium", "").replace("_easy", ""),
            "question": score_obj.get("question", doc.get("question", "")),
            "ground_truth": ground_truth,
            "prediction": prediction,
            "options": score_obj.get("options", doc.get("options")),
            "pruned": bool(score_obj.get("pruned", doc.get("pruned", False))),
            "metric": metric,
            "score": score,
            "status": status_for(prediction, score),
            "prompt": flatten_response((item.get("arguments") or [""])[0]),
            "target": flatten_response(item.get("target")),
        }
        samples.append(sample)

    if not samples:
        return None

    args = raw.get("args", {}) if isinstance(raw, dict) else {}
    result_path = path.with_name("results.json")
    results = load_json(result_path) if result_path.exists() else {}
    metrics = (
        results.get("results", {})
        .get("vsibench", {})
        .get("vsibench_score,none", {})
        if isinstance(results, dict)
        else {}
    )
    label = run_label(path, root, raw, len(samples))
    safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)[:90]
    if not safe_id:
        safe_id = f"run_{len(samples)}"

    data_file = f"{safe_id}.json"
    write_json(
        output_dir / data_file,
        {
            "run": {
                "id": safe_id,
                "label": label,
                "source": str(path),
                "results_source": str(result_path) if result_path.exists() else "",
                "model": args.get("model", ""),
                "model_args": args.get("model_args", ""),
                "created_at": raw.get("time", ""),
            },
            "summary": summarize(samples),
            "metrics": metrics,
            "samples": samples,
        },
    )

    summary = summarize(samples)
    return {
        "id": safe_id,
        "label": label,
        "data_file": data_file,
        "source": str(path),
        "results_source": str(result_path) if result_path.exists() else "",
        "total": summary["total"],
        "summary": summary,
        "metrics": metrics,
        "model_args": args.get("model_args", ""),
    }


def discover_runs(root: Path, min_samples: int) -> list[Path]:
    candidates = sorted(root.glob("logs/**/vsibench.json"))
    runs = []
    for path in candidates:
        try:
            raw = load_json(path)
            logs = raw.get("logs") if isinstance(raw, dict) else None
            if isinstance(logs, list) and len(logs) >= min_samples:
                runs.append(path)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {path}: {exc}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="VSI-Bench repository root")
    parser.add_argument("--output", default="tools/vsi_result_viewer/data", help="Output data directory")
    parser.add_argument("--min-samples", type=int, default=1000, help="Only include runs with at least this many samples")
    parser.add_argument("paths", nargs="*", help="Explicit vsibench.json paths")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = (root / args.output).resolve()
    paths = [Path(p).resolve() for p in args.paths] if args.paths else discover_runs(root, args.min_samples)

    runs = []
    for path in paths:
        run = normalize_run(path, root, output_dir)
        if run:
            runs.append(run)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "runs": runs,
    }
    write_json(output_dir / "runs.json", manifest)
    print(f"Wrote {len(runs)} run(s) to {output_dir}")


if __name__ == "__main__":
    main()
