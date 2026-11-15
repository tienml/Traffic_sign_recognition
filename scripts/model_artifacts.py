"""Shared paths for the KNN/DT experiment records and relocated evidence.

This is artifact I/O only; model algorithms live in src.models. Historical hashes
are never rewritten: a moved file or archived source must match the recorded bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELOCATIONS = "logs/integration/20261115-models/relocations.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_evidence(relative, expected=None, *, project_root=ROOT):
    """Resolve a recorded repository-relative path, checking exact bytes if given."""
    root = Path(project_root).resolve()
    relative = str(relative).replace("\\", "/")
    direct = (root / relative).resolve()
    if not direct.is_relative_to(root):
        raise ValueError("Evidence path escapes repository")
    candidates = [direct]
    manifest = root / RELOCATIONS
    if manifest.is_file():
        relocation = json.loads(manifest.read_text(encoding="utf-8"))
        moved = relocation["paths"].get(relative)
        if moved:
            candidates.append(root / moved)
        if expected is not None:
            candidates.extend(root / item["archived_path"]
                              for item in relocation["historical_sources"]
                              if item["original_path"] == relative and item["sha256"] == expected)
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate.is_relative_to(root) and candidate.is_file()
                and (expected is None or digest(candidate) == expected)):
            return candidate
    raise ValueError(f"Missing or changed recorded evidence: {relative}")


def load_model_runs(knn_run, dt_run=None):
    """Read separate KNN/DT runs, validating shared train/fold/feature evidence.

    A single combined run is accepted only for archived/synthetic callers. New
    command-line runs always supply the two algorithm-specific directories.
    """
    runs = {"knn": Path(knn_run).resolve(), "dt": Path(dt_run or knn_run).resolve()}
    selections, configs, items = {}, {}, []
    for algorithm, run in runs.items():
        selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if (selection.get("status") not in ("completed_train_cv", "smoke_only")
                or selection.get("official_test_read") is not False
                or selection.get("official_test_evaluated") is not False
                or config.get("official_test_read") is not False):
            raise ValueError("Require completed train-only KNN/DT runs")
        matching = [item for item in selection["algorithms"] if item["algorithm"] == algorithm]
        if len(matching) != 1:
            raise ValueError(f"Expected exactly one {algorithm} selection")
        for name, key in (("fold_manifest.csv", "fold_manifest_sha256"),
                          ("feature_audit.json", "feature_audit_sha256")):
            if digest(run / name) != selection[key]:
                raise ValueError(f"Changed {algorithm} {name}")
        selections[algorithm], configs[algorithm] = selection, config
        items.append(matching[0])
    if selections["knn"]["status"] != selections["dt"]["status"]:
        raise ValueError("Cannot mix smoke and full model runs")
    for name in ("fold_manifest.csv", "feature_audit.json", "train_image_audit.csv"):
        if digest(runs["knn"] / name) != digest(runs["dt"] / name):
            raise ValueError(f"KNN/DT must share the same {name}")
    if configs["knn"].get("train_sha256") != configs["dt"].get("train_sha256"):
        raise ValueError("KNN/DT Train.csv fingerprints differ")
    selection = {**selections["knn"], "algorithms": items}
    config = {**configs["knn"], "dt_candidates": configs["dt"].get("dt_candidates", [])}
    return runs, selection, config
