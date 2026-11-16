"""Audit completed KNN/DT train-CV runs without official test or model pickle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_source_hashes(expected, *, source_root=None):
    """Verify exact historical bytes, optionally against a preserved source tree.

    A snapshot is not current source: report working-tree drift separately and
    never update config/source hashes to hide a later refactor.
    """
    archived_root = Path(source_root).resolve() if source_root is not None else ROOT
    checks = []
    for relative, fingerprint in expected.items():
        path = archived_root / relative
        actual = sha256(path) if path.is_file() else None
        current_path = ROOT / relative
        current = sha256(current_path) if current_path.is_file() else None
        checks.append({"path": relative, "expected_sha256": fingerprint,
                       "verified_sha256": actual, "current_sha256": current,
                       "current_source_matches": current == fingerprint})
        if actual != fingerprint:
            raise ValueError(f"Historical source bytes do not match: {relative} ({archived_root})")
    return {"verified_from": str(archived_root), "historical_source_verified": True,
            "current_source_matches": all(row["current_source_matches"] for row in checks),
            "checks": checks}


def check(run_dir, *, dt_run=None, source_root=None):
    from scripts.model_artifacts import load_model_runs, resolve_evidence
    runs, selection, config = load_model_runs(run_dir, dt_run)
    run = runs["knn"]
    assert selection["official_test_read"] is False
    assert selection["official_test_evaluated"] is False
    assert selection["status"] == "completed_train_cv"
    assert config["representations"] == ["hog"]
    train = pd.read_csv(Path(config["dataset_root"]) / "Train.csv")
    train.insert(0, "CSVRow", np.arange(len(train)))
    manifest = pd.read_csv(run / "fold_manifest.csv")
    assert len(train) == len(manifest) == 39209
    assert manifest.Group.nunique() == 1307
    assert manifest.groupby("Group").ValidationFold.nunique().eq(1).all()
    assert manifest[["CSVRow", "Path", "ClassId"]].equals(train[["CSVRow", "Path", "ClassId"]])
    source_audit = check_source_hashes(config["source_sha256"], source_root=source_root)
    for item in selection["algorithms"]:
        algorithm = item["algorithm"]
        run = runs[algorithm]
        cv = pd.read_csv(run / f"{algorithm}_candidates.csv")
        expected_count = 28 if algorithm == "knn" else 11
        assert len(cv) == expected_count
        assert cv.candidate_id.nunique() == expected_count
        assert cv.iloc[0].candidate_id == f"{algorithm}_baseline"
        assert cv.iloc[np.argmax(cv.mean_f1_macro)].candidate_id == item["selected"]["candidate_id"]
        model = resolve_evidence(item["model_file"], item["model_sha256"])
        assert sha256(model) == item["model_sha256"]
        for role in ("baseline", "selected"):
            pred = pd.read_csv(run / f"oof_{algorithm}_{role}.csv")
            assert pred[["CSVRow", "Path", "ClassId", "Group", "ValidationFold"]].equals(manifest)
            assert pred.PredictedClassId.isin(range(43)).all()
            scores = []
            for fold in range(3):
                part = pred.loc[pred.ValidationFold.eq(fold)]
                scores.append(f1_score(part.ClassId, part.PredictedClassId,
                                       labels=list(range(43)), average="macro", zero_division=0))
            assert np.isclose(np.mean(scores), item[role]["mean_f1_macro"], atol=1e-12, rtol=0)
        print(f"{algorithm}: candidates, OOF alignment, metrics, source and model hashes passed")
    result = {"run_id": selection["run_id"], "train_rows": len(train),
              "tracks": manifest.Group.nunique(), "official_test_read": False,
              "joblib_deserialized": False, "status": "passed",
              "source_audit": source_audit}
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knn-run", type=Path, required=True)
    parser.add_argument("--dt-run", type=Path, required=True)
    parser.add_argument("--source-root", type=Path,
                        help="Verify historical source hashes against an archived source tree; "
                             "working-tree drift is reported separately (default: strict current source)")
    args = parser.parse_args(argv)
    check(args.knn_run, dt_run=args.dt_run, source_root=args.source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
