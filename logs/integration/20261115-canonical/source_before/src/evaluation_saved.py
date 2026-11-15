"""Pure evaluation of caller-supplied, saved prediction tables.

This module performs no file/image/model access, prediction, fitting, or plotting.
``PredictedClassId`` is the prediction column. CSVRow is the zero-based row
position in the original split CSV, not a DataFrame index. Fixed-label macro
metrics include absent classes as zeros; balanced accuracy averages recall only
across supported true classes, matching sklearn's unadjusted convention.

OOF checks verify a supplied manifest, not its provenance or training procedure.
Likewise, example selection and metadata strata describe errors, not their causes.
"""

from hashlib import sha256
from numbers import Integral, Real
import re

import numpy as np
import pandas as pd


LABELS = tuple(range(43))
_KEYS = ("CSVRow", "Path", "ClassId")
_PREDICTION = "PredictedClassId"
_ROI = ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")
_SPEED_CLASSES = frozenset((0, 1, 2, 3, 4, 5, 7, 8))
_ERROR_COLUMNS = (
    "true_class", "predicted_class", "count", "true_support",
    "error_share_within_true",
)


def _require_columns(frame, columns, name):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} has duplicate column names")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} must not be empty")


def _integer_values(series, name, minimum, maximum):
    values = series.to_numpy()
    if values.dtype.kind not in "iu":
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
               for value in values):
            raise ValueError(f"{name} must contain strict integers, not strings/floats/bools")
    if any(value < minimum or value > maximum for value in values):
        raise ValueError(f"{name} must be in {minimum}..{maximum}")
    return values.astype(np.int64)


def _integer_option(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _path_info(paths, classes, split=None):
    """Validate raw separators before parsing; canonicalize only for uniqueness."""
    canonical = []
    groups = []
    found_splits = set()
    for path, class_id in zip(paths, classes):
        if (not isinstance(path, str) or not path
                or any(ord(character) < 32 or character in '<>:"|?*' for character in path)):
            raise ValueError("Path must contain safe relative PNG path strings")
        normalized = path.replace("\\", "/")
        parts = normalized.split("/")
        if any(part in ("", ".", "..") or part.endswith((" ", ".")) for part in parts):
            raise ValueError(f"Unsafe relative Path: {path!r}")
        if len(parts) == 2 and parts[0] == "Test" and parts[1].endswith(".png"):
            found_split = "official_test"
            groups.append(None)
        elif len(parts) == 3 and parts[0] == "Train":
            match = re.fullmatch(r"([0-9]+)_([0-9]+)_([0-9]+)\.png", parts[2])
            if not match or parts[1] != str(int(class_id)) or int(match[1]) != class_id:
                raise ValueError(f"Train path/class disagreement: {path!r}")
            found_split = "train_oof"
            groups.append(f"{int(class_id)}:{int(match[2])}")
        else:
            raise ValueError(f"Expected Test/*.png or Train/class/class_track_frame.png: {path!r}")
        found_splits.add(found_split)
        canonical.append(normalized.casefold())
    if len(found_splits) != 1 or (split is not None and found_splits != {split}):
        raise ValueError("Paths do not belong to the requested single split")
    if len(set(canonical)) != len(canonical):
        raise ValueError("Path values must be unique, including separator/case aliases")
    return np.asarray(groups, dtype=object)


def _base_values(frame, name, split=None):
    _require_columns(frame, _KEYS, name)
    rows = _integer_values(frame["CSVRow"], f"{name}.CSVRow", 0, len(frame) - 1)
    if not np.array_equal(rows, np.arange(len(frame), dtype=np.int64)):
        raise ValueError(f"{name}.CSVRow must be unique 0..N-1 in original CSV order")
    classes = _integer_values(frame["ClassId"], f"{name}.ClassId", 0, 42)
    groups = _path_info(frame["Path"].to_numpy(), classes, split)
    return classes, groups


def _prediction_values(frame, name="frame", split=None):
    _require_columns(frame, (*_KEYS, _PREDICTION), name)
    classes, groups = _base_values(frame, name, split)
    predicted = _integer_values(frame[_PREDICTION], f"{name}.{_PREDICTION}", 0, 42)
    return classes, predicted, groups


def _same_keys(first, second, name):
    if len(first) != len(second) or any(
        not np.array_equal(first[column].to_numpy(), second[column].to_numpy())
        for column in _KEYS
    ):
        raise ValueError(f"{name} must match CSVRow/Path/ClassId exactly in CSV order")


def validate_predictions(predictions, metadata, *, split, manifest=None, expected_size=None):
    """Return a defensive copy after strict identity, label, path and OOF checks.

    Metadata must be the caller's actual root split CSV with CSVRow added. This
    function cannot establish its provenance. No requirement for all 43 classes
    is imposed on a table. OOF needs all three fold IDs and one fold per parsed
    class/track group, without importing model or preprocessing modules.
    """
    if split not in ("official_test", "train_oof"):
        raise ValueError("split must be 'official_test' or 'train_oof'")
    _, _, parsed_groups = _prediction_values(predictions, "predictions", split)
    _base_values(metadata, "metadata", split)
    _same_keys(predictions, metadata, "metadata")
    if expected_size is not None:
        size = _integer_option(expected_size, "expected_size", minimum=1)
        if len(predictions) != size:
            raise ValueError("Prediction count does not match expected_size")
    if split == "official_test":
        if manifest is not None:
            raise ValueError("A fold manifest is only valid for train_oof")
    else:
        _require_columns(predictions, ("Group", "ValidationFold"), "predictions")
        _require_columns(manifest, (*_KEYS, "Group", "ValidationFold"), "manifest")
        _base_values(manifest, "manifest", split)
        _same_keys(predictions, manifest, "manifest")
        assignment = None
        for name, table in (("predictions", predictions), ("manifest", manifest)):
            groups = table["Group"].to_numpy()
            if (any(not isinstance(group, str) for group in groups)
                    or not np.array_equal(groups, parsed_groups)):
                raise ValueError(f"{name}.Group must equal the parsed class:track group")
            folds = _integer_values(table["ValidationFold"], f"{name}.ValidationFold", 0, 2)
            if set(folds) != {0, 1, 2}:
                raise ValueError(f"{name} must contain folds 0, 1 and 2")
            group_folds = {}
            for group, fold in zip(groups, folds):
                if group in group_folds and group_folds[group] != fold:
                    raise ValueError(f"{name} splits a whole track across validation folds")
                group_folds[group] = fold
            if assignment is not None and not np.array_equal(assignment, folds):
                raise ValueError("Prediction folds do not match the manifest")
            assignment = folds
    return predictions.copy(deep=True)


def _label_values(labels):
    try:
        result = tuple(labels)
    except TypeError as error:
        raise ValueError("labels must be a nonempty sequence of unique class IDs") from error
    if (not result or any(isinstance(label, (bool, np.bool_))
                          or not isinstance(label, Integral) or not 0 <= label <= 42
                          for label in result) or len(set(result)) != len(result)):
        raise ValueError("labels must contain unique strict integer class IDs in 0..42")
    return np.asarray(result, dtype=np.int64)


def _divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64),
                     where=np.asarray(denominator) != 0)


def summarize_predictions(frame, *, labels=LABELS):
    """Return fixed-label metrics, class statistics, confusion arrays and errors.

    Unobserved true-class rows normalize to zero. Error pairs are ordered by
    decreasing count, then ascending true and predicted ID. Custom labels retain
    their supplied order and must include every observed true/predicted class.
    """
    actual, predicted, _ = _prediction_values(frame)
    label_values = _label_values(labels)
    if not np.isin(actual, label_values).all() or not np.isin(predicted, label_values).all():
        raise ValueError("labels must include every true and predicted class")
    lookup = np.full(43, -1, dtype=np.int64)
    lookup[label_values] = np.arange(len(label_values))
    size = len(label_values)
    confusion = np.bincount(lookup[actual] * size + lookup[predicted],
                            minlength=size * size).reshape(size, size).astype(np.int64)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    correct = confusion.diagonal().copy()
    precision = _divide(correct, predicted_support)
    recall = _divide(correct, support)
    f1 = _divide(2 * precision * recall, precision + recall)
    metrics = {
        "accuracy": float(correct.sum() / len(actual)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
    }
    classes = pd.DataFrame({
        "class_id": label_values, "support": support, "precision": precision,
        "recall": recall, "f1": f1, "correct": correct,
        "predicted_support": predicted_support,
    })
    true_index, predicted_index = np.nonzero(confusion)
    off_diagonal = true_index != predicted_index
    true_index, predicted_index = true_index[off_diagonal], predicted_index[off_diagonal]
    counts = confusion[true_index, predicted_index]
    errors = pd.DataFrame({
        "true_class": label_values[true_index],
        "predicted_class": label_values[predicted_index],
        "count": counts, "true_support": support[true_index],
        "error_share_within_true": _divide(counts, support[true_index]),
    }, columns=_ERROR_COLUMNS).sort_values(
        ["count", "true_class", "predicted_class"], ascending=[False, True, True],
        kind="stable").reset_index(drop=True)
    return {
        "metrics": metrics, "classes": classes, "confusion": confusion,
        "confusion_normalized": _divide(confusion, support[:, None]), "errors": errors,
    }


def paired_comparison(first, second, *, name_first, name_second):
    """Count paired outcomes; do not infer significance or independence of rows."""
    if (not isinstance(name_first, str) or not name_first.strip()
            or not isinstance(name_second, str) or not name_second.strip()
            or name_first == name_second):
        raise ValueError("Comparison names must be distinct nonempty strings")
    actual_first, predicted_first, _ = _prediction_values(first, "first")
    actual_second, predicted_second, _ = _prediction_values(second, "second")
    _same_keys(first, second, "second")
    correct_first, correct_second = actual_first == predicted_first, actual_second == predicted_second
    first_only = int(np.count_nonzero(correct_first & ~correct_second))
    second_only = int(np.count_nonzero(~correct_first & correct_second))
    return {
        "n": len(first), "both_correct": int(np.count_nonzero(correct_first & correct_second)),
        "first_only_correct": first_only, "second_only_correct": second_only,
        "both_wrong": int(np.count_nonzero(~correct_first & ~correct_second)),
        "accuracy_difference_first_minus_second": float((first_only - second_only) / len(first)),
        "discordant": first_only + second_only,
    }


def _numeric_values(series, name):
    values = series.to_numpy()
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in values):
        raise ValueError(f"{name} must contain finite numeric values")
    numeric = values.astype(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return numeric


def _category(value, name):
    if value is None or value is pd.NA:
        return "unknown"
    if not isinstance(value, (str, Real, np.bool_)):
        raise ValueError(f"{name} must contain scalar string/numeric categories")
    return "unknown" if pd.isna(value) else str(value)


def group_error_summary(frame, metadata, *, labels=LABELS):
    """Describe errors in fixed, observed metadata strata, never image causes.

    Dimensions are raw_min_dimension and roi_min_dimension (groups 'lt32',
    'ge32'), sign_family ('speed_limit', 'other'), and optional ColorId/ShapeId
    string categories (missing values become 'unknown'). ROI uses GTSRB's
    inclusive X2/Y2 coordinates. Other quality columns are deliberately ignored:
    adaptive thresholds must have been fitted separately on training folds.
    """
    actual, predicted, _ = _prediction_values(frame)
    label_values = _label_values(labels)
    if not np.isin(actual, label_values).all() or not np.isin(predicted, label_values).all():
        raise ValueError("labels must include every true and predicted class")
    _base_values(metadata, "metadata")
    _same_keys(frame, metadata, "metadata")
    _require_columns(metadata, ("Width", "Height", *_ROI), "metadata")
    optional = [column for column in ("ColorId", "ShapeId") if column in metadata.columns]
    joined = frame[[*_KEYS, _PREDICTION]].merge(
        metadata[[*_KEYS, "Width", "Height", *_ROI, *optional]],
        on=list(_KEYS), how="left", sort=False, validate="one_to_one")
    width, height, x1, y1, x2, y2 = (
        _numeric_values(joined[column], column) for column in ("Width", "Height", *_ROI))
    if (np.any(width <= 0) or np.any(height <= 0) or np.any(x1 < 0) or np.any(y1 < 0)
            or np.any(x2 < x1) or np.any(y2 < y1) or np.any(x2 >= width) or np.any(y2 >= height)):
        raise ValueError("Image dimensions and inclusive ROI coordinates must be valid and in bounds")
    dimensions = {
        "raw_min_dimension": np.where(np.minimum(width, height) < 32, "lt32", "ge32"),
        "roi_min_dimension": np.where(np.minimum(x2 - x1 + 1, y2 - y1 + 1) < 32, "lt32", "ge32"),
        "sign_family": np.where(np.isin(actual, tuple(_SPEED_CLASSES)), "speed_limit", "other"),
    }
    for column in optional:
        dimensions[column] = [_category(value, column) for value in joined[column]]
    rows = []
    for dimension, groups in dimensions.items():
        grouped = pd.DataFrame({"group": groups, "error": actual != predicted}).groupby(
            "group", sort=True)["error"].agg(["size", "sum"])
        for group, counts in grouped.iterrows():
            n, errors = int(counts["size"]), int(counts["sum"])
            rows.append({"dimension": dimension, "group": group, "n": n, "errors": errors,
                         "error_rate": errors / n})
    return pd.DataFrame(rows, columns=("dimension", "group", "n", "errors", "error_rate")).sort_values(
        ["dimension", "group"], kind="stable").reset_index(drop=True)


def select_error_examples(frame, *, max_pairs=4, per_pair=3, seed=42):
    """Choose top error pairs, then deterministic SHA-256-ranked rows per pair.

    Hash input is seed plus exact Path; CSVRow breaks ties. This is a reproducible
    illustration, not representative random sampling or evidence of error cause.
    Ranks are one-based; output retains original columns plus the two ranks.
    """
    max_pairs = _integer_option(max_pairs, "max_pairs", minimum=1)
    per_pair = _integer_option(per_pair, "per_pair", minimum=1)
    seed = _integer_option(seed, "seed")
    summary = summarize_predictions(frame)
    examples = []
    for pair_rank, pair in enumerate(summary["errors"].head(max_pairs).itertuples(index=False), 1):
        mask = ((frame["ClassId"] == pair.true_class)
                & (frame[_PREDICTION] == pair.predicted_class))
        candidates = frame.loc[mask]
        positions = sorted(range(len(candidates)), key=lambda index: (
            sha256(f"{seed}|{candidates.iloc[index]['Path']}".encode("utf-8")).hexdigest(),
            int(candidates.iloc[index]["CSVRow"])))[:per_pair]
        selected = candidates.iloc[positions].copy(deep=True)
        selected["pair_rank"] = pair_rank
        selected["example_rank"] = np.arange(1, len(selected) + 1, dtype=np.int64)
        examples.append(selected)
    if not examples:
        empty = frame.iloc[:0].copy(deep=True)
        empty["pair_rank"] = pd.Series(index=empty.index, dtype=np.int64)
        empty["example_rank"] = pd.Series(index=empty.index, dtype=np.int64)
        return empty.reset_index(drop=True)
    return pd.concat(examples, ignore_index=True)
