"""Namespaced static figures for Đặng Quang Tiến's saved-prediction evaluation.

All numeric figures have CSV equivalents. Blue/orange identity is validated by the
bundled dataviz validator; marker shape/line style also encode identity. Both light
and dark themes are selected explicitly, not obtained by inverting an image.
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from src.preprocessing import read_rgb, crop_roi

PALETTES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "grid": "#e1e0d9",
              "series": ("#2a78d6", "#eb6834"), "cmap": "Blues"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "grid": "#383835",
             "series": ("#3987e5", "#d95926"), "cmap": "Blues"},
}


def settings(theme):
    p = PALETTES[theme]
    return {"figure.facecolor": p["surface"], "axes.facecolor": p["surface"],
            "text.color": p["ink"], "axes.labelcolor": p["ink"],
            "xtick.color": p["ink"], "ytick.color": p["ink"],
            "axes.edgecolor": p["grid"], "grid.color": p["grid"],
            "font.family": "DejaVu Sans", "font.size": 10,
            "axes.spines.top": False, "axes.spines.right": False,
            "savefig.facecolor": p["surface"]}


def save(fig, directory, prefix, name, theme):
    path = directory / f"{prefix}_{name}{'_dark' if theme == 'dark' else ''}.png"
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=145)
    plt.close(fig)
    return path


def numeric_figures(summary, classes, groups, comparisons, confusions, output, prefix):
    paths = []
    for theme in ("light", "dark"):
        p = PALETTES[theme]
        with plt.rc_context(settings(theme)):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
            for ax, metric, title in zip(axes, ("accuracy", "f1_macro"), ("Accuracy", "Macro-F1")):
                for i, algorithm in enumerate(("knn", "dt")):
                    part = summary.loc[summary.algorithm.eq(algorithm)].set_index("method").loc[["pixel", "hog", "hog_pca"]]
                    x = part[metric].to_numpy()
                    y = np.arange(3) + (-.12 if i == 0 else .12)
                    ax.scatter(x, y, marker=("o" if i == 0 else "D"), s=70,
                               color=p["series"][i], edgecolor=p["surface"], linewidth=1.5,
                               label=algorithm.upper())
                    for value, position in zip(x, y):
                        ax.annotate(f"{value:.3f}", (value, position), xytext=(6, -4 if i else 3),
                                    textcoords="offset points", fontsize=9)
                ax.set(yticks=range(3), yticklabels=["Pixel", "HOG", "HOG + PCA"],
                       xlim=(0, 1.10), ylim=(2.5, -.5), xlabel=title)
                ax.grid(axis="x", linewidth=.6)
                ax.legend(frameon=False, loc="lower left")
            fig.suptitle("Official test — six historical frozen pipelines; no new inference")
            paths.append(save(fig, output, prefix, "official_comparison", theme))

            for name, cm in confusions.items():
                normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
                fig, axes = plt.subplots(1, 2, figsize=(17, 8.3), layout="constrained")
                for ax, values, title, maxval in zip(axes, (cm, normalized),
                        ("Counts", "Row-normalized proportion"), (750, 1)):
                    im = ax.imshow(values, cmap=p["cmap"], vmin=0, vmax=maxval, interpolation="nearest")
                    ax.set(xticks=range(43), yticks=range(43), xlabel="Predicted ClassId",
                           ylabel="True ClassId", title=title)
                    ax.tick_params(labelsize=7)
                    ax.tick_params(axis="x", rotation=90)
                    fig.colorbar(im, ax=ax, fraction=.043, pad=.02)
                fig.suptitle(f"Official test — {name.upper()} | all 43 classes; counts/labels in matching CSV")
                paths.append(save(fig, output, prefix, f"cm_{name}", theme))

            fig, ax = plt.subplots(figsize=(11.5, 16), layout="constrained")
            for i, algorithm in enumerate(("knn", "dt")):
                part = classes.loc[classes.algorithm.eq(algorithm) & classes.method.eq("hog")].sort_values("class_id")
                ax.scatter(part.f1, part.class_id + (-.13 if i == 0 else .13), s=54,
                           color=p["series"][i], marker=("o" if i == 0 else "D"),
                           edgecolor=p["surface"], linewidth=1.3, label=f"{algorithm.upper()} HOG")
                weak = part.nsmallest(3, "f1")
                for row in weak.itertuples():
                    ax.annotate(f"{algorithm.upper()} {row.f1:.2f}", (row.f1, row.class_id),
                                xytext=(8, -10 if i else 8), textcoords="offset points", fontsize=8)
            names = classes.loc[classes.algorithm.eq("knn") & classes.method.eq("hog")].sort_values("class_id")
            labels = [f"{r.class_id}: {r.class_name} (n={r.support})" for r in names.itertuples()]
            ax.set(yticks=range(43), yticklabels=labels, ylim=(42.7, -.7), xlim=(0, 1.13),
                   xlabel="Per-class F1", title="Official test per-class F1 — CV winners, not chosen by test")
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="x", linewidth=.6)
            ax.legend(frameon=False, loc="lower right")
            paths.append(save(fig, output, prefix, "per_class_hog", theme))

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
            for ax, dimension, label in zip(axes, ("raw_min_dimension", "sign_family"),
                                            ("Source image size", "Sign family")):
                selected = groups.loc[groups.method.eq("hog") & groups.dimension.eq(dimension)]
                order = ["lt32", "ge32"] if dimension == "raw_min_dimension" else ["speed_limit", "other"]
                for i, algorithm in enumerate(("knn", "dt")):
                    part = selected.loc[selected.algorithm.eq(algorithm)].set_index("group")
                    vals = [part.loc[g, "error_rate"] for g in order]
                    ax.plot(range(2), vals, marker=("o" if i == 0 else "D"), markersize=8,
                            markeredgecolor=p["surface"], markeredgewidth=1.5,
                            color=p["series"][i], linewidth=2, linestyle=("-" if i == 0 else "--"),
                            label=algorithm.upper())
                    ax.annotate(algorithm.upper(), (1, vals[1]), xytext=(-55, 12 if i else -18),
                                textcoords="offset points", fontsize=9,
                                arrowprops={"arrowstyle": "-", "color": p["grid"], "lw": .8})
                ax.set(xticks=[0, 1], xticklabels=order, ylim=(0, 1), xlim=(-.2, 1.2),
                       ylabel="Error rate", title=label)
                ax.grid(axis="y", linewidth=.6)
                ax.legend(frameon=False)
            fig.suptitle("Historical official-test error diagnostics — descriptive, not causal")
            paths.append(save(fig, output, prefix, "official_error_groups", theme))
    return paths


def error_gallery(examples, metadata, dataset, class_names, output, prefix, algorithm):
    """Top-pair/hash-ranked examples, not a representative random sample or a cause label."""
    examples = examples.head(8)
    if examples.empty:
        return []
    lookup = metadata.set_index("Path")
    rows = int(np.ceil(len(examples) / 2))
    fig, axes = plt.subplots(rows, 4, figsize=(13, rows * 3.5), layout="constrained", squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for position, row in enumerate(examples.itertuples(index=False)):
        meta = lookup.loc[row.Path]
        path = (dataset / row.Path).resolve()
        if not path.is_relative_to(dataset):
            raise ValueError("Gallery image path escapes dataset")
        image = read_rgb(path)
        roi = [int(meta[k]) for k in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")]
        crop = crop_roi(image, roi)
        ax, bx = axes[position // 2, (position % 2) * 2:(position % 2) * 2 + 2]
        ax.imshow(image)
        ax.add_patch(Rectangle((roi[0], roi[1]), roi[2]-roi[0]+1, roi[3]-roi[1]+1,
                               fill=False, edgecolor="#2a78d6", linewidth=1.5))
        ax.set_title(f"{row.Path}\nOriginal + CSV ROI", fontsize=8)
        bx.imshow(crop)
        title = (f"True {row.ClassId}: {class_names[int(row.ClassId)]}\n"
                 f"Pred {row.PredictedClassId}: {class_names[int(row.PredictedClassId)]}")
        bx.set_title("\n".join(textwrap.fill(line, 35) for line in title.splitlines()), fontsize=8)
    fig.suptitle(f"{algorithm.upper()} HOG errors — deterministic hash-ranked examples from top pairs\n"
                 "Original image and ROI; examples do not prove blur/lighting/crop caused the error", fontsize=12)
    return [save(fig, output, prefix, f"error_gallery_{algorithm}_hog", "light")]
