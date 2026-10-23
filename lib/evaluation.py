# -*- coding: utf-8 -*-
"""
Evaluation protocol shared by all classification methods (FGW, embeddings, LLM).

Every method must use the same transcripts, folds and few-shot examples: they
are created once by ``load_or_create_split`` and saved, then reloaded.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, adjusted_mutual_info_score,
                             adjusted_rand_score, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold

LEVELS = ["Low", "Medium", "High"]
PREFIX = "generated_instructions_with_intention_explanation_"


def load_transcripts(data_dir, dimensions, reference="instructed", perceived_file=None,
                     n_per_level=None, seed=0):
    """Transcripts of all ``full_*`` conditions with their reference level.

    Each transcript gets an ``id`` = ``<condition>#<row>`` (``generated_id`` is
    not unique across files).

    Parameters
    ----------
    reference : "instructed" or "perceived"
        For "perceived", ``perceived_file`` is a csv with columns id, dimension,
        perceived (Low / Medium / High); only the transcripts it lists are kept.
    n_per_level : int, optional
        Keep at most this many transcripts per dimension and level.

    Returns
    -------
    DataFrame with columns id, condition, dimension, text, instructed, level
    (and perceived).
    """
    frames = []
    for path in sorted(Path(data_dir).glob(PREFIX + "full_*.csv")):
        condition = path.stem[len(PREFIX):]
        if condition.endswith("ratedbyFGW"):
            continue
        data = pd.read_csv(path)
        for dimension in dimensions:
            if dimension not in data.columns:
                continue
            frames.append(pd.DataFrame({
                "id": [f"{condition}#{row}" for row in data.index],
                "condition": condition,
                "dimension": dimension,
                "text": data["instruction_with_intention"],
                "instructed": data[dimension].astype(str).str.split(" ").str[0],
            }))

    df = pd.concat(frames, ignore_index=True).dropna(subset=["text"])
    df = df[df["instructed"].isin(LEVELS)]

    if reference == "perceived":
        perceived = pd.read_csv(perceived_file)
        df = df.merge(perceived[["id", "dimension", "perceived"]], on=["id", "dimension"])
    df["level"] = df[reference]
    df = df[df["level"].isin(LEVELS)]

    if n_per_level is not None:
        df = pd.concat([group.sample(min(len(group), n_per_level), random_state=seed)
                        for _, group in df.groupby(["dimension", "level"])])
    return df.reset_index(drop=True)


def load_or_create_split(data, dimension, reference, split_dir, m=10, n_folds=5, seed=0):
    """Stratified folds and few-shot examples of one dimension.

    For each fold, ``m`` examples per level are drawn from the other folds; the
    transcripts of the fold are then classified with them. The split is saved
    in ``split_dir`` the first time and reloaded afterwards.

    Returns
    -------
    folds : DataFrame (id, level, fold)
    examples : DataFrame (fold, level, id)
    """
    os.makedirs(split_dir, exist_ok=True)
    folds_path = os.path.join(split_dir, f"folds_{dimension}_{reference}.csv")
    examples_path = os.path.join(split_dir, f"examples_{dimension}_{reference}_m{m}.csv")
    if os.path.exists(folds_path) and os.path.exists(examples_path):
        folds, examples = pd.read_csv(folds_path), pd.read_csv(examples_path)
        assert set(folds["id"]) == set(data["id"]), f"{folds_path} was made on other data"
        return folds, examples

    folds = data[["id", "level"]].copy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(skf.split(folds, folds["level"])):
        folds.loc[folds.index[test], "fold"] = fold
    folds["fold"] = folds["fold"].astype(int)

    rng = np.random.default_rng(seed)
    rows = []
    for fold in range(n_folds):
        train = folds[folds["fold"] != fold]
        for level in LEVELS:
            chosen = rng.choice(train.loc[train["level"] == level, "id"].to_numpy(), m, replace=False)
            rows += [{"fold": fold, "level": level, "id": i} for i in chosen]
    examples = pd.DataFrame(rows)

    folds.to_csv(folds_path, index=False)
    examples.to_csv(examples_path, index=False)
    return folds, examples


def zero_shot_subset(data, n_per_level=None, seed=0):
    """Transcripts clustered in the zero-shot setting: at most ``n_per_level``
    per level, the same for every method."""
    if n_per_level is None:
        return data
    return pd.concat([group.sample(min(len(group), n_per_level), random_state=seed)
                      for _, group in data.groupby("level")]).sort_index()


def map_clusters(clusters, true):
    """Maps each cluster to a level with the Hungarian algorithm: the one-to-one
    assignment that maximizes the agreement with the reference levels."""
    clusters, true = np.asarray(clusters), np.asarray(true)
    contingency = np.array([[np.sum((clusters == c) & (true == level)) for level in LEVELS]
                            for c in range(len(LEVELS))])
    cluster_idx, level_idx = linear_sum_assignment(-contingency)
    mapping = {c: LEVELS[l] for c, l in zip(cluster_idx, level_idx)}
    return np.array([mapping[c] for c in clusters])


def scores(pred):
    """Accuracy and macro-F1 of a DataFrame with columns true and pred."""
    return {"accuracy": accuracy_score(pred["true"], pred["pred"]),
            "macro_f1": f1_score(pred["true"], pred["pred"], labels=LEVELS, average="macro")}


def cluster_scores(pred):
    """Label-free agreement between clusters and reference levels."""
    return {"ARI": adjusted_rand_score(pred["true"], pred["cluster"]),
            "AMI": adjusted_mutual_info_score(pred["true"], pred["cluster"])}


def evaluate(predictions, method, reference):
    """Score table from {(dimension, setting): predictions}. Few-shot predictions
    are scored per fold and on all folds pooled."""
    records = []
    for (dimension, setting), pred in predictions.items():
        if "fold" in pred.columns:
            for fold, pred_fold in pred.groupby("fold"):
                records.append({"dimension": dimension, "setting": setting, "fold": fold,
                                **scores(pred_fold)})
        extra = cluster_scores(pred) if "cluster" in pred.columns else {}
        records.append({"dimension": dimension, "setting": setting, "fold": "all",
                        **scores(pred), **extra})
    results = pd.DataFrame(records)
    results.insert(0, "method", method)
    results.insert(1, "reference", reference)
    return results


def summarize(results):
    """Few-shot: mean and standard deviation over folds."""
    per_fold = results[results["fold"] != "all"]
    return (per_fold.groupby(["dimension", "setting"])[["accuracy", "macro_f1"]]
            .agg(["mean", "std"]).round(3))


def plot_confusion(predictions, reference, path=None):
    """Confusion matrices (rows normalized), one per dimension and setting."""
    dimensions = list(dict.fromkeys(d for d, _ in predictions))
    settings = list(dict.fromkeys(s for _, s in predictions))
    fig, axes = plt.subplots(len(dimensions), len(settings),
                             figsize=(5 * len(settings), 4.5 * len(dimensions)), squeeze=False)
    for row, dimension in enumerate(dimensions):
        for col, setting in enumerate(settings):
            pred = predictions[(dimension, setting)]
            cm = confusion_matrix(pred["true"], pred["pred"], labels=LEVELS, normalize="true")
            ConfusionMatrixDisplay(cm, display_labels=LEVELS).plot(
                ax=axes[row, col], cmap="Blues", values_format=".2f", colorbar=False)
            axes[row, col].set_title(f"{dimension}, {setting} (accuracy = {scores(pred)['accuracy']:.2f})")
            axes[row, col].set_xlabel("predicted level")
            axes[row, col].set_ylabel(f"{reference} level")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300)
    return fig
