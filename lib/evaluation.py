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

LEVELS = ["Low", "Medium", "High"]  # ordered: the agreement is weighted by the distance between levels
N_BOOTSTRAP = 1000                   # resamples for the confidence interval of kappa
PREFIX = "generated_instructions_with_intention_explanation_"


def load_ratings(ratings_file):
    """Individual ratings of the participants: csv with one row per rating and
    columns id (transcript id, <condition>#<row>), dimension, rater, rating
    (Low / Medium / High)."""
    ratings = pd.read_csv(ratings_file)[["id", "dimension", "rater", "rating"]]
    unknown = set(ratings["rating"]) - set(LEVELS)
    assert not unknown, f"ratings must be in {LEVELS}, found {unknown}"
    return ratings


def consensus(levels):
    """Median of ordered levels (indices). With an even number of ratings, a
    median halfway between two levels (Low-Medium or Medium-High) gives Medium."""
    median = np.median(levels)
    return 1 if median % 1 else int(median)


def perceived_levels(ratings):
    """Perceived level of each transcript: median of its ratings."""
    index = {level: i for i, level in enumerate(LEVELS)}
    perceived = (ratings.assign(rating=ratings["rating"].map(index))
                 .groupby(["id", "dimension"])["rating"]
                 .agg(lambda r: LEVELS[consensus(r.to_numpy())]))
    return perceived.rename("perceived").reset_index()


def load_transcripts(data_dir, dimensions, reference="instructed", ratings_file=None,
                     n_per_level=None, seed=0):
    """Transcripts of all ``full_*`` conditions with their reference level.

    Each transcript gets an ``id`` = ``<condition>#<row>`` (``generated_id`` is
    not unique across files).

    Parameters
    ----------
    reference : "instructed" or "perceived"
        For "perceived", the level is the median of the ratings of
        ``ratings_file`` (see ``load_ratings``); only rated transcripts are kept.
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
        df = df.merge(perceived_levels(load_ratings(ratings_file)), on=["id", "dimension"])
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
    """Accuracy, macro-F1 and quadratic-weighted kappa of a DataFrame with
    columns true and pred."""
    index = {level: i for i, level in enumerate(LEVELS)}
    return {"accuracy": accuracy_score(pred["true"], pred["pred"]),
            "macro_f1": f1_score(pred["true"], pred["pred"], labels=LEVELS, average="macro"),
            "kappa": quadratic_kappa(pred["true"].map(index).to_numpy(), pred["pred"].map(index).to_numpy())}


def quadratic_kappa(true, pred):
    """Quadratic-weighted Cohen's kappa between two arrays of level indices."""
    k = len(LEVELS)
    observed = np.bincount(true * k + pred, minlength=k * k).reshape(k, k)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    weights = np.subtract.outer(np.arange(k), np.arange(k)) ** 2 / (k - 1) ** 2
    return 1 - (weights * observed).sum() / (weights * expected).sum()


def kappa_with_ci(pred, n_bootstrap=N_BOOTSTRAP, seed=0):
    """Quadratic-weighted Cohen's kappa against the reference levels, with a 95%
    percentile bootstrap confidence interval (transcripts resampled with replacement)."""
    index = {level: i for i, level in enumerate(LEVELS)}
    true = pred["true"].map(index).to_numpy()
    predicted = pred["pred"].map(index).to_numpy()
    rng = np.random.default_rng(seed)
    resamples = []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, len(true), len(true))
        resamples.append(quadratic_kappa(true[sample], predicted[sample]))
    low, high = np.nanpercentile(resamples, [2.5, 97.5])
    return {"kappa": quadratic_kappa(true, predicted), "kappa_ci_low": low, "kappa_ci_high": high}


def human_agreement(ratings, dimension, n_bootstrap=N_BOOTSTRAP, seed=0):
    """Agreement between human raters: each rating is compared with the median of
    the other raters of the same transcript (leave-one-rater-out), with the
    quadratic-weighted kappa. The 95% confidence interval resamples transcripts
    (all the ratings of a transcript stay together). Transcripts with a single
    rating are skipped."""
    index = {level: i for i, level in enumerate(LEVELS)}
    pairs = []  # per transcript: (ratings, medians of the other raters)
    for _, group in ratings[ratings["dimension"] == dimension].groupby("id"):
        values = group["rating"].map(index).to_numpy()
        if len(values) < 2:
            continue
        others = [consensus(np.delete(values, r)) for r in range(len(values))]
        pairs.append((values, np.array(others)))

    def kappa(selection):
        return quadratic_kappa(np.concatenate([pairs[i][0] for i in selection]),
                               np.concatenate([pairs[i][1] for i in selection]))

    rng = np.random.default_rng(seed)
    resamples = [kappa(rng.integers(0, len(pairs), len(pairs))) for _ in range(n_bootstrap)]
    low, high = np.nanpercentile(resamples, [2.5, 97.5])
    return {"kappa": kappa(range(len(pairs))), "kappa_ci_low": low, "kappa_ci_high": high,
            "n_transcripts": len(pairs), "n_ratings": int(sum(len(v) for v, _ in pairs))}


def agreement_table(ratings, predictions_by_method, dimensions):
    """AGREEMENT (quadratic-weighted Cohen's kappa, 95% CI) with the human raters.

    The first row compares each rater with the other raters; the others compare
    the predictions of each method with the perceived level (median of the raters).

    Parameters
    ----------
    predictions_by_method : dict {method: {(dimension, setting): predictions}},
        predictions made with REFERENCE = "perceived" (column true = perceived level).
    """
    rows = []
    for dimension in dimensions:
        rows.append({"comparison": "Human raters vs human", "dimension": dimension,
                     **human_agreement(ratings, dimension)})
        for method, predictions in predictions_by_method.items():
            for (dim, setting), pred in predictions.items():
                if dim == dimension:
                    rows.append({"comparison": f"{method} ({setting}) vs human", "dimension": dimension,
                                 **kappa_with_ci(pred), "n_transcripts": len(pred)})
    table = pd.DataFrame(rows)
    table["agreement"] = [f"{k:.2f} [{lo:.2f}, {hi:.2f}]" for k, lo, hi in
                          zip(table["kappa"], table["kappa_ci_low"], table["kappa_ci_high"])]
    return table.pivot(index="comparison", columns="dimension", values="agreement").reindex(
        list(dict.fromkeys(table["comparison"])))


def cluster_scores(pred):
    """Label-free agreement between clusters and reference levels."""
    return {"ARI": adjusted_rand_score(pred["true"], pred["cluster"]),
            "AMI": adjusted_mutual_info_score(pred["true"], pred["cluster"])}


def evaluate(predictions, method, reference):
    """Score table from {(dimension, setting): predictions}. Few-shot predictions
    are scored per fold and on all folds pooled. Pooled rows also have the 95%
    bootstrap confidence interval of the quadratic-weighted kappa."""
    records = []
    for (dimension, setting), pred in predictions.items():
        if "fold" in pred.columns:
            for fold, pred_fold in pred.groupby("fold"):
                records.append({"dimension": dimension, "setting": setting, "fold": fold,
                                **scores(pred_fold)})
        extra = cluster_scores(pred) if "cluster" in pred.columns else {}
        records.append({"dimension": dimension, "setting": setting, "fold": "all",
                        **scores(pred), **kappa_with_ci(pred), **extra})
    results = pd.DataFrame(records)
    results.insert(0, "method", method)
    results.insert(1, "reference", reference)
    return results


def summarize(results):
    """Few-shot: mean and standard deviation over folds."""
    per_fold = results[results["fold"] != "all"]
    return (per_fold.groupby(["dimension", "setting"])[["accuracy", "macro_f1", "kappa"]]
            .agg(["mean", "std"]).round(3))


def load_predictions(output_dir, prefix, reference, dimensions, settings=("few-shot", "zero-shot")):
    """Predictions saved by a method notebook, as {(dimension, setting): DataFrame};
    missing files are skipped."""
    predictions = {}
    for dimension in dimensions:
        for setting in settings:
            path = os.path.join(output_dir, f"{prefix}_predictions_{dimension}_{setting}_{reference}.csv")
            if os.path.exists(path):
                predictions[(dimension, setting)] = pd.read_csv(path)
    return predictions


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


def separation(D, labels, level_a, level_b, n_permutations=10000, seed=0):
    """Separation between two levels in a distance matrix.

    Delta = mean distance between transcripts of the two levels minus mean
    distance between transcripts of the same level (pairs within level_a or
    within level_b). The one-sided p-value is the share of label permutations
    (within the transcripts of the two levels) giving a Delta at least as large.

    Returns
    -------
    delta, p_value, n_a, n_b
    """
    labels = np.asarray(labels)
    keep = (labels == level_a) | (labels == level_b)
    D = np.asarray(D)[np.ix_(keep, keep)]
    z = (labels[keep] == level_a).astype(float)
    n, n_a = len(z), int(z.sum())
    n_b = n - n_a
    n_between = n_a * n_b
    n_within = n_a * (n_a - 1) / 2 + n_b * (n_b - 1) / 2
    total = D[np.triu_indices(n, 1)].sum()
    row_sums = D.sum(axis=1)

    def deltas(Z):
        # between-level sum of each row of Z: z'D(1 - z) = z'D1 - z'Dz
        between = Z @ row_sums - np.einsum("pi,ij,pj->p", Z, D, Z)
        return between / n_between - (total - between) / n_within

    observed = deltas(z[None, :])[0]
    rng = np.random.default_rng(seed)
    permuted = []
    for start in range(0, n_permutations, 1000):  # by blocks, to bound memory
        Z = np.array([rng.permutation(z) for _ in range(min(1000, n_permutations - start))])
        permuted.append(deltas(Z))
    permuted = np.concatenate(permuted)
    p_value = (1 + np.sum(permuted >= observed - 1e-12)) / (1 + n_permutations)
    return observed, p_value, n_a, n_b


def holm(p_values):
    """Holm-Bonferroni adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.maximum.accumulate((len(p_values) - np.arange(len(p_values))) * p_values[order])
    result = np.empty_like(p_values)
    result[order] = np.minimum(adjusted, 1)
    return result
