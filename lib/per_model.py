# -*- coding: utf-8 -*-
"""
Per-generator-model evaluation without folds, shared by the FGW, embedding and
LLM-as-a-judge notebooks (human_*_per_model.ipynb).

For each model that generated the videos, each label (human or reference) and
each number of examples M, a draw takes M random videos per level among the
videos of the model; the videos of the model that are not in the draw are
classified with them. All the methods use the same draws (``draw_example_sets``),
and the same scores, plots and tables (below), so that they can be compared.
"""

import hashlib
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from evaluation import LEVELS, load_labeled_transcripts

INDEX = {level: k for k, level in enumerate(LEVELS)}

try:  # tables are displayed in notebooks, printed elsewhere
    from IPython.display import display
except ImportError:
    def display(table):
        print(table.to_string() if hasattr(table, "to_string") else table)
WEIGHTS = np.subtract.outer(np.arange(3), np.arange(3)) ** 2 / 4
DISTANCES = [f"distance_{level}" for level in LEVELS]


#%% Data and draws

def load_data(labels_file, text_column, label_column, reference_column=None, generator_column=None):
    """Labeled videos with their generator model.

    Returns
    -------
    df : DataFrame (id, text, level[, reference], generator)
    labels : dict {label: column of df}, "human" and, with a reference column, "reference"
    generators : sorted list of the generator models
    counts : videos per model, label and level
    max_m : dict {(generator, label): largest usable M} (one video per level must stay outside a draw)
    """
    raw = pd.read_csv(labels_file)
    if reference_column is None:
        reference_column = next((c for c in raw.columns if c not in (text_column, label_column)
                                 and re.search("ref|instruct", c, re.IGNORECASE)), None)
    if generator_column is None:
        generator_column = next((c for c in raw.columns if re.search("model|generator|llm", c, re.IGNORECASE)), None)
    print("reference column:", reference_column, "| generator column:", generator_column)
    if generator_column is None:
        raise ValueError("no generator column: set GENERATOR_COLUMN")

    df = load_labeled_transcripts(labels_file, text_column, label_column, reference_column)
    df["generator"] = df["id"].map(dict(zip([f"row{k}" for k in raw.index], raw[generator_column].astype(str))))
    labels = {"human": "level", **({"reference": "reference"} if reference_column else {})}
    generators = sorted(df["generator"].unique())
    counts = pd.concat({label: df.groupby("generator")[column].value_counts().unstack()
                        .reindex(columns=LEVELS).fillna(0) for label, column in labels.items()}, axis=1).astype(int)
    max_m = {(g, label): int(counts.loc[g, label].min()) - 1 for g in generators for label in labels}
    return df, labels, generators, counts, max_m


def draw_example_sets(df, labels, generators, generator, label, m, n_draws, seed):
    """The example sets of the draws of one model, label and M: a list of {level: ids}.
    The draws are generated in sequence, so the first k of n draws are the same for any n."""
    column = labels[label]
    part = df[df["generator"] == generator]
    rng = np.random.default_rng([seed, m, generators.index(generator), list(labels).index(label)])
    return [{level: rng.choice(part.loc[part[column] == level, "id"].to_numpy(), m, replace=False)
             for level in LEVELS} for _ in range(n_draws)]


def m_values_to_run(m_values, m_main):
    """M values to compute: the sensitivity grid, plus M_MAIN when it is a number."""
    return sorted(set(m_values) | ({m_main} if isinstance(m_main, int) else set()))


def data_hash(part, column):
    return hashlib.sha1(pd.util.hash_pandas_object(part[["id", "text", column]], index=False).values).hexdigest()[:10]


def run_draws(df, labels, generators, max_m, m_values, n_draws, seed, classify_draw, cache_dir, cache_tag, dimension):
    """Runs every draw of every model, label and M with ``classify_draw``.

    classify_draw(generator, label, m, draw, chosen, targets) returns a DataFrame with the
    columns id and distance_<level> for the targets (the videos of the model not in the draw).
    Results are cached per model, label and M under a name that includes the data used and
    ``cache_tag`` (the parameters of the method).

    Returns
    -------
    DataFrame (draw, id, distance_<level>, generator, examples, M, pred, label columns)
    """
    import time
    os.makedirs(cache_dir, exist_ok=True)
    results = []
    for generator in generators:
        part = df[df["generator"] == generator]
        for label in labels:
            for m in m_values:
                if m > max_m[(generator, label)]:
                    continue
                path = os.path.join(cache_dir, f"{dimension}_{generator}_{label}_m{m}_draws{n_draws}_seed{seed}"
                                               f"_{cache_tag}_{data_hash(part, labels[label])}.csv")
                start = time.time()
                if os.path.exists(path):
                    result = pd.read_csv(path)
                else:
                    rows = []
                    for draw, chosen in enumerate(draw_example_sets(df, labels, generators, generator, label, m,
                                                                    n_draws, seed)):
                        used = set(np.concatenate(list(chosen.values())))
                        targets = [i for i in part["id"] if i not in used]
                        rows.append(classify_draw(generator, label, m, draw, chosen, targets).assign(draw=draw))
                    result = pd.concat(rows, ignore_index=True)[["draw", "id", *DISTANCES]]
                    result.to_csv(path, index=False)
                results.append(result.assign(generator=generator, examples=label, M=m))
                print(f"{generator}, {label} examples, M = {m}: {time.time() - start:.0f} s")
    draws = pd.concat(results, ignore_index=True)
    draws["pred"] = np.array(LEVELS)[draws[DISTANCES].to_numpy().argmin(axis=1)]
    return draws.merge(df[["id", *labels.values()]], on="id")


#%% Scores

def confusion(true, pred):
    return np.bincount(np.asarray(true) * 3 + np.asarray(pred), minlength=9).reshape(3, 3)


def accuracy_of(cm):
    return np.trace(cm) / cm.sum() if cm.sum() else np.nan


def kappa_of(cm):
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / cm.sum()
    denominator = (WEIGHTS * expected).sum()
    return 1 - (WEIGHTS * cm).sum() / denominator if denominator else np.nan


def summarize_draws(part, target, rng, n_bootstrap):
    """Scores pooled over all (draw, video) pairs, their 95% CI over draws (draws resampled;
    videos resampled when there is a single draw, e.g. zero-shot) and the 5-95% range of the
    accuracy of a single draw."""
    groups = [d for _, d in part.groupby("draw")]
    if len(groups) == 1:  # no draws to resample: resample the videos
        groups = [d.to_frame().T for _, d in part.iterrows()]
    cms = np.array([confusion(d[target].map(INDEX), d["pred"].map(INDEX)) for d in groups])
    single = [accuracy_of(confusion(d[target].map(INDEX), d["pred"].map(INDEX))) for _, d in part.groupby("draw")]
    boot = [cms[rng.integers(0, len(cms), len(cms))].sum(axis=0) for _ in range(n_bootstrap)]
    total = cms.sum(axis=0)
    return {"n draws": part["draw"].nunique(), "pairs": int(total.sum()),
            "accuracy": accuracy_of(total),
            "accuracy low": np.percentile([accuracy_of(b) for b in boot], 2.5),
            "accuracy high": np.percentile([accuracy_of(b) for b in boot], 97.5),
            "single draw p5": np.nanpercentile(single, 5), "single draw p95": np.nanpercentile(single, 95),
            "kappa": kappa_of(total),
            "kappa low": np.nanpercentile([kappa_of(b) for b in boot], 2.5),
            "kappa high": np.nanpercentile([kappa_of(b) for b in boot], 97.5)}


def score_draws(draws, labels, generators, n_bootstrap, seed):
    """Scores per model (and "all" models pooled, when every model has the M), examples
    label, M and label evaluated against. Answers that are not a level are left out."""
    readable = draws[draws["pred"].isin(LEVELS)]
    if len(readable) < len(draws):
        print(f"{len(draws) - len(readable)} answers that are not a level are left out of the scores")
    rng = np.random.default_rng(seed)
    rows = []
    for (label, m), part in readable.groupby(["examples", "M"]):
        full = part.groupby("generator")["draw"].nunique()
        for generator in [*generators, "all"]:
            if generator == "all":
                if len(full) < len(generators):
                    continue
                subset = part
            else:
                subset = part[part["generator"] == generator]
                if subset.empty:
                    continue
            for target, column in labels.items():
                rows.append({"generator": generator, "examples": label, "evaluated against": target, "M": m,
                             **summarize_draws(subset, column, rng, n_bootstrap)})
    return pd.DataFrame(rows)


#%% Sensitivity to M

def plot_sensitivity(scores, labels, generators, metric, ylabel, chance, path=None, title=None):
    """Score as a function of M, one line per model (black: all models), band = 95% CI.
    Rows: label of the examples; columns: label evaluated against. M = 0 is zero-shot."""
    sources, targets = list(labels), list(labels)
    fig, axes = plt.subplots(len(sources), len(targets), figsize=(5.5 * len(targets), 3.8 * len(sources)),
                             squeeze=False, sharex=True, sharey=True)
    for r, source in enumerate(sources):
        for c, target in enumerate(targets):
            ax = axes[r, c]
            for generator in [*generators, "all"]:
                s = scores[(scores["examples"] == source) & (scores["evaluated against"] == target)
                           & (scores["generator"] == generator)].sort_values("M")
                if s.empty:
                    continue
                style = dict(color="black", lw=2) if generator == "all" else {}
                line, = ax.plot(s["M"], s[metric], marker="o", ms=3, label=generator, **style)
                ax.fill_between(s["M"], s[f"{metric} low"], s[f"{metric} high"], color=line.get_color(), alpha=0.15)
            ax.axhline(chance, ls="--", c="grey", lw=1)
            ax.set_title(f"examples: {source}, evaluated against {target}", fontsize=10)
            ax.set_xlabel("M (examples per level" + (", 0 = zero-shot)" if (scores["M"] == 0).any() else ")"))
            ax.set_ylabel(ylabel)
    axes[0, -1].legend(title="generator", fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300)
    plt.show()


def enough_table(scores, tolerance):
    """Smallest M whose score is within ``tolerance`` of the best score over M."""
    rows = []
    for keys, s in scores.groupby(["generator", "examples", "evaluated against"]):
        s = s.sort_values("M")
        best = s["accuracy"].max()
        enough = s[s["accuracy"] >= best - tolerance].iloc[0]
        best_kappa = s["kappa"].max()
        enough_kappa = s[s["kappa"] >= best_kappa - tolerance].iloc[0]
        rows.append({**dict(zip(["generator", "examples", "evaluated against"], keys)),
                     "best accuracy": best, "M enough (accuracy)": int(enough["M"]),
                     "single draw range at that M": f"{enough['single draw p5']:.2f}-{enough['single draw p95']:.2f}",
                     "best kappa": best_kappa, "M enough (kappa)": int(enough_kappa["M"]),
                     "largest M tested": int(s["M"].max()),
                     "still rising": bool(s["accuracy"].idxmax() == s.index[-1] and len(s) > 1)})
    return pd.DataFrame(rows)


#%% Main results

def main_m_values(m_main, draws, enough, generators, labels):
    """M of the main results for each (model, examples label).

    m_main: a number (the largest M computed that does not exceed it), or "auto": the
    largest "M enough (accuracy)" over the labels evaluated against (at least 1).
    """
    keys = {}
    for generator in generators:
        for label in labels:
            available = sorted(m for m in draws.loc[(draws["generator"] == generator)
                                                    & (draws["examples"] == label), "M"].unique() if m > 0)
            if not available:
                continue
            if m_main == "auto":
                wanted = enough.loc[(enough["generator"] == generator) & (enough["examples"] == label),
                                    "M enough (accuracy)"].max()
                wanted = max(int(wanted), 1)
            else:
                wanted = int(m_main)
            keys[(generator, label)] = max([m for m in available if m <= wanted], default=available[0])
    return keys


def main_results(draws, scores, main_keys):
    """Rows of the draws and scores at the main M of each (model, examples label)."""
    main = pd.concat([draws[(draws["generator"] == g) & (draws["examples"] == label) & (draws["M"] == m)]
                      for (g, label), m in main_keys.items()], ignore_index=True)
    main_scores = scores.merge(pd.DataFrame([{"generator": g, "examples": label, "M": m}
                                             for (g, label), m in main_keys.items()]),
                               on=["generator", "examples", "M"])
    main_scores["accuracy [95% CI]"] = [f"{a:.2f} [{lo:.2f}, {hi:.2f}]" for a, lo, hi in
                                        zip(main_scores["accuracy"], main_scores["accuracy low"], main_scores["accuracy high"])]
    main_scores["kappa [95% CI]"] = [f"{k:.2f} [{lo:.2f}, {hi:.2f}]" for k, lo, hi in
                                     zip(main_scores["kappa"], main_scores["kappa low"], main_scores["kappa high"])]
    return main, main_scores


def per_video_predictions(part):
    """One prediction per video over its draws: the level nearest on average when there are
    distances, else the most frequent level (ties: the first in Low, Medium, High)."""
    if set(DISTANCES) <= set(part.columns) and part[DISTANCES].notna().all().all():
        per_video = part.groupby(["generator", "id", "reference"])[DISTANCES].mean()
        per_video["pred"] = np.array(LEVELS)[per_video.to_numpy().argmin(axis=1)]
        return per_video.reset_index()[["generator", "id", "reference", "pred"]]
    votes = (part[part["pred"].isin(LEVELS)].groupby(["generator", "id", "reference"])["pred"]
             .value_counts().unstack().reindex(columns=LEVELS).fillna(0))
    votes["pred"] = np.array(LEVELS)[votes.to_numpy().argmax(axis=1)]
    return votes.reset_index()[["generator", "id", "reference", "pred"]]


def respect(df, labels, generators, main, main_scores, n_bootstrap, seed, method, dimension, path=None,
            extra_modes=None):
    """Does each LLM respect the requested level? For each model, the share of its videos whose
    level found by the method equals the reference level (examples labeled by each label, at the
    main M; 95% CI over draws), extra modes (e.g. zero-shot: {mode: DataFrame(generator, id,
    reference, pred)}, 95% CI over videos), and the share whose human level equals it. chi2:
    equal shares across models, with one prediction per video.

    Returns (respect table, chi2 tests), and draws the figure.
    """
    if "reference" not in labels:
        print("This section needs a reference level (REFERENCE_COLUMN).")
        return None, None
    rows, tests = [], []
    for source in labels:
        s = main_scores[(main_scores["examples"] == source) & (main_scores["evaluated against"] == "reference")]
        for _, r in s.iterrows():
            rows.append({"mode": f"examples: {source}", "generator": r["generator"], "accuracy": r["accuracy"],
                         "low": r["accuracy low"], "high": r["accuracy high"]})
        per_video = per_video_predictions(main[main["examples"] == source])
        table = pd.crosstab(per_video["generator"], per_video["pred"] == per_video["reference"])
        if table.shape[1] == 2 and len(table) > 1:
            tests.append({"mode": f"examples: {source}", "p (equal across models)": chi2_contingency(table)[1]})
    rng = np.random.default_rng(seed)

    def by_video(mode, frame):
        for generator in generators:
            correct = (frame.loc[frame["generator"] == generator, "pred"]
                       == frame.loc[frame["generator"] == generator, "reference"]).to_numpy(float)
            boot = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n_bootstrap)]
            rows.append({"mode": mode, "generator": generator, "accuracy": correct.mean(),
                         "low": np.percentile(boot, 2.5), "high": np.percentile(boot, 97.5)})
        table = pd.crosstab(frame["generator"], frame["pred"] == frame["reference"])
        if table.shape[1] == 2 and len(table) > 1:
            tests.append({"mode": mode, "p (equal across models)": chi2_contingency(table)[1]})

    by_video("humans", df.assign(pred=df["level"]))
    for mode, frame in (extra_modes or {}).items():
        by_video(mode, frame)
    table = pd.DataFrame(rows)
    display(table.assign(value=[f"{a:.2f} [{lo:.2f}, {hi:.2f}]" for a, lo, hi in
                                zip(table["accuracy"], table["low"], table["high"])])
            .pivot(index="mode", columns="generator", values="value"))
    tests = pd.DataFrame(tests)
    display(tests.round(4))

    mode_order = ["humans", *[f"examples: {source}" for source in labels], *(extra_modes or {})]
    width = 0.8 / len(generators)
    fig, ax = plt.subplots(figsize=(9.5, 4))
    for k, generator in enumerate(generators):
        part = table[table["generator"] == generator].set_index("mode").reindex(mode_order)
        x = np.arange(len(mode_order)) + (k - (len(generators) - 1) / 2) * width
        ax.bar(x, part["accuracy"], width, label=generator,
               yerr=[part["accuracy"] - part["low"], part["high"] - part["accuracy"]], capsize=3)
    ax.axvline(0.5, c="black", lw=0.8)
    ax.axhline(1 / len(LEVELS), ls="--", c="grey", lw=1, label="chance")
    ax.set_xticks(np.arange(len(mode_order)), mode_order)
    ax.set_ylabel("level found = requested level")
    ax.set_title(f"{method} per model: does each LLM respect the requested {dimension} level?")
    ax.legend(title="generator", fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300)
    plt.show()
    return table, tests


#%% Comparison of the methods

METHOD_FILES = {"graph + FGW": "human_fgw_per_model", "embeddings": "human_embedding_per_model",
                "LLM-as-a-judge": "human_llm_per_model"}


def compare_methods(output_dir, dimension, labels, metric="accuracy", path=None):
    """Sensitivity curves of the methods already run (all models pooled) and their main scores."""
    curves, mains = {}, []
    for method, prefix in METHOD_FILES.items():
        scores_path = os.path.join(output_dir, f"{prefix}_scores_{dimension}.csv")
        main_path = os.path.join(output_dir, f"{prefix}_main_{dimension}.csv")
        if os.path.exists(scores_path):
            curves[method] = pd.read_csv(scores_path)
        if os.path.exists(main_path):
            mains.append(pd.read_csv(main_path).assign(method=method))
    if len(curves) < 2:
        print("Run at least two of human_fgw_per_model, human_embeddings_per_model and "
              "human_llm_judge_per_model to compare them.")
        return
    sources, targets = list(labels), list(labels)
    fig, axes = plt.subplots(len(sources), len(targets), figsize=(5.5 * len(targets), 3.8 * len(sources)),
                             squeeze=False, sharex=True, sharey=True)
    for r, source in enumerate(sources):
        for c, target in enumerate(targets):
            ax = axes[r, c]
            for method, scores in curves.items():
                s = scores[(scores["examples"] == source) & (scores["evaluated against"] == target)
                           & (scores["generator"] == "all")].sort_values("M")
                if s.empty:
                    continue
                line, = ax.plot(s["M"], s[metric], marker="o", ms=3, label=method)
                ax.fill_between(s["M"], s[f"{metric} low"], s[f"{metric} high"], color=line.get_color(), alpha=0.15)
            ax.axhline(1 / len(LEVELS) if metric == "accuracy" else 0, ls="--", c="grey", lw=1)
            ax.set_title(f"examples: {source}, evaluated against {target}", fontsize=10)
            ax.set_xlabel("M (examples per level, 0 = zero-shot)")
            ax.set_ylabel(f"{metric} (all models)")
    axes[0, -1].legend(title="method", fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300)
    plt.show()
    if mains:
        mains = pd.concat(mains, ignore_index=True)
        display(mains.pivot_table(index=["examples", "evaluated against", "generator"], columns="method",
                                  values=["M", "accuracy [95% CI]", "kappa [95% CI]"], aggfunc="first"))
