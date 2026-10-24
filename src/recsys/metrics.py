"""Ranking and behavior metrics independent of neural libraries."""

import numpy as np

from .data import FIRST_ITEM


def topk(scores, k):
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] < FIRST_ITEM + 1:
        raise ValueError("Expected [batch, vocab] scores")
    k = min(k, scores.shape[1] - FIRST_ITEM)
    # Stable token-ID tie breaking makes popularity baselines reproducible.
    return np.argsort(-scores[:, FIRST_ITEM:], axis=1, kind="stable")[:, :k] + FIRST_ITEM


def per_user_ranking(recommendations, targets, ks=(10, 20)):
    hits = recommendations == np.asarray(targets)[:, None]
    result = {}
    for k in ks:
        sub = hits[:, :k]
        result[f"recall@{k}"] = sub.any(axis=1).astype(float)
        result[f"ndcg@{k}"] = (sub / np.log2(np.arange(sub.shape[1]) + 2)).sum(axis=1)
    return result


def behavior(recommendations, contexts, popularity):
    if not len(recommendations):
        raise ValueError("Empty recommendations")
    flat = recommendations.ravel()
    counts = np.bincount(flat, minlength=len(popularity))[FIRST_ITEM:]
    n_items = len(counts)
    sorted_counts = np.sort(counts)
    gini = float((2 * np.arange(1, n_items + 1) - n_items - 1) @ sorted_counts / (n_items * sorted_counts.sum()))
    popular_ids = np.argsort(-popularity[FIRST_ITEM:], kind="stable")[:max(1, int(np.ceil(.01 * n_items)))] + FIRST_ITEM
    # Deterministic adjacent pairs avoid an O(users^2) comparison.
    overlap = [len(set(a) & set(b)) / recommendations.shape[1]
               for a, b in zip(recommendations[:-1], recommendations[1:])]
    return {
        "coverage": float((counts > 0).mean()), "exposure_gini": gini,
        "top_1pct_popular_share": float(np.isin(flat, popular_ids).mean()),
        "seen_fraction": float((recommendations[:, :, None] == contexts[:, None, :]).any(axis=2).mean()),
        "within_list_repeat_fraction": float(np.mean([1 - len(set(r)) / len(r) for r in recommendations])),
        "mean_train_popularity": float(popularity[flat].mean()),
        "adjacent_user_overlap": float(np.mean(overlap)) if overlap else None,
    }


def paired_bootstrap(before, after, seed=42, repeats=1000):
    delta = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    if delta.ndim != 1 or not len(delta):
        raise ValueError("Expected one paired value per user")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)])
    return {"mean_delta": float(delta.mean()), "ci95": np.quantile(means, [.025, .975]).tolist()}
