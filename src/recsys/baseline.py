"""CPU-only popularity evaluation and reward-regression summaries."""

import numpy as np
import pandas as pd

from .metrics import behavior, per_user_ranking, topk


def regression_metrics(predicted, actual):
    valid = np.isfinite(predicted) & np.isfinite(actual)
    predicted, actual = np.asarray(predicted)[valid], np.asarray(actual)[valid]
    if not len(actual):
        return {"n": 0, "mae": None, "rmse": None, "spearman": None, "calibration": []}
    ranks_x = pd.Series(predicted).rank().to_numpy()
    ranks_y = pd.Series(actual).rank().to_numpy()
    correlation = float(np.corrcoef(ranks_x, ranks_y)[0, 1]) if ranks_x.std() > 0 and ranks_y.std() > 0 else None
    bins = np.minimum((predicted * 10).astype(int), 9).clip(0, 9)
    return {
        "n": len(actual), "mae": float(np.abs(predicted - actual).mean()),
        "rmse": float(np.sqrt(((predicted - actual) ** 2).mean())),
        "spearman": correlation,
        "calibration": [{"bin": i, "n": int((bins == i).sum()),
                         "predicted": float(predicted[bins == i].mean()),
                         "actual": float(actual[bins == i].mean())}
                        for i in range(10) if (bins == i).any()],
    }


def popularity_evaluation(data, popularity):
    one = topk(popularity[None, :], 20)
    recommendations = np.repeat(one, len(data["targets"]), axis=0)
    per_user = per_user_ranking(recommendations, data["targets"])
    return {
        "ranking": {name: float(values.mean()) for name, values in per_user.items()},
        "behavior@10": behavior(recommendations[:, :10], data["contexts"], popularity),
    }, per_user, recommendations
