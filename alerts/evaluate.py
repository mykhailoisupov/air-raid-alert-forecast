import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 0.001


def within_region_auc(y, p, region):
    aucs = []
    for r in np.unique(region):
        m = region == r
        if 0 < y[m].mean() < 1:
            aucs.append(roc_auc_score(y[m], p[m]))
    return float(np.mean(aucs))


def scores(y, p, region):
    y, region = np.asarray(y), np.asarray(region.astype(str))
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return {
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "roc_auc_within_region": within_region_auc(y, p, region),
    }


def skill_interval(predictions, method, reference, draws=2000, seed=0):
    day = predictions.origin.dt.floor("D")
    errors = pd.DataFrame({
        "method": (predictions[method] - predictions.y) ** 2,
        "reference": (predictions[reference] - predictions.y) ** 2,
    }).groupby(day).sum().to_numpy()
    rng = np.random.default_rng(seed)
    sample = errors[rng.integers(0, len(errors), size=(draws, len(errors)))].sum(axis=1)
    low, high = np.percentile(1 - sample[:, 0] / sample[:, 1], [2.5, 97.5])
    return float(low), float(high)
