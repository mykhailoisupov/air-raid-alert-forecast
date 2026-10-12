from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from alerts.features import feature_names

RECENT_DAYS = 90
SIMPLE = ["on_now", "alert_now", "clim_28d", "share_24h"]


def recent(fit):
    return fit[fit.origin > fit.origin.max() - pd.Timedelta(days=RECENT_DAYS)]


def climatology(fit, target):
    return target.clim_28d.to_numpy()


def persistence(fit, target):
    fit = recent(fit)
    overall = fit.groupby("on_now").y.mean()
    by_region = fit.groupby([fit.region.astype(str), "on_now"]).y.mean()
    keys = pd.MultiIndex.from_arrays([target.region.astype(str), target.on_now])
    p = by_region.reindex(keys).to_numpy()
    return np.where(np.isnan(p), target.on_now.map(overall).to_numpy(), p)


def logistic(fit, target):
    model = LogisticRegression(max_iter=1000)
    model.fit(recent(fit)[SIMPLE], recent(fit).y)
    return model.predict_proba(target[SIMPLE])[:, 1]


def boosting(fit, target, text=False):
    features = [f for f in feature_names(fit) if text or not f.startswith("tg_")]
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=False,
        random_state=0,
    )
    model.fit(fit[features], fit.y)
    return model.predict_proba(target[features])[:, 1]


METHODS = {
    "climatology": climatology,
    "persistence": persistence,
    "logistic": logistic,
    "boosting": boosting,
    "boosting_text": partial(boosting, text=True),
}
