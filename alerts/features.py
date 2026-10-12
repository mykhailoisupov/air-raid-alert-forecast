import numpy as np
import pandas as pd

HORIZONS = [1, 3, 6, 12, 24]
WINDOWS = [3, 6, 12, 24, 72, 168]
CAP = 168
LOCAL_TZ = "Europe/Kyiv"
META = ["origin", "y"]


def hours_since(mask):
    idx = np.arange(len(mask))[:, None]
    last = np.maximum.accumulate(np.where(mask.to_numpy(), idx, -CAP), axis=0)
    return pd.DataFrame(np.minimum(idx - last, CAP), index=mask.index, columns=mask.columns)


def build(panel, h, spatial=True, labeled_only=True):
    assert 1 <= h <= 24
    minutes, on_end = panel["minutes"], panel["on_end"]
    alert = (minutes > 0).astype(float)
    started = ((alert == 1) & (alert.shift(1) == 0)).astype(float)
    wide = {
        "on_now": on_end,
        "alert_now": alert,
        "minutes_now": minutes / 60,
        "alert_lag1": alert.shift(1),
        "alert_lag2": alert.shift(2),
        "since_alert": hours_since(alert == 1),
        "since_quiet": hours_since(alert == 0),
        "starts_24h": started.rolling(24).sum(),
        "same_hour_1d": alert.shift(24 - h),
        "same_hour_7d": alert.shift(168 - h),
        "clim_7d": sum(alert.shift(24 * k - h) for k in range(1, 8)) / 7,
        "clim_28d": sum(alert.shift(24 * k - h) for k in range(1, 29)) / 28,
    }
    for w in WINDOWS:
        wide[f"share_{w}h"] = alert.rolling(w).mean()
    national = on_end.mean(axis=1)
    by_time = {
        "national_now": national,
        "national_lag1": national.shift(1),
        "national_share_6h": alert.mean(axis=1).rolling(6).mean(),
    }
    if spatial:
        for region in on_end.columns:
            by_time[f"now_{region}"] = on_end[region]

    hours, regions = alert.index, list(alert.columns)
    origin = pd.Series(np.repeat(hours + pd.Timedelta(hours=1), len(regions)))
    target_hour = (origin + pd.Timedelta(hours=h - 1)).dt.tz_convert(LOCAL_TZ)
    cols = {
        "origin": origin,
        "region": pd.Categorical(np.tile(regions, len(hours)), categories=regions),
        "hour": target_hour.dt.hour,
        "weekday": target_hour.dt.weekday,
    }
    for name, frame in wide.items():
        cols[name] = frame.to_numpy(dtype=float).ravel()
    for name, series in by_time.items():
        cols[name] = np.repeat(series.to_numpy(dtype=float), len(regions))
    cols["y"] = alert.shift(-h).to_numpy().ravel()
    out = pd.DataFrame(cols)
    features = [c for c in out.columns if c not in META]
    out = out.dropna(subset=features + ["y"] if labeled_only else features)
    return out.reset_index(drop=True)


def feature_names(df):
    return [c for c in df.columns if c not in META]


def split(df, h, start, stop):
    known = df.origin + pd.Timedelta(hours=h)
    fit = df[known <= start]
    target = df[(df.origin >= start) & (known <= stop)]
    return fit, target
