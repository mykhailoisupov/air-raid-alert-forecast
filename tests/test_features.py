import numpy as np
import pandas as pd
import pytest

from alerts.features import HORIZONS, build, feature_names, hours_since, split

HOURS = 24 * 60
CUT = 24 * 45
CUTS = [24 * 40, 24 * 45 + 7, 24 * 50 + 13]


def random_panel(seed):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-01-01", periods=HOURS, freq="h", tz="UTC")
    minutes = rng.integers(1, 61, size=(HOURS, 4)) * (rng.random((HOURS, 4)) < 0.3)
    on_end = (minutes > 0) & (rng.random((HOURS, 4)) < 0.5)
    regions = ["A", "B", "C", "D"]
    return {
        "minutes": pd.DataFrame(minutes, index=hours, columns=regions),
        "on_end": pd.DataFrame(on_end.astype(int), index=hours, columns=regions),
    }


def with_opposite_future(panel, cut):
    minutes, on_end = panel["minutes"].copy(), panel["on_end"].copy()
    minutes.iloc[cut:] = np.where(minutes.iloc[cut:] > 0, 0, 60)
    on_end.iloc[cut:] = 1 - on_end.iloc[cut:]
    return {"minutes": minutes, "on_end": on_end}


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("h", HORIZONS)
def test_features_do_not_depend_on_the_future(h, cut):
    panel = random_panel(0)
    changed = with_opposite_future(panel, cut)
    cut_time = panel["minutes"].index[cut]
    a = build(panel, h, labeled_only=False)
    b = build(changed, h, labeled_only=False)
    a, b = a[a.origin <= cut_time], b[b.origin <= cut_time]
    assert len(a) > 1000
    cols = ["origin"] + feature_names(a)
    pd.testing.assert_frame_equal(a[cols], b[cols])


@pytest.mark.parametrize("h", HORIZONS)
def test_every_feature_reacts_to_the_hour_before_the_forecast(h):
    panel = random_panel(0)
    changed = with_opposite_future(panel, CUT - 1)
    cut_time = panel["minutes"].index[CUT]
    a = build(panel, h, labeled_only=False)
    b = build(changed, h, labeled_only=False)
    a, b = a[a.origin == cut_time], b[b.origin == cut_time]
    for col in ["on_now", "alert_now", "minutes_now", "national_now", "now_A", "share_3h"]:
        assert not a[col].equals(b[col]), col


@pytest.mark.parametrize("h", HORIZONS)
def test_features_change_when_the_past_changes(h):
    panel = random_panel(0)
    other = random_panel(1)
    changed = {k: pd.concat([other[k].iloc[:CUT], panel[k].iloc[CUT:]]) for k in panel}
    cut_time = panel["minutes"].index[CUT]
    a = build(panel, h, labeled_only=False)
    b = build(changed, h, labeled_only=False)
    a, b = a[a.origin == cut_time], b[b.origin == cut_time]
    assert not a[feature_names(a)].equals(b[feature_names(b)])


@pytest.mark.parametrize("h", HORIZONS)
def test_target_is_the_alert_h_hours_after_the_last_observed_hour(h):
    panel = random_panel(0)
    alert = (panel["minutes"] > 0).astype(float)
    df = build(panel, h)
    target_hour = df.origin + pd.Timedelta(hours=h - 1)
    expected = [alert.at[t, r] for t, r in zip(target_hour, df.region)]
    assert df.y.tolist() == expected
    assert (target_hour >= df.origin).all()


@pytest.mark.parametrize("h", HORIZONS)
def test_split_keeps_fit_labels_before_the_first_forecast(h):
    panel = random_panel(0)
    df = build(panel, h)
    hours = panel["minutes"].index
    start, stop = hours[24 * 40], hours[24 * 50]
    fit, target = split(df, h, start, stop)
    assert min(len(fit), len(target)) > 0
    assert (fit.origin + pd.Timedelta(hours=h)).max() <= start <= target.origin.min()
    assert (target.origin + pd.Timedelta(hours=h)).max() <= stop


def test_hours_since():
    mask = pd.DataFrame({"A": [False, True, False, False, True, False]})
    assert hours_since(mask)["A"].tolist() == [168, 0, 1, 2, 0, 1]
