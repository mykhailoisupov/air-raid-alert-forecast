import numpy as np
import pandas as pd
import pytest

from alerts.features import HORIZONS, build, feature_names, hours_since, split

HOURS = 24 * 60
CUT = 24 * 45


def random_panel(seed):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-01-01", periods=HOURS, freq="h", tz="UTC")
    minutes = rng.integers(0, 61, size=(HOURS, 4)) * (rng.random((HOURS, 4)) < 0.3)
    on_end = (minutes > 0) & (rng.random((HOURS, 4)) < 0.5)
    regions = ["A", "B", "C", "D"]
    return {
        "minutes": pd.DataFrame(minutes, index=hours, columns=regions),
        "on_end": pd.DataFrame(on_end.astype(int), index=hours, columns=regions),
    }


def with_different_future(panel, other):
    return {k: pd.concat([panel[k].iloc[:CUT], other[k].iloc[CUT:]]) for k in panel}


@pytest.mark.parametrize("h", HORIZONS)
def test_features_do_not_depend_on_the_future(h):
    panel = random_panel(0)
    changed = with_different_future(panel, random_panel(1))
    cut_time = panel["minutes"].index[CUT]
    a = build(panel, h, labeled_only=False)
    b = build(changed, h, labeled_only=False)
    a, b = a[a.origin <= cut_time], b[b.origin <= cut_time]
    assert len(a) > 1000
    cols = ["origin"] + feature_names(a)
    pd.testing.assert_frame_equal(a[cols], b[cols])


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
def test_split_keeps_training_labels_before_later_forecasts(h):
    panel = random_panel(0)
    df = build(panel, h)
    hours = panel["minutes"].index
    val_start, test_start = hours[24 * 40], hours[24 * 50]
    train, val, test = split(df, h, val_start, test_start)
    known = lambda part: part.origin + pd.Timedelta(hours=h)
    assert min(len(train), len(val), len(test)) > 0
    assert known(train).max() <= val_start <= val.origin.min()
    assert known(val).max() <= test_start <= test.origin.min()


def test_hours_since():
    mask = pd.DataFrame({"A": [False, True, False, False, True, False]})
    assert hours_since(mask)["A"].tolist() == [168, 0, 1, 2, 0, 1]
