import pandas as pd

from alerts.data import build_panel


def intervals(rows):
    df = pd.DataFrame(rows, columns=["region", "started_at", "finished_at"])
    for col in ["started_at", "finished_at"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def panel(rows, end="2026-01-01 08:00"):
    return build_panel(intervals(rows), start="2026-01-01", end=pd.Timestamp(end, tz="UTC"))


def test_alert_inside_one_hour():
    p = panel([("A", "2026-01-01 00:10", "2026-01-01 00:40")])
    assert p["minutes"]["A"].tolist() == [30, 0, 0, 0, 0, 0, 0, 0]
    assert p["on_end"]["A"].tolist() == [0, 0, 0, 0, 0, 0, 0, 0]


def test_alert_across_hours():
    p = panel([("A", "2026-01-01 01:50", "2026-01-01 03:05")])
    assert p["minutes"]["A"].tolist() == [0, 10, 60, 5, 0, 0, 0, 0]
    assert p["on_end"]["A"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]


def test_overlapping_alerts_are_merged():
    p = panel([
        ("A", "2026-01-01 05:00", "2026-01-01 05:30"),
        ("A", "2026-01-01 05:20", "2026-01-01 05:50"),
    ])
    assert p["minutes"]["A"].tolist() == [0, 0, 0, 0, 0, 50, 0, 0]


def test_alerts_are_clipped_to_the_panel():
    p = panel([
        ("A", "2025-12-31 23:00", "2026-01-01 00:15"),
        ("A", "2026-01-01 07:30", "2026-01-01 09:00"),
        ("A", "2026-01-01 10:00", "2026-01-01 11:00"),
    ])
    assert p["minutes"]["A"].tolist() == [15, 0, 0, 0, 0, 0, 0, 30]
    assert p["on_end"]["A"].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]


def test_regions_are_independent():
    p = panel([
        ("A", "2026-01-01 00:00", "2026-01-01 01:00"),
        ("B", "2026-01-01 02:00", "2026-01-01 02:30"),
    ])
    assert p["minutes"]["A"].tolist()[:3] == [60, 0, 0]
    assert p["minutes"]["B"].tolist()[:3] == [0, 0, 30]


def test_alerts_longer_than_the_cap_get_the_naive_end(tmp_path):
    from alerts.data import load_intervals

    path = tmp_path / "alerts.csv"
    path.write_text(
        "region,started_at,finished_at,naive\n"
        "A,2026-01-01 00:00:00+00:00,2026-01-01 11:00:00+00:00,False\n"
        "A,2026-01-02 00:00:00+00:00,2026-01-05 00:00:00+00:00,False\n"
        "Luhanska oblast,2026-01-02 00:00:00+00:00,2026-01-02 01:00:00+00:00,False\n"
        "A,2026-01-06 00:00:00+00:00,2026-01-06 00:00:00+00:00,False\n"
    )
    df = load_intervals(path)
    hours = ((df.finished_at - df.started_at).dt.total_seconds() / 3600).tolist()
    assert hours == [11.0, 0.5]
    assert df.naive.tolist() == [False, True]
