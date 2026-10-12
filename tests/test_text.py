import pandas as pd

from alerts.text import text_panel

REGIONS = ["Kharkivska oblast", "Kyiv City", "Kyivska oblast", "Sumska oblast"]


def posts(rows):
    df = pd.DataFrame(rows, columns=["posted_at", "edited", "text"])
    df["posted_at"] = pd.to_datetime(df.posted_at, utc=True)
    return df


def panel(rows):
    hours = pd.date_range("2026-06-01 00:00", periods=6, freq="h", tz="UTC")
    return text_panel(posts(rows), hours, REGIONS)


def test_posts_are_counted_in_the_hour_they_were_published():
    p = panel([
        ("2026-06-01 00:00:00", False, "БпЛА на Харків з півночі."),
        ("2026-06-01 01:59:59", False, "БпЛА на Харків з півночі."),
        ("2026-06-01 02:00:00", False, "БпЛА на Харків з півночі."),
    ])
    assert p["tg_posts"].tolist() == [1, 1, 1, 0, 0, 0]
    assert p["tg_heading"]["Kharkivska oblast"].tolist() == [1, 1, 1, 0, 0, 0]


def test_mention_and_heading_are_different():
    p = panel([
        ("2026-06-01 00:00:00", False, "start"),
        ("2026-06-01 03:10:00", False, "Чернігівщина: БпЛА в напрямку Київщини."),
        ("2026-06-01 03:20:00", False, "БпЛА на півночі Сумщини."),
        ("2026-06-01 03:30:00", False, "Загроза балістики для Києва!"),
    ])
    assert p["tg_mentions"].iloc[3].to_dict() == {
        "Kharkivska oblast": 0, "Kyiv City": 1, "Kyivska oblast": 1, "Sumska oblast": 1,
    }
    assert p["tg_heading"].iloc[3].to_dict() == {
        "Kharkivska oblast": 0, "Kyiv City": 0, "Kyivska oblast": 1, "Sumska oblast": 0,
    }


def test_edited_posts_count_but_their_text_is_ignored():
    p = panel([
        ("2026-06-01 00:00:00", False, "start"),
        ("2026-06-01 01:10:00", True, "БпЛА на Харків з півночі."),
    ])
    assert p["tg_posts"].tolist()[1] == 1
    assert p["tg_mentions"]["Kharkivska oblast"].tolist()[1] == 0


def test_hours_before_the_first_post_are_missing_not_zero():
    p = panel([("2026-06-01 02:30:00", False, "БпЛА на Харків з півночі.")])
    assert p["tg_posts"].isna().tolist() == [True, True, True, False, False, False]
    assert p["tg_mentions"]["Kharkivska oblast"].isna().tolist() == [True, True, True, False, False, False]
