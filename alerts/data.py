import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
FILE = "datasets/volunteer_data_en.csv"
RAW = Path("data/raw/volunteer_data_en.csv")
SOURCE = Path("data/raw/source.json")
DROP_REGIONS = ["Luhanska oblast"]
PANEL_START = "2023-01-01"
TRIM_HOURS = 12
MAX_HOURS = 12
NAIVE_MINUTES = 30


def download(ref="main"):
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/commits/{ref}") as r:
        commit = json.load(r)
    sha = commit["sha"]
    urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{REPO}/{sha}/{FILE}", RAW)
    source = {"repo": REPO, "file": FILE, "sha": sha, "committed_at": commit["commit"]["author"]["date"]}
    SOURCE.write_text(json.dumps(source, indent=2))
    return source


def load_intervals(path=RAW):
    df = pd.read_csv(path)
    for col in ["started_at", "finished_at"]:
        df[col] = pd.to_datetime(df[col], utc=True, format="mixed")
    df = df[~df.region.isin(DROP_REGIONS)]
    df = df[df.finished_at > df.started_at].copy()
    too_long = df.finished_at - df.started_at > pd.Timedelta(hours=MAX_HOURS)
    df.loc[too_long, "finished_at"] = df.started_at + pd.Timedelta(minutes=NAIVE_MINUTES)
    df.loc[too_long, "naive"] = True
    return df.sort_values("started_at").reset_index(drop=True)


def default_end(df):
    return df.finished_at.max().floor("h") - pd.Timedelta(hours=TRIM_HOURS)


def build_panel(df, start=PANEL_START, end=None):
    start = pd.Timestamp(start, tz="UTC")
    end = default_end(df) if end is None else pd.Timestamp(end)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    n = len(hours) * 60
    minute = pd.Timedelta(minutes=1)
    minutes, on_end = {}, {}
    for region in sorted(df.region.unique()):
        g = df[df.region == region]
        s = ((g.started_at.dt.floor("min") - start) // minute).to_numpy()
        e = ((g.finished_at.dt.ceil("min") - start) // minute).to_numpy()
        keep = (e > 0) & (s < n)
        delta = np.zeros(n + 1, dtype=np.int32)
        np.add.at(delta, np.clip(s[keep], 0, n), 1)
        np.add.at(delta, np.clip(e[keep], 0, n), -1)
        active = (np.cumsum(delta[:-1]) > 0).reshape(-1, 60)
        minutes[region] = active.sum(axis=1)
        on_end[region] = active[:, -1].astype(int)
    return {
        "minutes": pd.DataFrame(minutes, index=hours),
        "on_end": pd.DataFrame(on_end, index=hours),
    }


if __name__ == "__main__":
    print(json.dumps(download(*sys.argv[1:]), indent=2))
