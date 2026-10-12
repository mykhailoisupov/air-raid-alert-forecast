import json
import sys
from pathlib import Path

import pandas as pd

from alerts.data import SOURCE, build_panel, load_intervals
from alerts.evaluate import scores
from alerts.features import HORIZONS, build, split
from alerts.models import METHODS

TRAIN_START = "2025-01-01"
TEST_DAYS = 28
VAL_DAYS = 28
REPORTS = Path("reports")
SUBSETS = {
    "all": lambda df: df.y >= 0,
    "quiet_now": lambda df: df.on_now == 0,
}


def windows(panel, on):
    end = panel["minutes"].index[-1] + pd.Timedelta(hours=1)
    test_start = end - pd.Timedelta(days=TEST_DAYS)
    val_start = test_start - pd.Timedelta(days=VAL_DAYS)
    return (val_start, test_start) if on == "val" else (test_start, end)


def evaluate(panel, on="val", train_start=TRAIN_START, methods=METHODS, spatial=True):
    start, stop = windows(panel, on)
    rows, predictions = [], []
    for h in HORIZONS:
        data = build(panel, h, spatial=spatial)
        data = data[data.origin >= pd.Timestamp(train_start, tz="UTC")]
        fit, target = split(data, h, start, stop)
        out = target[["origin", "region", "on_now", "y"]].assign(horizon=h)
        for name, method in methods.items():
            out[name] = method(fit, target)
        predictions.append(out)
        for subset, mask in SUBSETS.items():
            part = out[mask(out)]
            for name in methods:
                rows.append({"horizon": h, "subset": subset, "method": name, **scores(part.y, part[name], part.region)})
    table = pd.DataFrame(rows)
    reference = table[table.method == "climatology"].set_index(["horizon", "subset"]).brier
    table["brier_skill"] = 1 - table.brier / pd.MultiIndex.from_frame(table[["horizon", "subset"]]).map(reference)
    info = {
        "evaluated_on": on,
        "fit_from": train_start,
        "forecasts_from": str(start),
        "forecasts_until": str(stop),
        "rows_per_horizon": int(len(predictions[0])),
        "alert_rate": float(predictions[0].y.mean()),
        "quiet_now_share": float((predictions[0].on_now == 0).mean()),
    }
    return table, pd.concat(predictions), info


def main(on="val"):
    panel = build_panel(load_intervals())
    table, predictions, info = evaluate(panel, on)
    info["source"] = json.loads(SOURCE.read_text())
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / f"metrics_{on}.csv", index=False)
    predictions.to_csv(REPORTS / f"predictions_{on}.csv.gz", index=False)
    (REPORTS / f"run_{on}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
