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
        out = target[["origin", "region", "y"]].assign(horizon=h)
        for name, method in methods.items():
            out[name] = method(fit, target)
            rows.append({"horizon": h, "method": name, **scores(target.y, out[name], target.region)})
        predictions.append(out)
    table = pd.DataFrame(rows)
    reference = table[table.method == "climatology"].set_index("horizon").brier
    table["brier_skill"] = 1 - table.brier / table.horizon.map(reference)
    info = {
        "evaluated_on": on,
        "fit_from": train_start,
        "forecasts_from": str(start),
        "forecasts_until": str(stop),
        "rows_per_horizon": int(len(predictions[0])),
        "alert_rate": float(predictions[0].y.mean()),
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
