import pandas as pd

from alerts.data import build_panel, load_intervals
from alerts.models import boosting, climatology
from alerts.run import REPORTS, evaluate

METHODS = {"climatology": climatology, "boosting": boosting}
CONFIGS = {
    "own region only, fit from 2025-01": {"spatial": False, "train_start": "2025-01-01"},
    "all regions, fit from 2023-02": {"spatial": True, "train_start": "2023-02-01"},
    "all regions, fit from 2025-01": {"spatial": True, "train_start": "2025-01-01"},
    "all regions, fit from 2026-01": {"spatial": True, "train_start": "2026-01-01"},
    "all regions, fit from 2026-05": {"spatial": True, "train_start": "2026-05-01"},
}


def main():
    panel = build_panel(load_intervals())
    tables = []
    for name, config in CONFIGS.items():
        table, _, _ = evaluate(panel, "val", methods=METHODS, **config)
        tables.append(table[table.method == "boosting"].assign(config=name))
        print(name, flush=True)
    out = pd.concat(tables)
    REPORTS.mkdir(exist_ok=True)
    out.to_csv(REPORTS / "experiments_val.csv", index=False)
    for metric in ["brier_skill", "roc_auc_within_region"]:
        print(metric)
        print(out.pivot_table(index="config", columns=["subset", "horizon"], values=metric, sort=False).round(4).to_string())


if __name__ == "__main__":
    main()
