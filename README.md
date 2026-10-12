# Air raid alert forecast

Hourly probability of an air raid alert for 24 regions of Ukraine, 1 to 24 hours ahead.
Stage 2 project for the KSE Agentic AI School.

## The task

At the end of every hour, for every region, the model answers one question:
what is the probability that an alert is active during the hour that is 1, 3, 6, 12 or 24 hours ahead?

The forecast time is the end of the last finished hour. Everything the model sees was published before that moment.

## Data

| Source | What it gives | Period used |
|---|---|---|
| [ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), `volunteer_data_en.csv` | start and end of every alert per region | 2023-01-01 to 2026-10-11 |
| Air Force channel [t.me/s/kpszsu](https://t.me/s/kpszsu) | text posts about drones, missiles and aviation | 2026-04-01 to 2026-10-12 |

The exact alert file version is recorded in `data/raw/source.json`. Raw files are downloaded, not stored in the repo.

## Parts

| File | What it does |
|---|---|
| `alerts/data.py` | Downloads the alert file, cleans it, and builds the hourly panel: minutes under alert per region per hour, and whether an alert was still on at the end of the hour. |
| `alerts/telegram.py` | Downloads channel posts from the public web preview, page by page, with checkpoints. |
| `alerts/text.py` | Turns posts into hourly counts: posts that mention a region, posts that say something is heading to a region, and a few topics (ballistic threat, all clear, aviation activity, guided bombs, fast target). |
| `alerts/features.py` | Builds the model inputs and the target for one horizon, and splits rows by time. |
| `alerts/models.py` | Three baselines (climatology, persistence, logistic regression on 4 inputs) and gradient boosting with and without text features. |
| `alerts/evaluate.py` | Metrics and a bootstrap interval that resamples whole days. |
| `alerts/run.py` | Fits every method and scores it on the validation or the test window. |
| `alerts/experiments.py` | Ablation on the validation window: other regions on or off, length of history. |
| `alerts/report.py` | Charts and result tables from a finished run. |
| `tests/` | 51 tests. The important ones check that no feature can see the future. |

## How leakage is prevented

1. **Features use finished hours only.** A forecast made at 14:00 uses alerts and posts up to 13:59:59.
2. **A test proves it.** It replaces every value after a cut point with its opposite and checks that no feature before the cut changes. Nine deliberate leaks were planted to confirm the test fails on each.
3. **The split respects label time.** A training row is used only if its answer was already known before the first forecast of the evaluation window.
4. **Choices are made on a validation window** (2026-08-16 to 2026-09-13). The test window (2026-09-13 to 2026-10-11) is run once at the end.
5. **Text rules.** Edited posts keep their timestamp but their text is ignored, because the text may have changed after publication. Town names were taken only from posts before the validation window.

## Results

Not final yet. Validation window, alert history only, Brier skill against 28-day climatology (0 means no better than climatology):

| Method | 1 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| persistence | 0.259 | -0.013 | -0.014 | -0.016 | -0.017 |
| logistic, 4 inputs | 0.278 | 0.015 | 0.013 | 0.011 | 0.009 |
| boosting, alert history | 0.296 | 0.035 | 0.026 | 0.026 | 0.021 |

When no alert is on at forecast time, one-hour skill of boosting drops to 0.067. Most of the headline skill is ongoing alerts continuing.

## Problems found in the data

- **The official file is frozen.** It stopped on 2026-09-07 while the repo still gets daily commits. The volunteer file is current, so it is the main source.
- **Fake long alerts.** All 18 alerts longer than 12 hours since 2024 have almost no matching official alert. They are missed all-clear messages. They are cut to start plus 30 minutes, the rule the dataset itself uses for a missing end.
- **The end of the file is incomplete.** Alerts still running at snapshot time are not in the file, so the last 12 hours are dropped.
- **Labels are noisy.** The volunteer and official files agree on 82 to 89% of region-hours.
- **Little memory.** Hourly autocorrelation of alerts is 0.36 at one hour and below 0.1 after two.

## Limits

- Region level only. Since late 2025 real alerts are declared per district.
- Hourly steps. A 15-minute step would suit the short alerts better.
- Cutting a fake long alert uses its final length, which is not known live.
- The town list is fixed, and the channel keeps changing how it writes.
- Luhansk is excluded: its permanent alert is not in the data.

## Run

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m alerts.data
.venv\Scripts\python -m alerts.telegram
.venv\Scripts\python -m pytest
.venv\Scripts\python -m alerts.run val
.venv\Scripts\python -m alerts.run test
.venv\Scripts\python -m alerts.report test
```
