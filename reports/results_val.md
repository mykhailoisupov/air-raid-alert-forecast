**Chosen model** (Telegram posts at 1 h, alert history only after), Brier skill vs climatology, 95% interval

| forecasts | 1 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| All forecasts | 0.313 (0.299 to 0.327) | 0.035 (0.025 to 0.045) | 0.026 (0.019 to 0.033) | 0.026 (0.018 to 0.034) | 0.021 (0.013 to 0.028) |
| No alert at forecast time | 0.090 (0.076 to 0.106) | 0.032 (0.024 to 0.040) | 0.024 (0.016 to 0.031) | 0.025 (0.017 to 0.032) | 0.021 (0.013 to 0.028) |

**Chosen model**, AUC inside each region (0.5 = coin flip)

| forecasts | 1 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| All forecasts | 0.778 | 0.574 | 0.562 | 0.566 | 0.549 |
| No alert at forecast time | 0.650 | 0.575 | 0.563 | 0.564 | 0.552 |

**All forecasts**, Brier skill vs climatology

| method | 1 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| persistence | 0.259 | -0.013 | -0.014 | -0.016 | -0.017 |
| logistic, 4 inputs | 0.278 | 0.015 | 0.013 | 0.011 | 0.009 |
| boosting, alert history | 0.296 | 0.035 | 0.026 | 0.026 | 0.021 |
| boosting + Telegram posts | 0.313 | 0.037 | 0.022 | 0.014 | 0.023 |

**No alert at forecast time**, Brier skill vs climatology

| method | 1 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| persistence | 0.017 | -0.020 | -0.018 | -0.015 | -0.014 |
| logistic, 4 inputs | 0.043 | 0.013 | 0.009 | 0.009 | 0.008 |
| boosting, alert history | 0.067 | 0.032 | 0.024 | 0.025 | 0.021 |
| boosting + Telegram posts | 0.090 | 0.035 | 0.020 | 0.014 | 0.022 |

**Gain from Telegram posts**, Brier skill of boosting + Telegram vs boosting, 95% interval

| forecasts | 1 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| All forecasts | 0.025 (0.015 to 0.033) | 0.002 (-0.001 to 0.005) | -0.004 (-0.011 to 0.002) | -0.012 (-0.017 to -0.007) | 0.003 (-0.002 to 0.008) |
| No alert at forecast time | 0.025 (0.016 to 0.034) | 0.003 (-0.002 to 0.007) | -0.003 (-0.010 to 0.004) | -0.011 (-0.016 to -0.006) | 0.002 (-0.003 to 0.006) |

**Gain from Telegram posts at 1 h by week of the window**

| week 1 | week 2 | week 3 | week 4 |
|---|---|---|---|
| 0.021 | 0.029 | 0.030 | 0.017 |

**Channel drift**

| period | posts per day | posts naming a known region |
|---|---|---|
| fit | 101 | 88% |
| validation | 180 | 90% |
| test | 255 | 79% |
