**Chosen model** (Telegram posts at 1 h, alert history only after), Brier skill vs climatology, 95% interval

| forecasts | 1 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| All forecasts | 0.267 (0.252 to 0.284) | 0.023 (0.016 to 0.029) | 0.017 (0.010 to 0.024) | 0.012 (0.006 to 0.019) | 0.007 (0.002 to 0.013) |
| No alert at forecast time | 0.065 (0.048 to 0.083) | 0.019 (0.012 to 0.025) | 0.017 (0.009 to 0.024) | 0.013 (0.006 to 0.020) | 0.008 (0.002 to 0.014) |

**Chosen model**, AUC inside each region (0.5 = coin flip)

| forecasts | 1 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| All forecasts | 0.758 | 0.583 | 0.575 | 0.551 | 0.549 |
| No alert at forecast time | 0.620 | 0.577 | 0.571 | 0.551 | 0.545 |

**All forecasts**, Brier skill vs climatology

| method | 1 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| persistence | 0.216 | -0.053 | -0.055 | -0.058 | -0.056 |
| logistic, 4 inputs | 0.257 | 0.009 | 0.006 | 0.006 | 0.005 |
| boosting, alert history | 0.264 | 0.023 | 0.017 | 0.012 | 0.007 |
| boosting + Telegram posts | 0.267 | 0.022 | 0.014 | 0.011 | 0.010 |

**No alert at forecast time**, Brier skill vs climatology

| method | 1 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| persistence | -0.001 | -0.063 | -0.056 | -0.059 | -0.054 |
| logistic, 4 inputs | 0.052 | 0.007 | 0.005 | 0.008 | 0.006 |
| boosting, alert history | 0.061 | 0.019 | 0.017 | 0.013 | 0.008 |
| boosting + Telegram posts | 0.065 | 0.019 | 0.014 | 0.012 | 0.011 |

**Gain from Telegram posts**, Brier skill of boosting + Telegram vs boosting, 95% interval

| forecasts | 1 h | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| All forecasts | 0.004 (-0.004 to 0.013) | -0.001 (-0.006 to 0.005) | -0.004 (-0.010 to 0.002) | -0.001 (-0.005 to 0.002) | 0.003 (-0.003 to 0.008) |
| No alert at forecast time | 0.005 (-0.004 to 0.014) | 0.000 (-0.005 to 0.006) | -0.003 (-0.010 to 0.003) | -0.001 (-0.005 to 0.003) | 0.003 (-0.002 to 0.008) |

**Gain from Telegram posts at 1 h by week of the window**

| week 1 | week 2 | week 3 | week 4 |
|---|---|---|---|
| 0.018 | 0.002 | 0.002 | -0.003 |

**Channel drift**

| period | posts per day | posts naming a known region |
|---|---|---|
| fit | 101 | 88% |
| validation | 180 | 90% |
| test | 255 | 79% |
