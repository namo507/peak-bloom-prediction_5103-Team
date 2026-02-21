# Cherry Blossom Peak Bloom Prediction — Team 5103 (2026)

> **Entry for the 2026 George Mason University Cherry Blossom Peak Bloom Prediction Competition.**

## Overview

This repository contains our predictions for the **2026** peak bloom date at five sites:

| Location | Prediction (DOY) | Interval |
|---|---|---|
| Kyoto, Japan | 90 | [83, 97] |
| Liestal-Weideli, Switzerland | 88 | [82, 93] |
| New York City, NY, USA | 92 | [84, 100] |
| Vancouver, BC, Canada | 92 | [82, 103] |
| Washington, D.C., USA | 83 | [75, 90] |

## Repository contents

| File | Description |
|---|---|
| `solution.qmd` | **Primary submission** — R Quarto document (includes abstract, models, predictions, and intervals). Fully reproducible. |
| `Solution.ipynb` | **Secondary pipeline** — Python Jupyter notebook with an independent GBR-based ensemble. |
| `cherry-predictions.csv` | R-generated predictions (competition schema). |
| `cherry-predictions-python.csv` | Python-generated predictions (competition schema). |
| `cherry-predictions-final.csv` | Blended final submission (R + Python averaged when methods agree). |
| `data/` | Competition and auxiliary data files. |
| `demo_analysis.qmd` | Demo analysis provided by the competition organizers. |

## Approach (see abstract in solution.qmd for full details)

We build a **two-model ensemble**:

1. **Model A — Local trend**: site-level recency-weighted quadratic regression (exponential decay, half-life ≈ 6 yr).
2. **Model B — Pooled GAM / GBR**: cross-site nonlinear model trained on competition + auxiliary data (Japan regional, MeteoSwiss, South Korea, USA-NPN phenometrics & status-intensity observations).
3. **Blending**: inverse-MAE weights from rolling-origin backtesting (1900–latest year).
4. **Intervals**: split-conformal 90th-percentile residual quantiles per location.

A Python GBR pipeline provides a cross-language robustness check; when both pipelines agree within 4 days, predictions and interval bounds are averaged.

## Reproducing results

```sh
# R solution (requires R ≥ 4.3, quarto, tidyverse, lubridate, mgcv)
quarto render solution.qmd

# Python solution (requires Python ≥ 3.9, numpy, pandas, scikit-learn)
jupyter nbconvert --to notebook --execute Solution.ipynb --inplace
```

## Original template

This repository was forked from the [GMU Cherry Blossom Competition template](https://github.com/GMU-CherryBlossomCompetition/peak-bloom-prediction).
The demo analysis requires a [working installation of R](https://cran.r-project.org) (version ≥ 4.3 strongly suggested), an [installation of quarto](https://www.quarto.org) (e.g., as part of the RStudio IDE), as well as the `tidyverse` suite of packages.

## Competition rules

Full competition rules at https://competition.statistics.gmu.edu.

**Entries must be submitted by the end of February 28, 2026 (anywhere on earth).**

Predictions are judged based on the sum of absolute differences between predicted and actual peak bloom dates across all five sites.  Prediction intervals are evaluated by coverage count (ties broken by sum of squared interval widths).

The true bloom dates for 2026 are taken to be the dates reported by:

- **Kyoto (Japan):** a local newspaper from Arashiyama,
- **Washington, D.C. (USA):** National Park Service,
- **Liestal-Weideli (Switzerland):** MeteoSwiss,
- **Vancouver, BC (Canada):** Vancouver Cherry Blossom Festival / UBC Botanical Garden,
- **New York City, NY (USA):** Washington Square Park Eco Projects / Nature Lab.

## License

![CC-BYNCSA-4](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)

Unless otherwise noted, the content in this repository is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

For the data sets in the _data/_ folder, please see [_data/README.md_](data/README.md) for the applicable copyrights and licenses.
