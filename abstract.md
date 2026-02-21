## Blinded Abstract

This entry predicts 2026 peak bloom date at five sites using an interpretable,
reproducible ensemble designed for nonstationary phenology under climate
change. The central idea is to combine (1) site-specific temporal dynamics and
(2) cross-site shared structure learned from broader auxiliary records.

Model A is a local recency-weighted quadratic trend, fitted separately for each
competition location. Exponential time-decay weights emphasize recent years,
allowing the model to adapt to acceleration in bloom timing while still using
long historical context where available (e.g., Kyoto and Washington, D.C.).
This component captures location-level momentum and curvature that are often
lost in fully pooled models.

Model B is a pooled nonlinear learner trained on competition data plus public
auxiliary phenology sources (Japan regional records, MeteoSwiss data, South
Korea data, and USA-NPN records for New York City including both status-
intensity observations and individual phenometrics for site 32789/species 228).
The pooled model captures transferable relationships across geography and time.
In R, this component is a GAM; in Python, an independent Gradient Boosting
Regressor provides a robustness check.

Selected features are: calendar year, centered year and squared year (trend and
curvature), latitude, longitude, altitude (log-transformed), source indicator,
and observation-count depth per site. These variables were chosen for
interpretability and direct ecological relevance: bloom timing is strongly tied
to climate gradients (latitude/altitude), long-run warming trajectories (year
terms), and data reliability/coverage (site-depth feature).

Final point predictions are produced by inverse-MAE blending of Model A and
Model B using rolling-origin backtesting over historical years. This avoids
fixed ad hoc weights and lets out-of-sample error determine contribution from
each component.

Prediction intervals are calibrated with conformal logic: for each site, the
90th percentile of historical absolute residuals (from the rolling backtest) is
used as an error half-width. This yields location-aware uncertainty bands that
balance coverage and interval sharpness under the competition tie-break rule.

The full workflow is implemented in a reproducible Quarto analysis with a
parallel Python notebook, uses only publicly available data, and does not rely
on manual post-hoc adjustment of predictions.