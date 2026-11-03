# Model card: BG day-ahead price forecast

## Model
* **Task:** forecast the 24 hourly day-ahead prices of the Bulgarian bidding zone (EUR/MWh) for the next delivery day, at gate
  closure (12:00 CET on D-1).
* **Headline model:** LightGBM regression (Optuna-tuned, 50 trials, seed 42) on 51 features: calendar, price lags and rolling
  statistics, neighbour prices (RO, GR, HU) of D-1, ENTSO-E day-ahead load / wind / solar forecasts and residual load, net flows up
  to D-2, and archived D-1 weather forecasts. Probabilistic output: quantile LightGBM (10/50/90 %) with conformal recalibration.
  Spike probability: LightGBM classifier on a level-relative definition.
* **Cross-checks shown next to it:** a level-adjusted LightGBM (target = price − yesterday's mean) and a 50/50 blend. These are
  hedges against extrapolation failure, **not** the validated headline model.
* **Not in the model (tested, did not help on CV):** fuel and carbon prices, interconnector capacities, plant outages (the latter
  cannot be used point-in-time from the API).

## Intended use
Analytics and decision support for the Bulgarian day-ahead market: understanding price drivers, hedging context, sizing a
day-ahead battery schedule. **Not** a trading system and not financial advice. Do not put capital behind these forecasts without
independent validation.

## Data
ENTSO-E Transparency Platform (prices, load, wind/solar forecasts, generation, flows, day-ahead NTC, reserve prices), Open-Meteo
(observed weather and archived D-1 forecasts), World Bank Pink Sheet and OilPriceAPI (gas), Yahoo Finance (Brent, carbon proxy).
Period 2023-01-01 to 2026-10-31, hourly (33,600 hours). 15-minute data are averaged to hourly.

## Evaluation (EUR/MWh, naive = same hour yesterday)
| | Walk-forward CV (25 months) | Final holdout (Aug–Oct 2026) |
|---|---|---|
| Naive MAE | 28.9 | 30.8 |
| LightGBM MAE | 22.8 | 30.0 |
| LightGBM skill vs naive (95 % interval) | +21 % [17, 24] | +3 % [−10, 19] |
| 10–90 % band coverage (target 80 %) | 80 % | 68 % |

Validation is expanding-window walk-forward with a final holdout that was never used for tuning, ensemble weights or thresholds.
Caveat: the holdout has since informed several *design discussions* (e.g. the level-anchored variants), which is why a live,
pre-gate-closure track record (`reports/live_track_record.csv`) is the intended clean test going forward.

## Battery revenue range (Monte Carlo, `src/montecarlo.py`)
The battery study settles a forecast-driven schedule at actual prices. To show a range rather than one number, past forecast errors
are resampled (schedule fixed by the forecast, 7-day blocks so bad regimes persist). The generator was chosen on the CV months only:
a simpler one that draws error vectors independently of the schedule covered 72 % of days in its 10-90 % range, because realised
revenue falls systematically short of the revenue expected at forecast prices (the optimiser's curse: the schedule sells where the
forecast is too high); the slippage generator covers 83 % on CV and **70 % on the holdout**, so the range is too narrow in a regime
shift and its tails are conservative (4.1 % predicted vs 0.6 % realised daily-loss rate). Scenario-year P5-P95 is about
EUR 74k-91k per MW-year (median 82k); the back-test realised EUR 87k.

## Re-tests of rejected inputs (`src/retest_optional.py`, weekly)
Last run 2026-11-03: capacity and fuels, re-run on the current data under the same CV-only rule, still do not help (CV MAE 20.53 and
20.60 against 20.13 without). Outages remain untestable without leakage: the API returns one revision per document (probe on April 2026:
22 documents, none with several) and only 1 of 180 required point-in-time snapshot days exist. The script re-runs the outage experiment
by itself once either precondition is met.

## Access and privacy of the deployed dashboard
The dashboard reads only precomputed files in `reports/`. An optional sign-in (`app/auth.py`: salted PBKDF2 hashes in Streamlit
secrets, constant-time comparison, lock-out after five failures) keeps casual visitors out of a hosted copy. It does not hide files in a
public repository; use a private repository and the host's viewer allow-list for anything sensitive. The interface is available in
English and Bulgarian; the Bulgarian text has not been reviewed by a native speaker.

## Known limitations and failure modes
* **Regime shifts.** Tree models cannot extrapolate. In October 2026 (average price €232/MWh, above anything in training) the model
  under-forecast by about €29/MWh on average. Level-anchored variants help in that episode but not in earlier level shifts, and no
  variant passed the pre-set adoption rule.
* **Bands** under-cover in a regime shift (68 % on the holdout).
* **Spikes and negative prices** are hard: error is several times larger in those hours; only 8 negative-price hours fall in the holdout.
* **Missing drivers:** plant outages (not usable point-in-time), hydro reservoir levels, intra-day gas moves, announced capacity changes.
* **Assumption:** the ENTSO-E day-ahead wind/solar forecast is treated as available at gate closure (its legal deadline is 18:00 on D-1).
* **Daily gas history** is only 16 months deep (free tier).
* The Diebold-Mariano test is mildly liberal (5.5–7 % false positives at the 5 % level, Monte Carlo); borderline p-values on the holdout
  (0.07–0.1) should be read accordingly.

## Ethical and operational notes
No personal data. Public market data only. The reserve-market and investment figures on the dashboard are illustrative: reserve prices
look administered and the markets are tiny, and all investment inputs are assumptions. Retraining rule: monthly, or when the rolling
30-day MAE stays more than 20 % above its CV level for 14 days or a key feature's PSI exceeds 0.25.
