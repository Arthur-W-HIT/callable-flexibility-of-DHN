# APEN-D-26-15702: data and computational materials

Identification method for duration-dependent callable flexibility in district heating networks under service constraints.

Repository: https://github.com/Arthur-W-HIT/callable-flexibility-of-DHN

Data and computational materials for the revised manuscript APEN-D-26-15702. The corrections of 18 October 2026 are incorporated below.

## Contents and units

- `inputs/`: raw/effective/equivalent ambient inputs, prices, availability flags, seasonal prehistory, and preselected weather windows. `weather_selection.json` records the selection rules. Weather originates from Open-Meteo historical data; the electricity-price shape is an anonymized historical sequence. The supplied sequences are the exact numerical inputs used here, not a claim about a current market tariff.
- `outputs/boundary_H6_floor16.csv`: the complete 2346-condition revised action library (51 ambient temperatures x 23 durations x 2 directions). Boundary categories and lower bounds remain explicit. `baseline_points.csv` and `baselines/` report the 51 converged baselines.
- `outputs/economic_configuration.json`: numerical CHP, state, action, and cost settings. Other CSVs contain paired economic ledgers, solver refinement, weather, pump, and history comparisons.
- `dispatch/`: saved inputs, actions, parameters, and summaries. `Service15` etc. denote the main grid-DP schedules; `MatchedService15`/`MatchedNeutral15` use the paired MILP. Refined-action and sensitivity cases are separate and must not be substituted for the main action catalog.
- `replays/`: integrated physical-model output tables and compact plot series. `Inherited_*` uses common seasonal history; fresh-start cases are separate. `actual` denotes detailed-model output, not field measurements.
- `figure_data/`: exact CSVs used by the original main-figure programs, the appendix connection map, and the Appendix C week slices. `figures/` contains the final exports. Main Fig. 2 is retained; Appendix Fig. A.1 adds pipe identifiers. Main Fig. 14 retains three seasonal panels; Appendix Fig. C.1 aligns a representative week.
- `portable_dispatch/`, `portable_figures/`, and `plotting/`: portable scheduling and figure examples and the original plotting bodies with explicit local input bindings.
- `SOURCE_MANIFEST.json`: source identifiers, original hashes, packaged hashes, and declared path sanitization. `MANIFEST_SHA256.json` records SHA-256 hashes of the repository files. Text files use LF newlines; `.gitattributes` preserves exact bytes on checkout.

Heat power is in MWth, heat/pump energy in MWh, temperatures in degrees C, flow in kg/s, and economic values in CNY unless columns state otherwise. One 80 MW subnetwork contains ten pump components; values marked N5/80MWx5 are already aggregated over five subnetworks. Main weekly physical accounting covers 168 h plus 6 h paid recovery; the pump-history bridge uses 168 h; the season covers 112 days. Fixed heating charges are assumed. Service and heat-neutral matched comparisons have distinct ledgers.

## Run the portable examples

Use Python 3.12 and install the versions in `requirements.txt` in your own environment. From this folder:

```text
python portable_dispatch/reproduce_day.py
python portable_dispatch/verify_pump_accounting.py
python portable_figures/reproduce_baseline.py
python plotting/reproduce_main_figures.py
python plotting/reproduce_explanatory_figures.py
```

`portable_dispatch/reproduce_day.py` rebuilds the first main-week day with 15 min decisions, seven actions, and 641 requested state points, and checks the action sequence and objective against the included reference. It verifies a scheduling subset.

`portable_dispatch/verify_pump_accounting.py` checks exported pump energy and same-interval time-varying price billing for the main-week, paired, and seasonal ledgers without Dymola.

`portable_figures/reproduce_baseline.py` regenerates an additional baseline diagnostic and checks pixels in the tested environment. `plotting/reproduce_main_figures.py` regenerates the eleven main quantitative figures from their frozen data using the original plotting bodies. `plotting/reproduce_explanatory_figures.py` redraws the appendix topology and representative-week drivers. Different font/rendering installations can affect image bytes or pixels.

The framework label correction preserves the original vector artwork and changes only the decision-step labels. Reproduce it from the archived input with:

```text
python plotting/update_framework_labels.py figure_data/framework_before_label_update.pdf regenerated/framework.pdf
```


Optional full seven-day grid-DP or paired command-level MILP runs:

```text
python portable_dispatch/dispatch.py --case Reproduced_Service15 --hold 1 --grid 641 --days 7
python portable_dispatch/fairness.py --case Reproduced_Neutral15 --neutral --hold 1 --days 7
```

The MILP uses a finite per-day time limit; feasible incumbents, gaps, and timings may vary. Neither command reruns physical replay. Physical heat neutrality and pump-electricity economics must be checked against continuous replay; command-level neutrality alone is not a physical heat balance.

## Complete dynamic model and dependencies

The complete dynamic model and component dependencies are available from the corresponding authors upon reasonable academic request. Physical simulations used Dymola 2023x, IBPSA 4.0.0, and the recorded Modelica dependencies. Commercial software, third-party libraries, original large MAT outputs, model source, and full state-initialization files are not included in this public-data candidate. Re-running the complete physical identification/replay workflow requires those model materials and a configured simulation environment. The included exported replay data and portable scheduling/plotting examples do not establish independent physical validation or an already tested full cross-machine simulation pipeline.

The repository contains exported numerical data and computational scripts. It does not redistribute commercial software or third-party model libraries. No additional software or data license is granted by this README.

## Current revision: time-varying pump pricing

Both the optimization pump-cost proxy and replay-based pump costs use the contemporaneous 15-minute `Price_CNY_per_MWh` series. The pump sensitivity multiplier is applied once to this same series. The final ledger replaces the proxy with integrated physical-replay pump electricity and includes the stated recovery period.

`CURRENT_RESULTS.json` identifies the current schedules, replay paths, economic ledgers, and figure data. `HISTORICAL_RESULTS.json` identifies retained earlier diagnostics; these are excluded from the current manuscript economic tables. Main-week, paired-neutrality, weather, price/fuel/pump sensitivities, the heating season, and numerical refinement have been recomputed for the current price rule. The boundary-identification library is unchanged because its physical admissibility tests do not use electricity prices.

Figures 11--14 and Appendix Fig. C.1 reuse the original plotting bodies with the updated numerical data. Figure 9 retains its corrected annotation clearance; Figure 13 includes all five decision steps. Source figure layouts, fonts, colors, and line styles are retained.
