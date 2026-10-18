# APEN-D-26-15702: data and computational materials

Identification method for duration-dependent callable flexibility in district heating networks under service constraints.

This folder is a local release candidate prepared on 18 October 2026. It has not been uploaded and has no public repository URL yet.

## Contents and units

- `inputs/`: raw/effective/equivalent ambient inputs, prices, availability flags, seasonal prehistory, and preselected weather windows. `weather_selection.json` records the selection rules. Weather originates from Open-Meteo historical data; the electricity-price shape is an anonymized historical sequence. The supplied sequences are the exact numerical inputs used here, not a claim about a current market tariff.
- `outputs/boundary_H6_floor16.csv`: the complete 2346-condition revised action library (51 ambient temperatures x 23 durations x 2 directions). Boundary categories and lower bounds remain explicit. `baseline_points.csv` and `baselines/` report the 51 converged baselines.
- `outputs/economic_configuration.json`: numerical CHP, state, action, and cost settings. Other CSVs contain paired economic ledgers, solver refinement, weather, pump, and history comparisons.
- `dispatch/`: saved inputs, actions, parameters, and summaries. `Service15` etc. denote the main grid-DP schedules; `MatchedService15`/`MatchedNeutral15` use the paired MILP. Refined-action and sensitivity cases are separate and must not be substituted for the main action catalog.
- `replays/`: integrated physical-model output tables and compact plot series. `Inherited_*` uses common seasonal history; fresh-start cases are separate. `actual` denotes detailed-model output, not field measurements.
- `figure_data/`: exact CSVs used by the original main-figure programs, the appendix connection map, and the supplementary week slices. `figures/` contains the final exports. Main Fig. 2 is retained; Appendix Fig. A.1 adds pipe identifiers. Main Fig. 14 retains three seasonal panels; Supplementary Fig. S1 aligns a representative week.
- `portable_dispatch/`, `portable_figures/`, and `plotting/`: portable scheduling and figure examples and the original plotting bodies with explicit local input bindings.
- `SOURCE_MANIFEST.json`: source identifiers, original hashes, packaged hashes, and declared path sanitization. `MANIFEST_SHA256.json` freezes the final local release files.

Heat power is in MWth, heat/pump energy in MWh, temperatures in degrees C, flow in kg/s, and economic values in CNY unless columns state otherwise. One 80 MW subnetwork contains ten pump components; values marked N5/80MWx5 are already aggregated over five subnetworks. Main weekly physical accounting covers 168 h plus 6 h paid recovery; the pump-history bridge uses 168 h; the season covers 112 days. Fixed heating charges are assumed. Service and heat-neutral matched comparisons have distinct ledgers.

## Run the portable examples

Use Python 3.12 and install the versions in `requirements.txt` in your own environment. From this folder:

```text
python portable_dispatch/reproduce_day.py
python portable_figures/reproduce_baseline.py
python plotting/reproduce_main_figures.py
python plotting/reproduce_explanatory_figures.py
```

The first example rebuilds the first main-week day with 15 min decisions, seven actions, and 641 requested state points, and checks the action sequence and objective against the included reference. It verifies a scheduling subset. The second regenerates an additional baseline diagnostic and checks pixels in the tested environment. The third regenerates the eleven refreshed main result figures from their frozen revised data using the original plotting bodies. The fourth redraws the appendix topology and supplementary week from verified exported data. Different font/rendering installations can affect image bytes or pixels.

Optional full seven-day grid-DP or paired command-level MILP runs:

```text
python portable_dispatch/dispatch.py --case Reproduced_Service15 --hold 1 --grid 641 --days 7
python portable_dispatch/fairness.py --case Reproduced_Neutral15 --neutral --hold 1 --days 7
```

The MILP uses a finite per-day time limit; feasible incumbents, gaps, and timings may vary. Neither command reruns physical replay. Physical heat neutrality and pump-electricity economics must be checked against continuous replay; command-level neutrality alone is not a physical heat balance.

## Complete dynamic model and dependencies

The complete dynamic model and component dependencies are available from the corresponding authors upon reasonable academic request. Physical simulations used Dymola 2023x, IBPSA 4.0.0, and the recorded Modelica dependencies. Commercial software, third-party libraries, original large MAT outputs, model source, and full state-initialization files are not included in this public-data candidate. Re-running the complete physical identification/replay workflow requires those model materials and a configured simulation environment. The included exported replay data and portable scheduling/plotting examples do not establish independent physical validation or an already tested full cross-machine simulation pipeline.

No publication, repository upload, external access, or licensing choice is implied by preparing this folder. Before public release, the authors must choose the repository/license, confirm redistribution rights for inputs and dependencies, upload the frozen files, verify access, and insert the permanent link in the manuscript and response.
