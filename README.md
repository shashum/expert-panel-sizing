# expert-panel-sizing

Code and derived data for the manuscript
"How Large Must an Expert Panel Be? Decision Type and Cross-Panel Reproducibility
in Evaluating LLM-Generated Threat Scenarios" (under review).

## Contents
- `code/` — analysis, simulation and figure scripts
- `data/derived/` — mean squares and panel means (individual responses are not shared)
- `outputs/figures/` — Figures 1–5

## Reproduce
pip install -r requirements.txt
python code/30_reproduce_from_public_data.py   # main results from the public data
./run_all.sh                                   # full pipeline (requires raw data in data/raw/)

## Data availability
Individual responses are not shared because participant consent did not cover
their release and the small population of inspection organisations makes
re-identification possible. The derived data are sufficient to recompute the
reliability coefficients, variance components, required panel sizes and
cross-panel rank correlations.

## License
Code: MIT. Derived data: CC BY 4.0. Figures are not separately licensed;
they can be regenerated with the code.
