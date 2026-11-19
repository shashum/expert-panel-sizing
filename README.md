# expert-panel-sizing

Analysis code, simulation code and derived data for the manuscript

> **How Large Must an Expert Panel Be? Decision Type and Cross-Panel Reproducibility in Evaluating LLM-Generated Threat Scenarios**
> Anonymous authors (under review).

The study examines at what panel size, and for which type of decision, small expert panels can serve as evidence for adopting LLM-generated threat scenarios. Twenty domain experts (communication, energy, healthcare) rated scenarios from four open-weight 8B-parameter models (416 observations). Raters were split by recruitment time into two independent panels. The analyses estimate rater agreement, panel reliability and cross-panel reproducibility; the simulations, parameterised by the estimated variance components, examine when agreement and reproducibility diverge and how many raters comparative and threshold decisions require.

## Repository structure

```
code/            analysis, simulation and figure scripts (run from the repository root)
  common.py        data loading, panel definitions, reliability metrics
  sim_core.py      helpers for the simulations
data/
  raw/             individual responses — not included (see data/README.md)
  derived/         public derived data: mean squares and panel means
outputs/
  figures/         Figures 1–5 (SVG, PNG; Figure 5 also as editable PowerPoint)
  out_30_public_reproduction.json   main numbers reproduced from the public derived data
run_all.sh       runs every step in order
```

## Two ways to reproduce

**A. From the public derived data (no raw data needed).**

```bash
pip install -r requirements.txt
python code/30_reproduce_from_public_data.py
```

This recomputes, from `data/derived/` alone, Krippendorff's α, all four ICC forms, the variance components, the number of raters required for ICC(C,k) ≥ .70 and .80, the cross-panel rank correlations (16 targets and 4 models), the attenuation predictions and the inverse single-rater consistency of Table 8, the ratio of raters required by threshold and comparative decisions with its 95% interval, and the decision-accuracy table. The simulations of the separation conditions, attenuation and decision accuracy (`11_simulations.py`, blocks D–F of `12_sim_additions.py`) need only the variance components and can be adapted to start from them.

**B. Full pipeline (requires the raw data).** Place the response file at `data/raw/survey.xlsx` (or set `SURVEY_XLSX`) and run

```bash
./run_all.sh
```

Outputs go to `outputs/` and console logs to `outputs/logs/`. The full pipeline takes about two minutes on a laptop. Figure 5 additionally needs Node.js and `npm install pptxgenjs`.

## Scripts and manuscript items

| Script | Manuscript |
|---|---|
| `00_derive_public_data.py` | builds `data/derived/` from the raw data |
| `01_alpha.py` | Table 3 (α and bootstrap CI) |
| `02_resampling.py` | Section 4.3, Figure 1(a): subset enumeration and permutation null |
| `03_icc.py` | Table 3 (ICC), Table 5, Section 4.3 required panel sizes |
| `04_reproducibility.py` | Table 6, Section 4.4: four-model and 16-target rank correlations |
| `05_variance_groups.py` | Section 4.1: rater severity, panel comparison (Mann–Whitney, TOST), dimension means |
| `06_lme.py` | Table 2: linear mixed-effects model and residual diagnostics |
| `07_sensitivity.py` | Section 4.6: response patterns, exclusion of the zero-variance rater, weighting sensitivity, dimension correlations |
| `08_split_distributions.py` | Table 7, Figure 2, Section 4.5 (split distributions at the actual panel sizes); Section 5.4 ratio interval; Table 6 note (paired model differences, Holm); Section 4.1 (within-rater dimension contrasts, Holm) |
| `09_supplementary_checks.py` | Table 4; permutation p-values in Section 4.4 and Tables 7 and 9; Section 4.5 (panel exchangeability, exclusion of shared raters, 16-target profiles of multi-domain raters, bootstrap of domain differences) |
| `10_sim_reproduce.py` | checks the inputs of the simulations |
| `11_simulations.py` | Section 5: model fit, separation conditions, robustness variants, attenuation, decision accuracy |
| `12_sim_additions.py` | Sections 4.3, 5.1–5.4, Table 10, Figures 1, 3, 4, Table 12 |
| `20_figures.py` | Figures 1–4 |
| `21_figure5.js` | Figure 5 (editable PowerPoint shapes) |
| `30_reproduce_from_public_data.py` | reproduction path A (Tables 1, 3, 8, 12; Section 4.3 required panel sizes; Section 5.4 ratio interval) |

Panel labels in the code: `derivation` = first panel, `validation` = repeat panel.

## Random seeds

All seeds are fixed in the scripts, so the full pipeline reproduces the reported values. Resampling-based quantities (bootstrap intervals, permutation p-values) depend on the seed; with other seeds they vary in the second or third decimal place.

## Data availability

Individual responses are not shared. Participant consent did not extend to the release of individual responses, and the responses include affiliation information; because the population of organisations performing critical-infrastructure inspections is small, combining domain assignment with affiliation carries a real risk of re-identification. The derived data in `data/derived/` (mean squares by domain and panel; mean scores of the 16 targets for each panel) are sufficient to recompute the reliability coefficients, required panel sizes, cross-panel rank correlations and variance components. The evaluated scenarios, prompts, XML schema and generation settings are available in the repository of the prior study cited in the manuscript.

## Environment

Python 3.13 with the packages in `requirements.txt` (tested versions are pinned). Figure 5: Node.js with pptxgenjs 4.0.

## License

Code: MIT License (see `LICENSE`). Derived data and figures: CC BY 4.0.
