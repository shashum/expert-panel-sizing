# Data

## raw/ (not included)

The individual responses are not distributed (see the main README). To run the full pipeline, place the response file at `data/raw/survey.xlsx`, or point the environment variable `SURVEY_XLSX` to it. The scripts expect:

- one sheet named `종합`
- one row per rater × domain × model × scenario (416 rows)
- columns `Expert` (E1–E21; E6 is absent because the response was incomplete), `Domain` (`Communication`, `Energy`, `Healthcare`), `Model` (`LLaMA 3.1 8B Instruct`, `Foundation-Sec-8B-Instruct`, `Qwen3-8B`, `DeepSeek-R1-0528-Qwen3-8B`), `Scenario` (`Basic/Simple`, `Basic/Complex`, `Expert/Simple`, `Expert/Complex`), and `Q1`–`Q8` (5-point ratings)

Panel membership is defined in `code/common.py` (`DERIVATION` = first panel, `VALIDATION` = repeat panel).

## derived/

Produced by `code/00_derive_public_data.py`.

**mean_squares.csv** — two-way ANOVA (targets × raters) on the mean of the eight items, by domain and panel.

| Column | Meaning |
|---|---|
| `domain` | Communication, Energy, Healthcare |
| `panel` | `full` (all raters of the domain), `first`, `repeat` |
| `n_targets` | number of evaluated targets (16 = 4 models × 4 conditions) |
| `k_raters` | number of raters |
| `ms_target`, `ms_rater`, `ms_resid` | mean squares for targets, raters and residual |

**panel_means.csv** — mean score of each target in each panel.

| Column | Meaning |
|---|---|
| `domain`, `model`, `condition` | target (condition = query level / structural complexity) |
| `first_mean`, `repeat_mean` | mean of the eight-item scores over the raters of each panel |
| `k_first`, `k_repeat` | panel sizes |
