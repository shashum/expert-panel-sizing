"""
00_derive_public_data.py — 원자료에서 공개용 파생 자료를 만든다(원자료가 있을 때만 실행).

data/derived/mean_squares.csv
    도메인 × 패널(full, first, repeat)별 이원 분산분석의 평균제곱(대상 간, 평가자 간, 잔차)과 대상 수, 평가자 수
data/derived/panel_means.csv
    도메인 × 평가 대상(모델 × 조건) 16개의 최초 패널 평균과 반복 패널 평균

개별 평가자의 응답은 포함하지 않는다. 이 두 파일만으로 30_reproduce_from_public_data.py가
급내상관 전 계열, α, 분산성분, 소요 패널 규모, 패널 간 순위상관, 판정 형태별 소요 인원을 재산출한다.

실행
    python code/00_derive_public_data.py
"""
import numpy as np
import pandas as pd

from common import ROOT, DOMAINS, load_data, experts_in, rating_matrix

OUT = ROOT / "data" / "derived"
OUT.mkdir(parents=True, exist_ok=True)
df = load_data()


def mean_squares(X):
    """X: 대상 × 평가자."""
    n, k = X.shape
    g = X.mean()
    ms_t = k * ((X.mean(1) - g) ** 2).sum() / (n - 1)
    ms_r = n * ((X.mean(0) - g) ** 2).sum() / (k - 1)
    res = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + g
    ms_e = (res ** 2).sum() / ((n - 1) * (k - 1))
    return n, k, ms_t, ms_r, ms_e


ms_rows, mean_rows = [], []
for dom in DOMAINS:
    ex = experts_in(df, dom)
    first = [e for e in ex if df.loc[df.Expert == e, "Cohort"].iloc[0] == "derivation"]
    repeat = [e for e in ex if e not in first]
    for panel, grp in [("full", ex), ("first", first), ("repeat", repeat)]:
        X = rating_matrix(df, dom, grp).values.T
        n, k, t, r, e = mean_squares(X)
        ms_rows.append(dict(domain=dom, panel=panel, n_targets=n, k_raters=k, ms_target=t, ms_rater=r, ms_resid=e))
    A = rating_matrix(df, dom, first, unit=("M", "Scenario"))
    B = rating_matrix(df, dom, repeat, unit=("M", "Scenario"))
    for (m, s), a, b in zip(A.columns, A.values.mean(0), B.values.mean(0)):
        mean_rows.append(dict(domain=dom, model=m, condition=s, first_mean=a, repeat_mean=b, k_first=len(first), k_repeat=len(repeat)))

pd.DataFrame(ms_rows).to_csv(OUT / "mean_squares.csv", index=False, float_format="%.10f")
pd.DataFrame(mean_rows).to_csv(OUT / "panel_means.csv", index=False, float_format="%.10f")
print(pd.DataFrame(ms_rows).round(4).to_string(index=False))
print(f"\nsaved: {OUT / 'mean_squares.csv'}, {OUT / 'panel_means.csv'}")
