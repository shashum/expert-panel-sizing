"""
01_alpha.py — 도메인 × 패널별 Krippendorff's alpha(간격 계량)와 부트스트랩 신뢰구간

평가 대상(모델 × 시나리오 16개)을 재표집하는 부트스트랩 5,000회. 원고 표 3의 α 열.

실행
    python code/01_alpha.py
"""

import numpy as np
import pandas as pd

from common import (
    DOMAINS, DATA_PATH, load_data, experts_in, rating_matrix,
    krippendorff_alpha_interval, OUT_DIR,
)

N_BOOT = 5000
SEED = 20260818


def bootstrap_alpha(mat, n_boot=N_BOOT, seed=SEED):
    """평가대상(열) 재표집 부트스트랩."""
    rng = np.random.default_rng(seed)
    A = np.asarray(mat, dtype=float)
    n_units = A.shape[1]
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_units, n_units)
        a = krippendorff_alpha_interval(A[:, idx])
        if np.isfinite(a):
            out.append(a)
    out = np.asarray(out)
    if out.size == 0:
        return np.nan, np.nan
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def main(path=DATA_PATH):
    df = load_data(path)
    rows = []

    for dom in DOMAINS:
        allx = experts_in(df, dom)
        deriv = [e for e in allx if df.loc[df.Expert == e, "Cohort"].iloc[0]
                 == "derivation"]
        valid = [e for e in allx if e not in deriv]

        for label, grp in [("derivation", deriv),
                           ("validation", valid),
                           ("full", allx)]:
            if len(grp) < 2:
                rows.append(dict(Domain=dom, Cohort=label, m=len(grp),
                                 alpha=np.nan, lo=np.nan, hi=np.nan))
                continue
            mat = rating_matrix(df, dom, grp).values      # (raters × units)
            a = krippendorff_alpha_interval(mat)
            m = len(grp)
            lo, hi = bootstrap_alpha(mat)
            rows.append(dict(Domain=dom, Cohort=label, m=m,
                             alpha=a, lo=lo, hi=hi))

    res = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    print("=" * 78)
    print("Krippendorff's alpha (interval) — 평가대상 = 모델 × 시나리오 16개")
    print("=" * 78)
    print(res[["Domain", "Cohort", "m", "alpha", "lo", "hi"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


    res.to_csv(OUT_DIR / "out_01_alpha.csv", index=False)
    print("\nsaved: out_01_alpha.csv")
    return res


if __name__ == "__main__":
    main()
