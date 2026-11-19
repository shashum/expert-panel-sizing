"""
05_variance_groups.py — 평가자 severity 및 집단 비교

수행 내용
    1) 평가자 간 분산 비율 (severity heterogeneity) + 부트스트랩 CI
    2) 코호트 간 Mann-Whitney U (등가성 주장 금지 문구 포함)
    3) 도메인 간 Kruskal-Wallis + Dunn 사후비교(Bonferroni)
    4) CQS 도메인 × 모델 표
    5) 차원별 평균 (Technical Validity 최저 여부 확인)

주의
    비유의한 Mann-Whitney 결과는 등가성의 근거가 아니다.
    참고용으로 TOST 등가성 검정을 함께 출력한다.

실행
    python code/05_variance_groups.py
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import OUT_DIR, DIMENSIONS, DATA_PATH, load_data

N_BOOT = 5000
SEED = 20260818
TOST_MARGIN = 0.5      # 등가 한계 (5점 척도 기준 0.5점)


def main(path=DATA_PATH):
    df = load_data(path)
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------- 1) 분산 비율
    em = df.groupby("Expert").CQS.mean()
    total_var = df.CQS.var(ddof=1)
    between_var = em.var(ddof=1)
    ratio = between_var / total_var

    boot = []
    experts = em.index.to_numpy()
    for _ in range(N_BOOT):
        pick = rng.choice(experts, len(experts), replace=True)
        sub = pd.concat([df[df.Expert == e] for e in pick])
        b = sub.groupby("Expert").CQS.mean().var(ddof=1)
        t = sub.CQS.var(ddof=1)
        if np.isfinite(b) and t > 0:
            boot.append(b / t)
    boot = np.asarray(boot)

    print("=" * 78)
    print("1) 평가자 severity heterogeneity")
    print("=" * 78)
    print(f"  전체 평균          {df.CQS.mean():.3f}")
    print(f"  평가자 간 분산     {between_var:.4f}")
    print(f"  전체 분산          {total_var:.4f}")
    print(f"  비율               {ratio:.4f} "
          f"[{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]")
    print(f"  평가자 평균 범위   {em.min():.3f} ~ {em.max():.3f}")
    print("\n  용어: 'response bias'가 아니라 'evaluator severity "
          "heterogeneity'로 기술할 것")

    # ---------------------------------------------- 2) 코호트 비교
    d_mean = em[em.index.isin(df[df.Cohort == "derivation"].Expert.unique())]
    v_mean = em[em.index.isin(df[df.Cohort == "validation"].Expert.unique())]
    U, p = stats.mannwhitneyu(d_mean, v_mean, alternative="two-sided")

    # TOST (Welch t 기반 등가성)
    t1 = stats.ttest_ind(d_mean, v_mean - TOST_MARGIN,
                         equal_var=False, alternative="greater")
    t2 = stats.ttest_ind(d_mean, v_mean + TOST_MARGIN,
                         equal_var=False, alternative="less")
    p_tost = max(t1.pvalue, t2.pvalue)

    print()
    print("=" * 78)
    print("2) 코호트 간 severity 비교")
    print("=" * 78)
    print(f"  derivation M = {d_mean.mean():.3f} (n = {len(d_mean)})")
    print(f"  validation M = {v_mean.mean():.3f} (n = {len(v_mean)})")
    print(f"  Mann-Whitney U = {U:.1f}, p = {p:.4f}")
    print(f"  TOST (±{TOST_MARGIN}) p = {p_tost:.4f}")
    print("\n  비유의한 U 검정은 등가성을 입증하지 않는다. "
          "등가성을 주장하려면 TOST 결과를 근거로 삼을 것.")

    # ---------------------------------------------- 3) 도메인 비교
    groups = [g.CQS.values for _, g in df.groupby("Domain")]
    names = [k for k, _ in df.groupby("Domain")]
    H, pk = stats.kruskal(*groups)

    print()
    print("=" * 78)
    print("3) 도메인 간 비교")
    print("=" * 78)
    print(f"  Kruskal-Wallis H({len(groups) - 1}) = {H:.4f}, p = {pk:.4f}")
    for name, g in zip(names, groups):
        print(f"    {name:15s} M = {g.mean():.3f}  Mdn = {np.median(g):.3f}  "
              f"SD = {g.std(ddof=1):.3f}  n = {len(g)}")

    pairs = list(itertools.combinations(range(len(groups)), 2))
    print("\n  사후비교 (Mann-Whitney, Bonferroni 보정)")
    for i, j in pairs:
        u, pp = stats.mannwhitneyu(groups[i], groups[j],
                                   alternative="two-sided")
        print(f"    {names[i]:15s} vs {names[j]:15s} "
              f"U = {u:8.1f}  p = {pp:.4f}  p_adj = "
              f"{min(1.0, pp * len(pairs)):.4f}")
    print("\n  주의: 도메인 효과는 평가자 severity와 교락되어 있다. "
          "패널 구성이 도메인별로 다르면 실질적 도메인 차이로 해석 불가.")

    # ---------------------------------------------- 4) CQS 표
    print()
    print("=" * 78)
    print("4) CQS (동등 가중): 도메인 × 모델")
    print("=" * 78)
    cqs = (df.groupby(["Domain", "M"]).CQS
           .agg(["mean", "std", "count"]).reset_index()
           .sort_values(["Domain", "mean"], ascending=[True, False]))
    print(cqs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------- 5) 차원별 평균
    print()
    print("=" * 78)
    print("5) 평가 차원별 평균")
    print("=" * 78)
    dim = df[list(DIMENSIONS)].agg(["mean", "std"]).T
    dim["rank"] = dim["mean"].rank(ascending=False).astype(int)
    print(dim.to_string(float_format=lambda x: f"{x:.4f}"))

    cqs.to_csv(OUT_DIR / "out_05_cqs.csv", index=False)
    dim.to_csv(OUT_DIR / "out_05_dimensions.csv")
    em.to_csv(OUT_DIR / "out_05_evaluator_means.csv")
    print("\nsaved: out_05_cqs.csv, out_05_dimensions.csv, "
          "out_05_evaluator_means.csv")
    return cqs, dim, em


if __name__ == "__main__":
    main()
