"""
07_sensitivity.py — 민감도 분석 및 공통방법편의 진단

수행 내용
    1) 응답 패턴 점검: 평가자별 표준편차, 고유값 개수, 직선응답 탐지
    2) 평가자 배제 민감도 (예: E20) — alpha, ICC, 순위 변화
    3) CQS 가중치 민감도: 무작위 디리클레 가중 1000회 + 차원 단독 순위
    4) 공통방법편의: 차원 간 상관, Harman 단일요인, 신뢰도

배제 기준은 결과를 보기 전에 정의되어야 한다. 본 스크립트는
'전 문항 응답 분산 0'이라는 응답 패턴 기준만을 사용하며,
점수 수준(관대/엄격)은 배제 사유로 삼지 않는다.

실행
    python code/07_sensitivity.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    OUT_DIR,
    DOMAINS, DIMENSIONS, ITEMS, DATA_PATH, load_data, experts_in,
    rating_matrix, krippendorff_alpha_interval, icc_all,
)

N_WEIGHT = 1000
SEED = 20260818


def main(path=DATA_PATH):
    df = load_data(path)
    rng = np.random.default_rng(SEED)

    # ------------------------------------------ 1) 응답 패턴
    print("=" * 84)
    print("1) 평가자별 응답 패턴")
    print("=" * 84)
    rows = []
    for e, g in df.groupby("Expert"):
        v = g[ITEMS].values.ravel()
        rows.append(dict(Expert=e, n=len(v), mean=v.mean(),
                         sd=v.std(ddof=1), n_unique=len(np.unique(v)),
                         zero_variance=v.std(ddof=1) == 0))
    pat = pd.DataFrame(rows).sort_values("mean")
    pat["Expert_num"] = pat.Expert.str[1:].astype(int)
    print(pat[["Expert", "n", "mean", "sd", "n_unique", "zero_variance"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    flagged = pat.loc[pat.zero_variance, "Expert"].tolist()
    print(f"\n  분산 0 (직선응답) 해당: {flagged if flagged else '없음'}")

    # ------------------------------------------ 2) 배제 민감도
    print()
    print("=" * 84)
    print("2) 배제 민감도")
    print("=" * 84)
    sens = []
    for drop in [None] + flagged:
        d = df if drop is None else df[df.Expert != drop]
        for dom in DOMAINS:
            ex = experts_in(d, dom)
            if len(ex) < 2:
                continue
            mat = rating_matrix(d, dom, ex)
            a = krippendorff_alpha_interval(mat.values)
            ic = icc_all(mat.T.values)
            order = (d[d.Domain == dom].groupby("M").CQS.mean()
                     .sort_values(ascending=False))
            sens.append(dict(dropped=drop or "-", Domain=dom, k=len(ex),
                             alpha=a, ICC_C1=ic["ICC_C1"],
                             ICC_Ck=ic["ICC_Ck"],
                             order=" > ".join(order.index)))
    sens = pd.DataFrame(sens)
    pd.set_option("display.width", 220)
    print(sens.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n  주: 결과를 본 뒤 배제하면 사후 조정으로 읽힌다. "
          "배제 기준과 시점을 본문에 명시할 것.")

    # ------------------------------------------ 3) 가중치 민감도
    print()
    print("=" * 84)
    print("3) CQS 가중치 민감도")
    print("=" * 84)
    dims = list(DIMENSIONS)
    for dom in DOMAINS:
        d = df[df.Domain == dom]
        base = d.groupby("M")[dims].mean()
        equal_order = tuple(base.mean(axis=1)
                            .sort_values(ascending=False).index)

        same = 0
        for _ in range(N_WEIGHT):
            w = rng.dirichlet(np.ones(len(dims)))
            score = (base.values * w).sum(axis=1)
            order = tuple(base.index[np.argsort(-score)])
            same += (order == equal_order)

        print(f"\n  [{dom}] 동등가중 순위: {' > '.join(equal_order)}")
        print(f"    무작위 가중 {N_WEIGHT}회 중 동일 순위 비율: "
              f"{same / N_WEIGHT:.3f}")
        for dim in dims:
            o = base[dim].sort_values(ascending=False)
            print(f"    {dim:20s} {' > '.join(o.index)}")

    # ------------------------------------------ 4) 공통방법편의
    print()
    print("=" * 84)
    print("4) 공통방법편의 진단")
    print("=" * 84)
    C = df[dims].corr()
    print("  차원 간 상관")
    print(C.to_string(float_format=lambda x: f"{x:.3f}"))

    # Harman 단일요인 (상관행렬 고유값 기반)
    eig = np.linalg.eigvalsh(C.values)[::-1]
    print(f"\n  고유값: {np.round(eig, 3)}")
    print(f"  제1요인 설명분산: {eig[0] / eig.sum():.3f}")
    print("  50%를 크게 상회하면 공통방법분산 가능성을 논의할 것")

    # 8문항 내적 일관성 (Cronbach alpha) — 도구 신뢰도이지 일치도가 아님
    X = df[ITEMS].values
    k = X.shape[1]
    ca = (k / (k - 1)) * (1 - X.var(axis=0, ddof=1).sum()
                          / X.sum(axis=1).var(ddof=1))
    print(f"\n  Cronbach's alpha (8문항) = {ca:.4f}")
    print("  주: 도구의 내적 일관성이며 평가자 간 일치도가 아니다. "
          "둘을 혼동하지 말 것.")

    pat.to_csv(OUT_DIR / "out_07_response_patterns.csv", index=False)
    sens.to_csv(OUT_DIR / "out_07_exclusion_sensitivity.csv", index=False)
    C.to_csv(OUT_DIR / "out_07_dimension_corr.csv")
    print("\nsaved: out_07_response_patterns.csv, "
          "out_07_exclusion_sensitivity.csv, out_07_dimension_corr.csv")
    return pat, sens, C


if __name__ == "__main__":
    main()
