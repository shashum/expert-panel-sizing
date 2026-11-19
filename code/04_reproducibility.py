"""
04_reproducibility.py — 코호트 간 순위 재현성

수행 내용
    1) 4개 모델 수준 Spearman rho + 정확 순열 검정
       (4! = 24가지뿐이므로 완전일치도 exact p = .083)
    2) 16개 평가대상(모델 × 시나리오) 수준 rho — 검정력 확보
    3) 분할반복(split-half): 평가자를 무작위 2분할한 재현성 표집분포
       실제 관측된 완전일치가 얼마나 전형적인지 판정
    4) 패널별 모델 평균 점수 전체 표 (원고 표 6)

실행
    python code/04_reproducibility.py
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import OUT_DIR, DOMAINS, DATA_PATH, load_data, experts_in

N_SPLIT = 2000
SEED = 20260818


def exact_rank_test(x, y):
    """4개 항목 순위 상관의 정확 순열 양측 p."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = stats.spearmanr(x, y).statistic
    null = [stats.spearmanr(x, y[list(p)]).statistic
            for p in itertools.permutations(range(len(y)))]
    null = np.asarray(null)
    p = float(np.mean(np.abs(null) >= abs(rho) - 1e-12))
    return rho, p


def main(path=DATA_PATH):
    df = load_data(path)
    rng = np.random.default_rng(SEED)
    model_rows, unit_rows, split_rows, score_rows = [], [], [], []

    for dom in DOMAINS:
        d = df[df.Domain == dom]
        deriv = d[d.Cohort == "derivation"]
        valid = d[d.Cohort == "validation"]

        # --- 코호트별 모델 평균 (보고용 전체 표)
        for label, sub in [("derivation", deriv), ("validation", valid),
                           ("full", d)]:
            if sub.empty:
                continue
            g = sub.groupby("M").CQS.agg(["mean", "std", "count"])
            for m, r in g.iterrows():
                se = r["std"] / np.sqrt(r["count"])
                score_rows.append(dict(
                    Domain=dom, Cohort=label, Model=m,
                    mean=r["mean"], sd=r["std"], n=int(r["count"]),
                    ci_lo=r["mean"] - 1.96 * se, ci_hi=r["mean"] + 1.96 * se))

        if deriv.empty or valid.empty:
            continue

        # --- 4개 모델 수준
        a = deriv.groupby("M").CQS.mean()
        b = valid.groupby("M").CQS.mean()
        models = sorted(a.index)
        if len(models) == 4 and set(models) == set(b.index):
            rho, p = exact_rank_test(a[models].values, b[models].values)
            model_rows.append(dict(
                Domain=dom, level="4 models", rho=rho, exact_p=p,
                deriv_order=" > ".join(a.sort_values(ascending=False).index),
                valid_order=" > ".join(b.sort_values(ascending=False).index)))

        # --- 16개 평가대상 수준
        au = deriv.groupby(["M", "Scenario"]).CQS.mean()
        bu = valid.groupby(["M", "Scenario"]).CQS.mean()
        idx = sorted(set(au.index) & set(bu.index))
        if len(idx) >= 5:
            sr = stats.spearmanr(au[idx].values, bu[idx].values)
            pr = stats.pearsonr(au[idx].values, bu[idx].values)
            unit_rows.append(dict(
                Domain=dom, n_units=len(idx),
                rho=sr.statistic, rho_p=sr.pvalue,
                r=pr.statistic, r_p=pr.pvalue))

        # --- 분할반복
        ex = experts_in(df, dom)
        N = len(ex)
        if N >= 4:
            U16 = d.pivot_table(index=["M", "Scenario"], columns="Expert",
                                values="CQS")[ex].values
            U4 = d.pivot_table(index="M", columns="Expert",
                               values="CQS")[ex].values
            h = N // 2
            r16, r4, ident = [], [], []
            for _ in range(N_SPLIT):
                p = rng.permutation(N)
                A, B = p[:h], p[h:2 * h]
                r16.append(stats.spearmanr(U16[:, A].mean(1),
                                           U16[:, B].mean(1)).statistic)
                rr = stats.spearmanr(U4[:, A].mean(1),
                                     U4[:, B].mean(1)).statistic
                r4.append(rr)
                ident.append(rr == 1.0)
            r16 = np.asarray(r16, dtype=float)
            r4 = np.asarray(r4, dtype=float)
            split_rows.append(dict(
                Domain=dom, half=h, n_iter=N_SPLIT,
                rho16_mean=np.nanmean(r16),
                rho16_lo=np.nanpercentile(r16, 2.5),
                rho16_hi=np.nanpercentile(r16, 97.5),
                rho4_mean=np.nanmean(r4),
                pr_identical_order=float(np.mean(ident))))

    model = pd.DataFrame(model_rows)
    unit = pd.DataFrame(unit_rows)
    split = pd.DataFrame(split_rows)
    scores = pd.DataFrame(score_rows)

    pd.set_option("display.width", 240)
    print("=" * 100)
    print("(1) 4개 모델 순위 재현 — 정확 순열 검정")
    print("=" * 100)
    print(model.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("주: 모델이 4개뿐이면 완전일치라도 exact p = .083으로 유의에 도달 불가")

    print()
    print("=" * 100)
    print("(2) 16개 평가대상 수준 재현 — 검정력 확보")
    print("=" * 100)
    print(unit.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print()
    print("=" * 100)
    print("(3) 분할반복 재현성 표집분포")
    print("=" * 100)
    print(split.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("주: pr_identical_order = 무작위 2분할에서 4모델 순위가 "
          "완전히 일치할 확률")

    print()
    print("=" * 100)
    print("(4) 코호트 × 도메인 × 모델 평균 (95% CI)")
    print("=" * 100)
    print(scores.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    model.to_csv(OUT_DIR / "out_04_rank_models.csv", index=False)
    unit.to_csv(OUT_DIR / "out_04_rank_units.csv", index=False)
    split.to_csv(OUT_DIR / "out_04_splithalf.csv", index=False)
    scores.to_csv(OUT_DIR / "out_04_model_scores.csv", index=False)
    print("\nsaved: out_04_rank_models.csv, out_04_rank_units.csv, "
          "out_04_splithalf.csv, out_04_model_scores.csv")
    return model, unit, split, scores


if __name__ == "__main__":
    main()
