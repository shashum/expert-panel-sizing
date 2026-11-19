"""
03_icc.py — 급내상관 및 목표 신뢰도 달성 소요 패널 규모

핵심 논지
    ICC(C,1)은 패널 규모에 불변이고, ICC(C,k)만 Spearman-Brown을 따라
    상승한다. 따라서 패널 확대는 합의 수준을 높이는 것이 아니라
    평균화로 오차를 줄일 뿐이다.

수행 내용
    1) 코호트별 ICC(C,1)/ICC(C,k)/ICC(A,1)/ICC(A,k)
    2) 평가대상 수준 분산분해 (대상 / 평가자 / 잔차)
    3) 패널 규모별 ICC 리샘플링 곡선
    4) ICC(C,1) 부트스트랩 CI 및 목표 신뢰도 소요 k

실행
    python code/03_icc.py
"""

import numpy as np
import pandas as pd

from common import (
    OUT_DIR,
    DOMAINS, DATA_PATH, load_data, experts_in, rating_matrix,
    icc_all, spearman_brown_k, enumerate_subsets,
)

TARGETS = (0.70, 0.80)
N_BOOT = 5000
CAP = 2000
SEED = 20260818


def main(path=DATA_PATH):
    df = load_data(path)
    rng = np.random.default_rng(SEED)
    cohort_rows, curve_rows, need_rows = [], [], []

    for dom in DOMAINS:
        ex = experts_in(df, dom)
        # (units × raters) 방향
        U = rating_matrix(df, dom, ex).T
        A = U.values
        N = A.shape[1]

        deriv = [e for e in ex
                 if df.loc[df.Expert == e, "Cohort"].iloc[0] == "derivation"]
        valid = [e for e in ex if e not in deriv]

        for label, grp in [("derivation", deriv),
                           ("validation", valid),
                           ("full", ex)]:
            if len(grp) < 2:
                continue
            r = icc_all(U[grp].values)
            r.update(Domain=dom, Cohort=label, k=len(grp))
            cohort_rows.append(r)

        # 패널 규모별 곡선
        for n in range(2, N + 1):
            subs, exhaustive = enumerate_subsets(list(range(N)), n, cap=CAP)
            vals = {key: [] for key in ("ICC_C1", "ICC_Ck", "ICC_A1", "ICC_Ak")}
            for s in subs:
                r = icc_all(A[:, s])
                for key in vals:
                    vals[key].append(r[key])
            row = dict(Domain=dom, n=n, k_subsets=len(subs),
                       exhaustive=exhaustive)
            for key, arr in vals.items():
                arr = np.asarray(arr, dtype=float)
                row[key + "_mean"] = np.nanmean(arr)
                row[key + "_lo"] = np.nanpercentile(arr, 2.5)
                row[key + "_hi"] = np.nanpercentile(arr, 97.5)
            curve_rows.append(row)

        # 부트스트랩(평가대상 재표집) + 소요 패널 규모
        full = icc_all(A)
        r1 = full["ICC_C1"]
        bs = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, A.shape[0], A.shape[0])
            try:
                bs.append(icc_all(A[idx])["ICC_C1"])
            except Exception:
                pass
        bs = np.asarray(bs, dtype=float)
        bs = bs[np.isfinite(bs)]
        need = dict(Domain=dom, k_observed=N, ICC_C1=r1,
                    lo=np.percentile(bs, 2.5) if bs.size else np.nan,
                    hi=np.percentile(bs, 97.5) if bs.size else np.nan,
                    ICC_Ck=full["ICC_Ck"],
                    p_target=full["p_target"], p_rater=full["p_rater"],
                    p_resid=full["p_resid"])
        for t in TARGETS:
            need[f"k_for_{int(t * 100)}"] = np.ceil(spearman_brown_k(r1, t))
        need_rows.append(need)

    cohort = pd.DataFrame(cohort_rows)
    curve = pd.DataFrame(curve_rows)
    need = pd.DataFrame(need_rows)

    pd.set_option("display.width", 240)
    print("=" * 96)
    print("코호트별 급내상관 (평가대상 = 모델 × 시나리오 16개)")
    print("=" * 96)
    print(cohort[["Domain", "Cohort", "k", "ICC_C1", "ICC_Ck",
                  "ICC_A1", "ICC_Ak"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print()
    print("=" * 96)
    print("패널 규모별 ICC (평균)")
    print("=" * 96)
    print(curve[["Domain", "n", "ICC_C1_mean", "ICC_Ck_mean",
                 "ICC_Ck_lo", "ICC_Ck_hi"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\nICC(C,1)이 n에 대해 평탄하면 패널 확대는 합의를 만들지 않는다.")

    print()
    print("=" * 96)
    print("분산분해 및 목표 신뢰도 소요 패널 규모")
    print("=" * 96)
    print(need.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    cohort.to_csv(OUT_DIR / "out_03_icc_cohort.csv", index=False)
    curve.to_csv(OUT_DIR / "out_03_icc_curve.csv", index=False)
    need.to_csv(OUT_DIR / "out_03_required_panel.csv", index=False)
    print("\nsaved: out_03_icc_cohort.csv, out_03_icc_curve.csv, "
          "out_03_required_panel.csv")
    return cohort, curve, need


if __name__ == "__main__":
    main()
