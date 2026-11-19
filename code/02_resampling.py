"""
02_resampling.py — 패널 규모 리샘플링 및 순열 영분포

패널 규모에 따른 α의 상승이 신뢰도 개선인지 추정량의 성질인지 구분한다(원고 4.3절, 그림 1(a)).

수행 내용
    1) 각 패널 규모 n에 대해 전수 부분집합(가능하면) alpha 분포
    2) 순열 영분포: 평가자 내에서 평가대상 라벨을 무작위 치환
       (평가자 severity와 주변분포는 보존, 참 신호만 제거)
    3) 실제 도출 코호트 alpha의 동일 규모 부분집합 내 백분위
    4) 그림 저장 (matplotlib)

실행
    python code/02_resampling.py
"""

import itertools

import numpy as np
import pandas as pd

from common import (
    OUT_DIR,
    DOMAINS, DATA_PATH, load_data, experts_in, rating_matrix,
    krippendorff_alpha_interval, enumerate_subsets,
)

CAP = 2000        # 부분집합 전수열거 상한
N_NULL = 2000     # 순열 영분포 반복
SEED = 20260818
MAKE_PLOT = True


def null_distribution(A, n, n_null=N_NULL, rng=None):
    """평가자 내 라벨 치환 후 크기 n 부분집합의 alpha 분포."""
    rng = rng or np.random.default_rng(SEED)
    N = A.shape[0]
    out = []
    for _ in range(n_null):
        B = np.array([rng.permutation(row) for row in A])
        s = rng.choice(N, n, replace=False)
        a = krippendorff_alpha_interval(B[s])
        if np.isfinite(a):
            out.append(a)
    return np.asarray(out)


def main(path=DATA_PATH):
    df = load_data(path)
    rng = np.random.default_rng(SEED)
    rows, perc = [], []

    for dom in DOMAINS:
        ex = experts_in(df, dom)
        A = rating_matrix(df, dom, ex).values       # (raters × units)
        N = len(ex)

        for n in range(2, N + 1):
            subs, exhaustive = enumerate_subsets(list(range(N)), n, cap=CAP)
            obs = np.array([krippendorff_alpha_interval(A[s]) for s in subs])
            obs = obs[np.isfinite(obs)]
            nul = null_distribution(A, n, rng=rng)

            rows.append(dict(
                Domain=dom, n=n, k_subsets=len(obs), exhaustive=exhaustive,
                obs_mean=obs.mean(),
                obs_lo=np.percentile(obs, 2.5),
                obs_hi=np.percentile(obs, 97.5),
                null_mean=nul.mean(),
                null_lo=np.percentile(nul, 2.5),
                null_hi=np.percentile(nul, 97.5),
                pct_obs_in_null=float((nul < obs.mean()).mean() * 100),
            ))

        # 실제 도출 코호트의 위치
        deriv_idx = [i for i, e in enumerate(ex)
                     if df.loc[df.Expert == e, "Cohort"].iloc[0] == "derivation"]
        nd = len(deriv_idx)
        if 2 <= nd <= N:
            subs = [list(c) for c in itertools.combinations(range(N), nd)]
            dist = np.array([krippendorff_alpha_interval(A[s]) for s in subs])
            a_obs = krippendorff_alpha_interval(A[deriv_idx])
            perc.append(dict(Domain=dom, n_derivation=nd, alpha=a_obs,
                             mean_same_size=dist.mean(),
                             percentile=float((dist < a_obs).mean() * 100),
                             k_subsets=len(dist)))

    curve = pd.DataFrame(rows)
    place = pd.DataFrame(perc)

    pd.set_option("display.width", 240)
    print("=" * 100)
    print("패널 규모별 alpha: 관측 부분집합 vs 순열 영분포")
    print("=" * 100)
    print(curve.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print()
    print("=" * 100)
    print("실제 도출 코호트의 alpha가 동일 규모 부분집합 분포에서 차지하는 위치")
    print("=" * 100)
    print(place.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n해석: 관측 곡선이 영분포와 중첩되고 도출 코호트가 특이하지 않다면,")
    print("      alpha(n)의 상승은 추정량 편의이며 신뢰도 개선이 아니다.")

    curve.to_csv(OUT_DIR / "out_02_resampling_curve.csv", index=False)
    place.to_csv(OUT_DIR / "out_02_derivation_percentile.csv", index=False)
    print("\nsaved: out_02_resampling_curve.csv, out_02_derivation_percentile.csv")

    if MAKE_PLOT:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, len(DOMAINS), figsize=(13, 3.8))
            for ax, dom in zip(np.atleast_1d(axes), DOMAINS):
                c = curve[curve.Domain == dom]
                ax.fill_between(c.n, c.null_lo, c.null_hi, color="0.85",
                                label="Permutation null (95%)")
                ax.plot(c.n, c.null_mean, "--", color="0.45", lw=1.3)
                ax.fill_between(c.n, c.obs_lo, c.obs_hi, color="#c6dbef",
                                alpha=.75, label="Observed subsets (95%)")
                ax.plot(c.n, c.obs_mean, "o-", color="#08519c", ms=4, lw=1.8,
                        label="Observed mean")
                ax.axhline(0, color="k", lw=.8)
                ax.axhline(.667, color="#a50f15", lw=.9, ls=":")
                ax.set_title(dom, fontsize=11)
                ax.set_xlabel("Panel size n")
                ax.set_ylim(-1.0, .9)
            np.atleast_1d(axes)[0].set_ylabel(r"Krippendorff's $\alpha$")
            np.atleast_1d(axes)[-1].legend(fontsize=7, loc="lower right")
            fig.tight_layout()
            fig.savefig(OUT_DIR / "fig_02_resampling.png", dpi=200)
            print("saved: fig_02_resampling.png")
        except Exception as exc:      # 그림 실패가 분석을 막지 않도록
            print(f"[plot skipped] {exc}")

    return curve, place


if __name__ == "__main__":
    main()
