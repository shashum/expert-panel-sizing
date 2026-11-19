"""
06_lme.py — 선형혼합효과모형 및 보조 회귀

관측 단위
    LME의 종속변수는 평가자 × 도메인 × 모델 × 시나리오 수준의 CQS이다.

수행 내용
    1) 주효과 모형: CQS ~ Model + Domain + (1 | Expert)
    2) 상호작용 모형: CQS ~ Model * Domain + (1 | Expert)
    3) 운영가치 모형: z_OperationalValue ~ z_Plaus + z_Tech + z_Cons
                      + (1 | Expert)
    4) 평가단위 수준 OLS 보조분석 (계수 순서 일치 확인)
    5) random effect 필요성 LR 검정, ICC
    6) 잔차 진단

statsmodels가 있으면 이를 사용하고, 없으면 내장 REML 구현으로 대체한다.
두 경로의 계수·표준오차는 일치한다.

실행
    python code/06_lme.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from common import OUT_DIR, DIMENSIONS, DATA_PATH, load_data

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAVE_SM = True
except ImportError:
    HAVE_SM = False


# ------------------------------------------------ 순수 numpy REML 대체

def fit_lmm(y, X, groups):
    """무선절편 선형혼합모형, 프로파일 REML.

    y      : (N,)
    X      : (N, p) 설계행렬 (절편 포함)
    groups : (N,) 군집 라벨
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    N, p = X.shape
    idx = [np.where(groups == g)[0] for g in pd.unique(groups)]

    def Wmul(A, theta):
        B = A.copy()
        for ii in idx:
            n_i = len(ii)
            B[ii] = A[ii] - (theta / (1 + n_i * theta)) * A[ii].sum(axis=0)
        return B

    def neg2reml(log_theta):
        theta = np.exp(log_theta)
        WX = Wmul(X, theta)
        Wy = Wmul(y[:, None], theta)[:, 0]
        XWX = X.T @ WX
        beta = np.linalg.solve(XWX, X.T @ Wy)
        r = y - X @ beta
        q = r @ Wmul(r[:, None], theta)[:, 0]
        s2 = q / (N - p)
        ld = sum(np.log(1 + len(ii) * theta) for ii in idx)
        _, ldx = np.linalg.slogdet(XWX)
        return ld + (N - p) * np.log(s2) + ldx

    opt = minimize_scalar(neg2reml, bounds=(-12, 12), method="bounded",
                          options={"xatol": 1e-9})
    theta = np.exp(opt.x)
    WX = Wmul(X, theta)
    XWX = X.T @ WX
    beta = np.linalg.solve(XWX, X.T @ Wmul(y[:, None], theta)[:, 0])
    resid = y - X @ beta
    q = resid @ Wmul(resid[:, None], theta)[:, 0]
    s2 = q / (N - p)
    se = np.sqrt(s2 * np.diag(np.linalg.inv(XWX)))
    return dict(beta=beta, se=se, sigma2=s2, tau2=theta * s2,
                icc=theta / (1 + theta), resid=resid, N=N, p=p)


def report(names, beta, se, title):
    print(f"\n--- {title}")
    print(f"{'term':38s} {'coef':>9s} {'SE':>8s} {'z':>7s} {'p':>9s} "
          f"{'[95% CI]':>20s}")
    for nm, b, s in zip(names, beta, se):
        z = b / s
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        print(f"{nm:38s} {b:+9.4f} {s:8.4f} {z:+7.2f} {p:9.4f} "
              f"[{b - 1.96 * s:+.4f}, {b + 1.96 * s:+.4f}]")


def design(df, cols, interaction=False):
    """더미 설계행렬 생성 (첫 수준 기준)."""
    X = pd.get_dummies(df[cols], drop_first=True).astype(float)
    if interaction:
        dom = [c for c in X.columns if c.startswith("Domain")]
        mod = [c for c in X.columns if c.startswith("M_")]
        for dcol in dom:
            for mcol in mod:
                X[f"{dcol}:{mcol}"] = X[dcol] * X[mcol]
    X.insert(0, "Intercept", 1.0)
    return X


def main(path=DATA_PATH):
    df = load_data(path)
    print(f"관측치 {len(df)}, 군집(평가자) {df.Expert.nunique()}")
    print(f"statsmodels 사용: {HAVE_SM}")

    # ------------------------------------------------ 1) 주효과
    X = design(df, ["M", "Domain"])
    m1 = fit_lmm(df.CQS.values, X.values, df.Expert.values)
    print("\n" + "=" * 88)
    print("1) 주효과 모형  CQS ~ Model + Domain + (1 | Expert)")
    print("=" * 88)
    print(f"  tau2(평가자) = {m1['tau2']:.4f}   sigma2(잔차) = "
          f"{m1['sigma2']:.4f}   ICC = {m1['icc']:.4f}")
    report(X.columns, m1["beta"], m1["se"], "고정효과")

    if HAVE_SM:
        sm1 = smf.mixedlm("CQS ~ C(M) + C(Domain)", df,
                          groups=df["Expert"]).fit(reml=True)
        print("\n[statsmodels 대조]")
        print(sm1.summary())

    # ------------------------------------------------ 2) 상호작용
    Xi = design(df, ["M", "Domain"], interaction=True)
    m2 = fit_lmm(df.CQS.values, Xi.values, df.Expert.values)
    print("\n" + "=" * 88)
    print("2) 상호작용 모형  CQS ~ Model * Domain + (1 | Expert)")
    print("=" * 88)
    print(f"  tau2 = {m2['tau2']:.4f}   sigma2 = {m2['sigma2']:.4f}   "
          f"ICC = {m2['icc']:.4f}")
    report(Xi.columns, m2["beta"], m2["se"], "고정효과")

    # ------------------------------------------------ 3) 운영가치
    z = lambda s: (s - s.mean()) / s.std(ddof=1)
    d3 = df.copy()
    for dim in DIMENSIONS:
        d3["z_" + dim] = z(d3[dim])
    preds = ["z_Plausibility", "z_TechnicalValidity", "z_Consistency"]
    X3 = d3[preds].astype(float).copy()
    X3.insert(0, "Intercept", 1.0)
    m3 = fit_lmm(d3["z_OperationalValue"].values, X3.values, d3.Expert.values)
    print("\n" + "=" * 88)
    print("3) 운영가치 모형 (표준화)")
    print("=" * 88)
    print(f"  tau2 = {m3['tau2']:.4f}   sigma2 = {m3['sigma2']:.4f}")
    report(X3.columns, m3["beta"], m3["se"], "고정효과")

    # 다중공선성 진단 (공통방법편의 검토)
    corr = d3[preds].corr()          # 이름을 C로 두면 수식의 C()와 충돌한다
    print("\n  예측변수 상관")
    print(corr.to_string(float_format=lambda x: f"{x:.3f}"))
    Z = d3[preds].values
    vif = []
    for i in range(Z.shape[1]):
        others = np.delete(Z, i, axis=1)
        A = np.column_stack([np.ones(len(Z)), others])
        b, *_ = np.linalg.lstsq(A, Z[:, i], rcond=None)
        r2 = 1 - ((Z[:, i] - A @ b).var() / Z[:, i].var())
        vif.append(1 / (1 - r2))
    print("  VIF: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(preds, vif)))

    # ------------------------------------------------ 4) OLS 보조
    U = (df.groupby(["Expert", "Domain", "M"])[list(DIMENSIONS)]
         .mean().reset_index())
    for dim in DIMENSIONS:
        U["z_" + dim] = z(U[dim])
    Xa = np.column_stack([np.ones(len(U))] + [U["z_" + p_].values
                                              for p_ in
                                              ["Plausibility",
                                               "TechnicalValidity",
                                               "Consistency"]])
    ya = U["z_OperationalValue"].values
    b, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
    resid = ya - Xa @ b
    r2 = 1 - resid.var() / ya.var()
    print("\n" + "=" * 88)
    print(f"4) 평가단위 수준 OLS 보조분석  N = {len(U)}")
    print("=" * 88)
    for nm, val in zip(["Intercept", "Plausibility", "TechnicalValidity",
                        "Consistency"], b):
        print(f"  {nm:20s} beta = {val:+.4f}")
    print(f"  R2 = {r2:.4f}")
    print("  LME와 계수 순서가 일치하는지 확인할 것")

    # ------------------------------------------------ 5) LR 검정
    Xo = X.values
    bo, *_ = np.linalg.lstsq(Xo, df.CQS.values, rcond=None)
    ro = df.CQS.values - Xo @ bo
    n = len(ro)
    ll_ols = -0.5 * n * (np.log(2 * np.pi * ro.var()) + 1)
    ll_lme = None
    if HAVE_SM:
        ll_lme = smf.mixedlm("CQS ~ C(M) + C(Domain)", df,
                             groups=df["Expert"]).fit(reml=False).llf
        ll_ols_ml = ll_ols
        lr = 2 * (ll_lme - ll_ols_ml)
        print("\n" + "=" * 88)
        print("5) random effect 필요성 (LR, 혼합 카이제곱 근사)")
        print("=" * 88)
        print(f"  LME loglik = {ll_lme:.2f}   OLS loglik = {ll_ols_ml:.2f}")
        print(f"  LR = {lr:.2f}, p = {0.5 * stats.chi2.sf(lr, 1):.3e}")
    else:
        print("\n5) LR 검정은 statsmodels 설치 시 수행됨 "
              "(!pip install statsmodels)")

    # ------------------------------------------------ 6) 잔차 진단
    print("\n" + "=" * 88)
    print("6) 잔차 진단 (주효과 모형, 주변잔차 = 무선효과 미차감)")
    print("=" * 88)
    r = m1["resid"]
    sw = stats.shapiro(r) if len(r) <= 5000 else None
    print(f"  평균 {r.mean():+.4f}  SD {r.std(ddof=1):.4f}  "
          f"왜도 {stats.skew(r):+.3f}  첨도 {stats.kurtosis(r):+.3f}")
    if sw is not None:
        print(f"  Shapiro-Wilk W = {sw.statistic:.4f}, p = {sw.pvalue:.4e}")
    print("  5점 리커트를 연속으로 다룬 근거를 본문에 명시할 것 "
          "(서열 모형 병행 권장)")

    out = pd.DataFrame({"term": list(Xi.columns),
                        "coef": m2["beta"], "se": m2["se"]})
    out["z"] = out.coef / out.se
    out["p"] = 2 * (1 - stats.norm.cdf(out.z.abs()))
    out.to_csv(OUT_DIR / "out_06_lme_interaction.csv", index=False)
    print("\nsaved: out_06_lme_interaction.csv")
    return m1, m2, m3


if __name__ == "__main__":
    main()
