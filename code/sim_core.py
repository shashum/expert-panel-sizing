"""sim_core.py — 모의실험용 공통 함수: 자료 로딩, 신뢰도 지표, 분산성분(완전 이원 설계)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DATA_PATH, DERIVATION, OUT_DIR, SHEET

XLSX = DATA_PATH
Q = [f"Q{i}" for i in range(1, 9)]
DOMS = ["Communication", "Energy", "Healthcare"]
FIRST = set(DERIVATION)   # 최초 패널


def load():
    df = pd.read_excel(XLSX, sheet_name=SHEET)
    df["score"] = df[Q].mean(axis=1)
    df["target"] = df["Model"] + "|" + df["Scenario"]
    df["panel"] = np.where(df["Expert"].isin(FIRST), "first", "repeat")
    return df


def matrix(df, dom, experts=None):
    """targets x raters matrix of 8-item means for one domain."""
    d = df[df.Domain == dom]
    if experts is not None:
        d = d[d.Expert.isin(experts)]
    m = d.pivot(index="target", columns="Expert", values="score").sort_index()
    return m


def anova(X):
    """Two-way ANOVA mean squares for an n x k matrix (targets x raters)."""
    n, k = X.shape
    g = X.mean()
    ms_r = k * ((X.mean(1) - g) ** 2).sum() / (n - 1)
    ms_c = n * ((X.mean(0) - g) ** 2).sum() / (k - 1)
    res = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + g
    ms_e = (res ** 2).sum() / ((n - 1) * (k - 1))
    return ms_r, ms_c, ms_e


def icc(X):
    n, k = X.shape
    r, c, e = anova(X)
    return dict(
        C1=(r - e) / (r + (k - 1) * e),
        Ck=(r - e) / r,
        A1=(r - e) / (r + (k - 1) * e + k * (c - e) / n),
        Ak=(r - e) / (r + (c - e) / n),
    )


def kalpha(X):
    """Krippendorff alpha, interval metric, complete data (targets x raters)."""
    n, m = X.shape
    N = n * m
    ss_w = ((X - X.mean(1, keepdims=True)) ** 2).sum()
    ss_t = ((X - X.mean()) ** 2).sum()
    return 1 - ss_w * (N - 1) / (n * (m - 1) * ss_t)


def varcomp(X):
    """ANOVA variance components: target, rater, residual."""
    n, k = X.shape
    r, c, e = anova(X)
    return max((r - e) / k, 0.0), max((c - e) / n, 0.0), e
