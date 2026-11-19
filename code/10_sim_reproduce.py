"""10_sim_reproduce.py — 모의실험 모수의 출처 확인: 도메인별 신뢰도 지표, 분산성분, 패널 간 순위상관."""
import numpy as np
from scipy.stats import spearmanr
from sim_core import *

df = load()
print("overall mean", round(df.score.mean(), 3), "SD", round(df.score.std(), 3), "n", len(df))
print(df.groupby("Domain").score.agg(["mean", "std", "count"]).round(3))
print("panel sizes per domain:")
print(df.groupby(["Domain", "panel"]).Expert.nunique().unstack())

print("\n== reliability (full panel) ==")
VC = {}
for dom in DOMS:
    X = matrix(df, dom).values
    ic = icc(X)
    vt, vr, ve = varcomp(X)
    VC[dom] = (vt, vr, ve)
    print(dom, X.shape, "alpha=%.3f" % kalpha(X),
          " ".join(f"{k}={v:.3f}" for k, v in ic.items()),
          "| var target=%.3f rater=%.3f resid=%.3f" % (vt, vr, ve))

print("\n== cross-panel rho (16 targets) ==")
for dom in DOMS:
    a = matrix(df, dom, set(df[(df.Domain == dom) & (df.panel == "first")].Expert)).mean(1)
    b = matrix(df, dom, set(df[(df.Domain == dom) & (df.panel == "repeat")].Expert)).mean(1)
    r, p = spearmanr(a, b)
    print(dom, "rho=%.3f p=%.3f" % (r, p))
