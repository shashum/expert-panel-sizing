"""11_simulations.py — 모의실험 1–5: 모형 정합성, 분리 조건, 강건성 변형, 감쇠 예측, 판정 정확도.

결과는 outputs/sim/에 저장된다. 원고 3.6절, 5장.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sim_core import *

OUT = str(OUT_DIR / "sim") + "/"
import os; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261102)
N_T = 16
res = {}


# ---------- vectorised metrics: arrays are (reps, n_targets, n_raters) ----------
def v_alpha(X):
    reps, n, m = X.shape
    N = n * m
    ss_w = ((X - X.mean(2, keepdims=True)) ** 2).sum((1, 2))
    ss_t = ((X - X.mean((1, 2), keepdims=True)) ** 2).sum((1, 2))
    return 1 - ss_w * (N - 1) / (n * (m - 1) * ss_t)


def v_icc_c1(X):
    reps, n, k = X.shape
    g = X.mean((1, 2), keepdims=True)
    ms_r = k * ((X.mean(2) - g[:, :, 0]) ** 2).sum(1) / (n - 1)
    resid = X - X.mean(2, keepdims=True) - X.mean(1, keepdims=True) + g
    ms_e = (resid ** 2).sum((1, 2)) / ((n - 1) * (k - 1))
    return (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)


def v_spearman(a, b):
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def v_pearson(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))


def gen(reps, n, k, vt, vr, ve, mu=3.5, rng=rng):
    """additive model: score = mu + target + rater severity + residual."""
    a = rng.normal(0, np.sqrt(vt), (reps, n, 1))
    r = rng.normal(0, np.sqrt(vr), (reps, 1, k))
    e = rng.normal(0, np.sqrt(ve), (reps, n, k))
    return mu + a + r + e


def split_rho(X, kA, kB, spearman=True):
    A, B = X[:, :, :kA].mean(2), X[:, :, kA:kA + kB].mean(2)
    return v_spearman(A, B) if spearman else v_pearson(A, B)


df = load()
OBS = {}
for dom in DOMS:
    M = matrix(df, dom)
    first = sorted(set(df[(df.Domain == dom) & (df.panel == "first")].Expert))
    rep = sorted(set(df[(df.Domain == dom) & (df.panel == "repeat")].Expert))
    X = M.values
    vt, vr, ve = varcomp(X)
    a = matrix(df, dom, set(first)).mean(1).values
    b = matrix(df, dom, set(rep)).mean(1).values
    from scipy.stats import spearmanr
    OBS[dom] = dict(X=X, M=M, kA=len(first), kB=len(rep), vt=vt, vr=vr, ve=ve,
                    alpha=kalpha(X), c1=icc(X)["C1"], rho=spearmanr(a, b)[0], mu=X.mean())

# ====================== SIM 1: posterior-predictive check ======================
REPS1 = 5000
rows = []
for dom in DOMS:
    o = OBS[dom]
    kA, kB = o["kA"], o["kB"]
    m = kA + kB
    # (a) plug-in
    X = gen(REPS1, N_T, m, o["vt"], o["vr"], o["ve"], mu=o["mu"])
    for label, XX in [("plug-in", X)]:
        al, c1, rho = v_alpha(XX), v_icc_c1(XX), split_rho(XX, kA, kB)
        rows.append((dom, label, al.mean(), c1.mean(), rho.mean(), np.percentile(rho, 5), np.percentile(rho, 95),
                     (rho < o["rho"]).mean() * 100, (rho >= o["rho"]).mean() * 100))
    # (b) parameter uncertainty: bootstrap targets -> re-estimate -> simulate one dataset each
    Xo = o["X"]
    out_al, out_c1, out_rho = [], [], []
    for _ in range(REPS1):
        idx = rng.integers(0, N_T, N_T)
        vt, vr, ve = varcomp(Xo[idx])
        ve = max(ve, 1e-6)
        Y = gen(1, N_T, m, vt, vr, ve, mu=o["mu"])
        out_al.append(v_alpha(Y)[0]); out_c1.append(v_icc_c1(Y)[0]); out_rho.append(split_rho(Y, kA, kB)[0])
    rho = np.array(out_rho)
    rows.append((dom, "bootstrap-param", np.mean(out_al), np.mean(out_c1), rho.mean(), np.percentile(rho, 5),
                 np.percentile(rho, 95), (rho < o["rho"]).mean() * 100, (rho >= o["rho"]).mean() * 100))
sim1 = pd.DataFrame(rows, columns=["domain", "version", "mean_alpha", "mean_ICC_C1", "mean_rho", "rho_p5", "rho_p95",
                                   "obs_rho_percentile", "P(sim_rho>=obs)"])
sim1["obs_alpha"] = sim1.domain.map({d: OBS[d]["alpha"] for d in DOMS})
sim1["obs_C1"] = sim1.domain.map({d: OBS[d]["c1"] for d in DOMS})
sim1["obs_rho"] = sim1.domain.map({d: OBS[d]["rho"] for d in DOMS})
sim1["vt"] = sim1.domain.map({d: OBS[d]["vt"] for d in DOMS})
sim1["vr"] = sim1.domain.map({d: OBS[d]["vr"] for d in DOMS})
sim1["ve"] = sim1.domain.map({d: OBS[d]["ve"] for d in DOMS})
sim1.to_csv(OUT + "sim1_ppc.csv", index=False)
print("=== SIM1 ===")
print(sim1.round(3).to_string())

# ====================== SIM 2: separation region (grid) ======================
VT, VE0, VR0 = np.mean([OBS[d]["vt"] for d in DOMS]), np.mean([OBS[d]["ve"] for d in DOMS]), np.mean([OBS[d]["vr"] for d in DOMS])
print("\npooled vt=%.4f ve=%.4f vr=%.4f  -> vr/vt=%.0f  ve/vt=%.1f" % (VT, VE0, VR0, VR0 / VT, VE0 / VT))
REPS2 = 3000
Rgrid = [0, 10, 30, 70, 150, 300]
Kgrid = [2, 3, 5, 8, 12, 20, 30]
Egrid = [5, 10, 15, 25, 50]
rowsA = []
for R in Rgrid:
    k = 5
    X = gen(REPS2, N_T, 2 * k, VT, R * VT, VE0)
    al, rho = v_alpha(X), split_rho(X, k, k)
    rowsA.append(dict(vr_over_vt=R, k=k, mean_alpha=al.mean(), mean_rho=rho.mean(), p_rho_ge_5=(rho >= .5).mean(),
                      p_sep=((al < .667) & (rho >= .5)).mean()))
simA = pd.DataFrame(rowsA)
rowsB = []
for k in Kgrid:
    for E in Egrid:
        X = gen(REPS2, N_T, 2 * k, VT, VR0, E * VT)
        al, rho = v_alpha(X), split_rho(X, k, k)
        rowsB.append(dict(k=k, ve_over_vt=E, mean_alpha=al.mean(), mean_rho=rho.mean(), p_rho_ge_5=(rho >= .5).mean(),
                          p_sep=((al < .667) & (rho >= .5)).mean()))
simB = pd.DataFrame(rowsB)
simA.to_csv(OUT + "sim2a_severity.csv", index=False)
simB.to_csv(OUT + "sim2b_grid.csv", index=False)
print("\n=== SIM2a (k=5/5, vary severity ratio) ===")
print(simA.round(3).to_string())
print("\n=== SIM2b (E[rho], P(rho>=.5)) at observed severity ===")
print(simB.pivot(index="k", columns="ve_over_vt", values="mean_rho").round(2).to_string())
print(simB.pivot(index="k", columns="ve_over_vt", values="p_rho_ge_5").round(2).to_string())

# ====================== SIM 3: robustness (non-additive severity etc.) ======================
def pool(dom):
    M = matrix(df, dom)
    X = M.values
    off = X.mean(0) - X.mean()
    sd = X.std(0, ddof=1)
    return off, sd / sd.mean()


def gen_variant(variant, dom, reps, k, rng=rng):
    o = OBS[dom]
    vt, vr, ve, mu = o["vt"], o["vr"], o["ve"], o["mu"]
    a = rng.normal(0, np.sqrt(vt), (reps, N_T, 1))
    e = rng.normal(0, np.sqrt(ve), (reps, N_T, k))
    if variant == "additive-normal":
        r = rng.normal(0, np.sqrt(vr), (reps, 1, k))
        return mu + a + r + e
    if variant == "scale-heterogeneity":      # lognormal multiplicative rater scale, mean 1
        r = rng.normal(0, np.sqrt(vr), (reps, 1, k))
        s = np.exp(rng.normal(-0.125, 0.5, (reps, 1, k)))
        return mu + r + s * (a + e)
    if variant == "clipped[1,5]":
        r = rng.normal(0, np.sqrt(vr), (reps, 1, k))
        return np.clip(mu + a + r + e, 1, 5)
    if variant == "empirical-raters":         # resample observed rater offsets & scales (incl. constant raters), clip
        off, sc = pool(dom)
        j = rng.integers(0, len(off), (reps, 1, k))
        return np.clip(mu + off[j] + sc[j] * (a + e), 1, 5)
    raise ValueError(variant)


rows3 = []
for dom in DOMS:
    o = OBS[dom]
    for variant in ["additive-normal", "scale-heterogeneity", "clipped[1,5]", "empirical-raters"]:
        kA, kB = o["kA"], o["kB"]
        X = gen_variant(variant, dom, 3000, kA + kB)
        al, c1, rho = v_alpha(X), v_icc_c1(X), split_rho(X, kA, kB)
        fin = np.isfinite(rho)
        rows3.append(dict(domain=dom, variant=variant, mean_alpha=al.mean(), mean_C1=c1.mean(), mean_rho=np.nanmean(rho),
                          p_rho_ge_5=(rho[fin] >= .5).mean(), obs_rho_pct=(rho[fin] < o["rho"]).mean() * 100,
                          share_undefined=1 - fin.mean()))
sim3 = pd.DataFrame(rows3)
sim3.to_csv(OUT + "sim3_robust.csv", index=False)
print("\n=== SIM3 ===")
print(sim3.round(3).to_string())

# ====================== SIM 4: attenuation formula & required panel size ======================
Ks = list(range(2, 46))
rows4 = []
for dom in DOMS:
    o = OBS[dom]
    r1 = o["vt"] / (o["vt"] + o["ve"])
    for k in Ks:
        X = gen(2000, N_T, 2 * k, o["vt"], o["vr"], o["ve"])
        rp, rs = split_rho(X, k, k, spearman=False), split_rho(X, k, k, spearman=True)
        Rk = k * r1 / (1 + (k - 1) * r1)
        rows4.append(dict(domain=dom, k=k, pred=Rk, sim_pearson=rp.mean(), sim_spearman=rs.mean()))
sim4 = pd.DataFrame(rows4)
sim4.to_csv(OUT + "sim4_attenuation.csv", index=False)
print("\n=== SIM4: predicted vs simulated at k=3,5,10,20,45 ===")
print(sim4[sim4.k.isin([3, 5, 10, 20, 45])].round(3).to_string())

need = []
for dom in DOMS:
    Xo = OBS[dom]["X"]
    ks = []
    for _ in range(5000):
        idx = rng.integers(0, N_T, N_T)
        e = icc(Xo[idx])["C1"]
        ks.append(np.inf if e <= 0 else max(np.ceil(0.7 / 0.3 * (1 - e) / e), 1))
    ks = np.array(ks)
    r1 = icc(Xo)["C1"]
    pt = np.ceil(0.7 / 0.3 * (1 - r1) / r1)
    fin = ks[np.isfinite(ks)]
    need.append(dict(domain=dom, point_k=pt, median_k=np.percentile(ks, 50, method="nearest"), p25=np.percentile(ks, 25, method="nearest"), p75=np.percentile(ks, 75, method="nearest"),
                     p90=np.percentile(ks, 90, method="nearest"), share_inf=np.isinf(ks).mean(), share_le_30=(ks <= 30).mean()))
need = pd.DataFrame(need)
need.to_csv(OUT + "sim4_required_k.csv", index=False)
print("\n=== required raters for ICC(C,k)>=.70, bootstrap over targets ===")
print(need.round(3).to_string())

# ====================== SIM 5: comparison vs baseline decisions ======================
def decision_metrics(k, reps=4000):
    a = rng.normal(0, np.sqrt(VT), (reps, N_T, 1))
    rA, rB = rng.normal(0, np.sqrt(VR0), (reps, 1, k)), rng.normal(0, np.sqrt(VR0), (reps, 1, k))
    eA, eB = rng.normal(0, np.sqrt(VE0), (reps, N_T, k)), rng.normal(0, np.sqrt(VE0), (reps, N_T, k))
    mu = 3.5
    PA = (mu + a + rA + eA).mean(2)
    PB = (mu + a + rB + eB).mean(2)
    # pairwise ordering agreement over all target pairs
    dA, dB = PA[:, :, None] - PA[:, None, :], PB[:, :, None] - PB[:, None, :]
    iu = np.triu_indices(N_T, 1)
    order = (np.sign(dA[:, iu[0], iu[1]]) == np.sign(dB[:, iu[0], iu[1]])).mean(1)
    # model-level top-1 agreement (4 models x 4 conditions)
    mA, mB = PA.reshape(reps, 4, 4).mean(2), PB.reshape(reps, 4, 4).mean(2)
    top1 = (mA.argmax(1) == mB.argmax(1))
    # baseline: absolute level of the grand mean within 0.5 pt, and identical pass/fail verdicts at cutoff = 3.5
    lvl = np.abs(mA.mean(1) - mB.mean(1)) < 0.5
    lvl25 = np.abs(mA.mean(1) - mB.mean(1)) < 0.25
    return dict(k=k, pairwise_order_agree=order.mean(), top1_model_agree=top1.mean(),
                level_within_0_5=lvl.mean(), level_within_0_25=lvl25.mean())


sim5 = pd.DataFrame([decision_metrics(k) for k in [3, 5, 10, 20, 30, 45, 100]])
sim5.to_csv(OUT + "sim5_decisions.csv", index=False)
print("\n=== SIM5 ===")
print(sim5.round(3).to_string())

json.dump(dict(VT=VT, VE=VE0, VR=VR0, obs={d: {k: float(v) for k, v in OBS[d].items() if k in ("kA", "kB", "vt", "vr", "ve", "alpha", "c1", "rho", "mu")} for d in DOMS}),
          open(OUT + "params.json", "w"), indent=1, ensure_ascii=False)
