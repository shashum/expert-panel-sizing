"""12_sim_additions.py — 모의실험 보조 분석과 그림용 수치(outputs/sim_numbers.json)

11_simulations.py의 출력(outputs/sim/)을 읽는다. 실험별로 독립 시드를 쓴다.
  A  부분집합 전수 열거와 순열 영분포 (4.3절, 그림 1)
  C  둘째 예측 통계량: 실제 패널 규모의 전 분할 평균 rho와 Fisher 결합 (5.1절, 표 10)
  D  공통 난수 실험과 alpha의 닫힌 형태 (5.2절, 그림 3(a))
  E  Moran(1948) 기댓값과 모의값 대조 (5.2절, 5.3절, 그림 3(b))
  F  판정 형태별 정확도 (5.4절, 그림 4, 표 12)
  G  모형 가정 진단: 잔차 첨도, 평가자별 잔차 분산, 척도 이질성 (3.6.3절, 5.1절)
분할반복 분포(그림 2)는 08_split_distributions.py가 계산한다.

실행
    python code/12_sim_additions.py
"""
import argparse, itertools, json, math, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, chi2

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None)
ap.add_argument("--simdir", default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()
import sim_core
if args.data: sim_core.XLSX = args.data
from sim_core import load, matrix, icc, kalpha, varcomp, DOMS, FIRST, OUT_DIR
args.simdir = args.simdir or str(OUT_DIR / "sim")
args.out = args.out or str(OUT_DIR / "sim_numbers.json")

SS = np.random.SeedSequence(20261113)
seeds = dict(zip("ABCDEF", SS.spawn(6)))
N_T = 16
df = load()
OUT = {}


def v_alpha(X):                      # X: (reps, n, m)
    r, n, m = X.shape
    N = n * m
    ss_w = ((X - X.mean(2, keepdims=True)) ** 2).sum((1, 2))
    ss_t = ((X - X.mean((1, 2), keepdims=True)) ** 2).sum((1, 2))
    return 1 - ss_w * (N - 1) / (n * (m - 1) * ss_t)


def v_spear(a, b):
    ra, rb = rankdata(a, axis=-1), rankdata(b, axis=-1)
    ra = ra - ra.mean(-1, keepdims=True)
    rb = rb - rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))


def sb(r1, k):
    return k * r1 / (1 + (k - 1) * r1)


def moran(r, n=N_T):
    """E[Spearman] for a bivariate-normal sample of size n with correlation r."""
    return 6 / (math.pi * (n + 1)) * (math.asin(r) + (n - 2) * math.asin(r / 2))


def alpha_cf(vt, vr, ve, n, m):
    return 1 - (n * m - 1) * (vr + ve) / (m * (n - 1) * vt + n * (m - 1) * vr + (n * m - 1) * ve)


VC = {d: varcomp(matrix(df, d).values) for d in DOMS}
VT, VR, VE = (float(np.mean([VC[d][i] for d in DOMS])) for i in range(3))
OUT["pooled"] = dict(vt=VT, vr=VR, ve=VE)
OUT["domain_vc"] = {d: dict(vt=VC[d][0], vr=VC[d][1], ve=VC[d][2], ratio=(VC[d][1] + VC[d][2]) / (2 * VC[d][2])) for d in DOMS}

# ---------------------------------------------------------------- A
rng = np.random.default_rng(seeds["A"])
A = {}
for dom in DOMS:
    X = matrix(df, dom).values
    m = X.shape[1]
    perm = np.stack([np.stack([rng.permutation(X[:, j]) for j in range(m)], 1) for _ in range(2000)])  # (2000,16,m)
    rows = []
    for n in range(2, m + 1):
        combos = [list(c) for c in itertools.combinations(range(m), n)]
        al = np.array([kalpha(X[:, c]) for c in combos])
        ic = [icc(X[:, c]) for c in combos]
        nul = np.stack([v_alpha(perm[:, :, c]) for c in combos], 1)        # (2000, n_subsets)
        nul_mean = nul.mean(1)
        rows.append(dict(n=n, n_sub=len(combos), a_mean=al.mean(), a_lo=np.percentile(al, 2.5), a_hi=np.percentile(al, 97.5),
                         a_max=al.max(), null_mean=nul.mean(), null_lo=np.percentile(nul, 2.5), null_hi=np.percentile(nul, 97.5),
                         obs_pct_in_null=float((nul_mean < al.mean()).mean() * 100),
                         c1=np.mean([i["C1"] for i in ic]), ck=np.mean([i["Ck"] for i in ic]),
                         sb=sb(icc(X)["C1"], n), floor=-(N_T - 1) / (N_T * (n - 1))))
    A[dom] = rows
OUT["A_subsets"] = A


# ---------------------------------------------------------------- C
rng = np.random.default_rng(seeds["C"])
C = {}
for dom in DOMS:
    M = matrix(df, dom)
    X = M.values
    m = X.shape[1]
    kA = sum(e in FIRST for e in M.columns)
    combos = [list(c) for c in itertools.combinations(range(m), kA)]
    others = [[j for j in range(m) if j not in c] for c in combos]

    def T(Y):
        return np.nanmean(np.array([v_spear(Y[:, :, c].mean(2), Y[:, :, o].mean(2)) for c, o in zip(combos, others)]), 0)

    t_obs = float(T(X[None])[0])
    t_obs_pearson = float(np.mean([np.corrcoef(X[:, c].mean(1), X[:, o].mean(1))[0, 1] for c, o in zip(combos, others)]))
    vt, vr, ve = VC[dom]
    reps = 5000
    Y = rng.normal(0, math.sqrt(vt), (reps, N_T, 1)) + rng.normal(0, math.sqrt(ve), (reps, N_T, m))
    t_sim = T(Y)
    C[dom] = dict(kA=kA, kB=m - kA, n_splits=len(combos), t_obs=t_obs, t_obs_pearson=t_obs_pearson, t_model_mean=float(t_sim.mean()),
                  t_model_sd=float(t_sim.std()), obs_pct=float((t_sim < t_obs).mean() * 100))
OUT["C_second_statistic"] = C
p1 = [0.0840, 0.0394, 0.6284]      # Sim1 plug-in: P(sim rho >= obs)
st = -2 * np.sum(np.log(p1))
OUT["C_fisher_plugin"] = dict(chi2=float(st), df=6, p=float(chi2.sf(st, 6)))

# ---------------------------------------------------------------- D
rng = np.random.default_rng(seeds["D"])
reps, k = 5000, 5
a = rng.normal(0, math.sqrt(VT), (reps, N_T, 1))
e = rng.normal(0, math.sqrt(VE), (reps, N_T, 2 * k))
z = rng.normal(0, 1, (reps, 1, 2 * k))
D = []
for R in [0, 5, 10, 20, 30, 50, 70, 100, 150, 200, 300]:
    X = 3.5 + a + math.sqrt(R * VT) * z + e
    rho = v_spear(X[:, :, :k].mean(2), X[:, :, k:].mean(2))
    D.append(dict(R=R, alpha_sim=float(v_alpha(X).mean()), alpha_cf=alpha_cf(VT, R * VT, VE, N_T, 2 * k),
                  rho_mean=float(rho.mean()), p_rho_ge_5=float((rho >= 0.5).mean())))
OUT["D_sim2a_crn"] = dict(k_per_panel=k, raters_for_alpha=2 * k, ve_over_vt=VE / VT, rows=D,
                          alpha_floor=-(N_T - 1) / (N_T * (2 * k - 1)), moran_pred=moran(sb(VT / (VT + VE), k)))

# ---------------------------------------------------------------- E
s2b = pd.read_csv(f"{args.simdir}/sim2b_grid.csv")
E = []
for Ev in sorted(s2b.ve_over_vt.unique()):
    r1 = 1 / (1 + Ev)
    curve = [dict(k=kk, pearson=sb(r1, kk), spearman=moran(sb(r1, kk))) for kk in range(2, 31)]
    pts = s2b[s2b.ve_over_vt == Ev][["k", "mean_rho", "p_rho_ge_5"]].to_dict("records")
    E.append(dict(ve_over_vt=float(Ev), r1=r1, curve=curve, sim=pts))
s4 = pd.read_csv(f"{args.simdir}/sim4_attenuation.csv")
s4["moran"] = s4.pred.map(moran)
OUT["E_curves"] = E
OUT["E_sim4_vs_moran"] = dict(mean_diff=float((s4.sim_spearman - s4.moran).mean()), max_abs=float((s4.sim_spearman - s4.moran).abs().max()),
                              ratio_spearman_to_pred=[float((s4.sim_spearman / s4.pred).min()), float((s4.sim_spearman / s4.pred).max())],
                              ratio_moran_to_pred=[float((s4.moran / s4.pred).min()), float((s4.moran / s4.pred).max())])
OUT["E_table8_spearman"] = {d: moran(r) for d, r in zip(DOMS, [0.290, 0.246, 0.159])}

# ---------------------------------------------------------------- F
rng = np.random.default_rng(seeds["F"])
offs, scs = [], []
for d in DOMS:
    X = matrix(df, d).values
    sd = X.std(0, ddof=1)
    offs += list(X.mean(0) - X.mean())
    scs += list(sd / sd.mean())
offs, scs = np.array(offs), np.array(scs)


def p_compare(D, k, ve=VE):
    return float(norm.cdf(D / math.sqrt(2 * ve / k)))


def p_threshold(D, k, vr=VR, ve=VE):
    return float(norm.cdf(D / math.sqrt((vr + ve) / k)))


def sim_dec(D, k, kind, variant, reps=40000):
    t = np.array([D / 2, -D / 2])[None, :, None] if kind == "compare" else np.array([D])[None, :, None]
    e_ = rng.normal(0, math.sqrt(VE), (reps, t.shape[1], k))
    if variant == "normal":
        Y = 3.5 + t + rng.normal(0, math.sqrt(VR), (reps, 1, k)) + e_
    else:                                     # resample observed rater offsets and scales, clip to the 1-5 range
        j = rng.integers(0, len(offs), (reps, 1, k))
        Y = np.clip(3.5 + offs[j] + scs[j] * (t + e_), 1, 5)
    P = Y.mean(2)
    if kind == "compare":
        return float((P[:, 0] > P[:, 1]).mean() + 0.5 * (P[:, 0] == P[:, 1]).mean())
    return float((P[:, 0] > 3.5).mean())


F_rows = []
for D_ in [0.11, 0.25, 0.50]:
    for k_ in [3, 5, 10, 20, 30]:
        F_rows.append(dict(delta=D_, k=k_, cmp_cf=p_compare(D_, k_), cmp_norm=sim_dec(D_, k_, "compare", "normal"),
                           cmp_emp=sim_dec(D_, k_, "compare", "emp"), thr_cf=p_threshold(D_, k_),
                           thr_norm=sim_dec(D_, k_, "thr", "normal"), thr_emp=sim_dec(D_, k_, "thr", "emp")))
need = []
for D_ in [0.11, 0.25, 0.50]:
    row = dict(delta=D_)
    for p in [0.80, 0.90]:
        zz = norm.ppf(p)
        row[f"cmp_{int(p*100)}"] = math.ceil(zz * zz * 2 * VE / D_ ** 2)
        row[f"thr_{int(p*100)}"] = math.ceil(zz * zz * (VR + VE) / D_ ** 2)
    need.append(row)
curves = [dict(k=kk, cmp_25=p_compare(.25, kk), thr_25=p_threshold(.25, kk), cmp_50=p_compare(.5, kk), thr_50=p_threshold(.5, kk))
          for kk in range(1, 41)]
agree = [dict(k=kk, order=0.5 + math.asin(VT / (VT + VE / kk)) / math.pi, verdict=0.5 + math.asin(VT / (VT + (VR + VE) / kk)) / math.pi)
         for kk in [3, 5, 10, 20, 30, 45, 100]]
OUT["F_decisions"] = dict(rows=F_rows, need=need, curves=curves, ratio_pooled=(VR + VE) / (2 * VE), sd_targets=math.sqrt(VT),
                          two_panel_agreement=agree)

# ---------------------------------------------------------------- G
from scipy.stats import kurtosis
G = {}
for dom in DOMS:
    X = matrix(df, dom).values
    res = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()
    sd = X.std(0, ddof=1)
    rv = res.var(0, ddof=1)
    G[dom] = dict(resid_excess_kurtosis=float(kurtosis(res.ravel())), resid_var_ratio=float(rv[rv > 0].max() / rv[rv > 0].min()),
                  log_sd_of_rater_sd=float(np.log(sd[sd > 0]).std(ddof=1)), share_at_ceiling=float((X == 5).mean()))
OUT["G_diagnostics"] = G

json.dump(OUT, open(args.out, "w"), indent=1, ensure_ascii=False, default=float)

# ---------------------------------------------------------------- console summary
print("pooled vt=%.4f vr=%.4f ve=%.4f  ratio=%.2f  sd_t=%.3f" % (VT, VR, VE, (VR + VE) / (2 * VE), math.sqrt(VT)))
for d in DOMS:
    print(" ", d, {k: round(v, 4) for k, v in OUT["domain_vc"][d].items()})
print("\nA  subsets (n, alpha mean [lo,hi], null mean [lo,hi], pct of obs mean in null, C1, Ck, SB, floor)")
for d in DOMS:
    for r in A[d]:
        print("  %-13s n=%2d a=%+.3f [%+.3f,%+.3f] null=%+.3f [%+.3f,%+.3f] pct=%5.1f C1=%.3f Ck=%+.3f SB=%.3f floor=%+.3f"
              % (d, r["n"], r["a_mean"], r["a_lo"], r["a_hi"], r["null_mean"], r["null_lo"], r["null_hi"], r["obs_pct_in_null"], r["c1"], r["ck"], r["sb"], r["floor"]))
print("\nC  second statistic", json.dumps(C, indent=1, default=float), OUT["C_fisher_plugin"])
print("\nG ", json.dumps(G, default=float))
print("\nD  Sim2a CRN")
for r in D:
    print("  R=%3d alpha_sim=%+.4f alpha_cf=%+.4f rho=%.6f P(rho>=.5)=%.4f" % (r["R"], r["alpha_sim"], r["alpha_cf"], r["rho_mean"], r["p_rho_ge_5"]))
print("  floor=%.4f moran_pred=%.4f" % (OUT["D_sim2a_crn"]["alpha_floor"], OUT["D_sim2a_crn"]["moran_pred"]))
print("\nE ", OUT["E_sim4_vs_moran"], OUT["E_table8_spearman"])
for c in E:
    print("  ve/vt=%g" % c["ve_over_vt"], [(p["k"], round(p["mean_rho"], 3), round(moran(sb(c["r1"], p["k"])), 3)) for p in c["sim"]])
print("\nF  decisions")
for r in F_rows:
    print("  D=%.2f k=%2d compare %.3f/%.3f/%.3f threshold %.3f/%.3f/%.3f" % (r["delta"], r["k"], r["cmp_cf"], r["cmp_norm"], r["cmp_emp"], r["thr_cf"], r["thr_norm"], r["thr_emp"]))
print(" ", need)
print(" ", [(a_["k"], round(a_["order"], 3), round(a_["verdict"], 3)) for a_ in agree])
