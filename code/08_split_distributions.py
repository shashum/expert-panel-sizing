"""
08_split_distributions.py — 실제 패널 규모 분할의 전수 열거와 보조 계산

  - 실제 패널 규모(통신 6/4, 에너지 6/3, 보건의료 3/4)로 평가자를 나누는 모든 분할의 패널 간 순위상관
    (원고 3.5.3절, 4.4절, 표 7, 그림 2)
  - 4개 모델 순위의 완전일치 비율, 분할반복 평균을 예측하는 역산 r1(4.5절)
  - 기준선/비교 판정 소요 평가자 비의 95% 구간(F 분포, 5.4절)
  - 평가자 내 모델 쌍체 차이와 Holm 보정(표 6 주석)
  - 평가자 내 차원 대비와 Holm 보정(4.1절), 복수 도메인 평가자의 도메인 간 대비(표 2 주석)

실행
    python code/08_split_distributions.py
"""
import itertools, json, warnings
import numpy as np
from scipy import stats, optimize
from common import OUT_DIR, DOMAINS, load_data, experts_in
warnings.filterwarnings("ignore")
df = load_data(); out = {}
BINS = np.round(np.arange(-0.6, 1.0001, 0.1), 1)

def anova(X):
    n, k = X.shape; gm = X.mean()
    mst = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msr = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return mst, msr, mse
def sb(r1, k): return k * r1 / (1 + (k - 1) * r1)
def pred(r1, a, b): return np.sqrt(sb(r1, a) * sb(r1, b))

for dom in DOMAINS:
    d = df[df.Domain == dom]; ex = experts_in(df, dom); N = len(ex)
    U16 = d.pivot_table(index=["M", "Scenario"], columns="Expert", values="CQS")[ex].values
    U4 = d.pivot_table(index="M", columns="Expert", values="CQS")[ex].values
    der = [i for i, e in enumerate(ex) if d.loc[d.Expert == e, "Cohort"].iloc[0] == "derivation"]
    val = [i for i in range(N) if i not in der]; kA, kB = len(der), len(val)
    obs = stats.spearmanr(U16[:, der].mean(1), U16[:, val].mean(1)).statistic
    r16, ident = [], []
    for A in itertools.combinations(range(N), kA):
        B = [i for i in range(N) if i not in A]
        r16.append(stats.spearmanr(U16[:, list(A)].mean(1), U16[:, B].mean(1)).statistic)
        ident.append(stats.spearmanr(U4[:, list(A)].mean(1), U4[:, B].mean(1)).statistic == 1.0)
    r = np.array(r16); m = float(np.nanmean(r))
    mst, msr, mse = anova(U16); n = U16.shape[0]
    vt, vr, ve = (mst - mse) / N, (msr - mse) / n, mse; r1 = vt / (vt + ve)
    inv = optimize.brentq(lambda x: pred(x, kA, kB) - m, 1e-6, 0.99)
    # exact interval for theta = vr/ve under normality: F = msr/mse ~ (1 + n*theta) * F(k-1, (n-1)(k-1))
    F = msr / mse; df1, df2 = N - 1, (n - 1) * (N - 1)
    th_lo = (F / stats.f.ppf(0.975, df1, df2) - 1) / n; th_hi = (F / stats.f.ppf(0.025, df1, df2) - 1) / n
    out[dom] = dict(kA=kA, kB=kB, n_splits=len(r), n_nan=int(np.isnan(r).sum()), mean=m, lo=float(np.nanpercentile(r, 2.5)), hi=float(np.nanpercentile(r, 97.5)),
                    obs=float(obs), obs_pct=float(np.nanmean(r <= obs + 1e-12) * 100), rank_from_top=int((r > obs + 1e-12).sum()) + 1,
                    p_identical=float(np.mean(ident)), n_identical=int(np.sum(ident)), obs_identical=bool(stats.spearmanr(U4[:, der].mean(1), U4[:, val].mean(1)).statistic == 1.0),
                    r1=float(r1), pred=float(pred(r1, kA, kB)), inv_r1_mean=float(inv),
                    hist=(np.histogram(r[~np.isnan(r)], bins=BINS)[0] / (~np.isnan(r)).sum()).tolist(),
                    ratio=float((vr + ve) / (2 * ve)), ratio_lo=float((th_lo + 1) / 2), ratio_hi=float((th_hi + 1) / 2), theta=float(vr / ve))
    print(dom, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[dom].items() if k != "hist"})
    # paired model differences (rater-level model means)
    M = d.groupby(["Expert", "M"]).CQS.mean().unstack(); pm = []
    for a, b in itertools.combinations(M.columns, 2):
        diff = M[a] - M[b]; t, p = stats.ttest_1samp(diff, 0)
        pm.append(dict(pair=f"{a}-{b}", d=float(diff.mean()), se=float(diff.std(ddof=1) / np.sqrt(len(diff))), p=float(p)))
    out[dom]["paired_models"] = pm
allp = sorted((x["p"], dom, x["pair"], x["d"], x["se"]) for dom in DOMAINS for x in out[dom]["paired_models"])
print("paired: n=%d  SE range %.3f-%.3f" % (len(allp), min(x[4] for x in allp), max(x[4] for x in allp)))
for i, x in enumerate(allp[:4]): print("  ", x, "holm=%.3f" % min(1, x[0] * (len(allp) - i)))
dims = {"Realism": "Plausibility", "Accuracy": "TechnicalValidity", "Coherence": "Consistency", "Utility": "OperationalValue"}
R = df.groupby("Expert")[list(dims.values())].mean(); dc = {}
for name, col in dims.items():
    if name == "Accuracy": continue
    dd = R["TechnicalValidity"] - R[col]; t, p = stats.ttest_1samp(dd, 0)
    dc[name] = dict(d=float(dd.mean()), t=float(t), p=float(p), n_lower=int((dd < 0).sum()))
print("Accuracy minus:", {k: {a: round(b, 4) for a, b in v.items()} for k, v in dc.items()})
ps = sorted(dc, key=lambda k: dc[k]["p"])
for i, k in enumerate(ps): dc[k]["p_holm"] = float(min(1, max(dc[q]["p"] * (len(ps) - j) for j, q in enumerate(ps[:i + 1]))))
print("Holm-adjusted:", {k: round(dc[k]["p_holm"], 4) for k in ps})
out["_dimension_contrasts"] = dc
ce = df[df.Domain.isin(["Communication", "Energy"])].groupby(["Expert", "Domain"]).CQS.mean().unstack().dropna()
out["_energy_minus_comm_within_rater"] = (ce["Energy"] - ce["Communication"]).round(3).tolist(); print("Energy-Comm within rater:", out["_energy_minus_comm_within_rater"])
print("rater-level grand mean: %.3f" % df.groupby("Expert").CQS.mean().mean())
json.dump(dict(bins=BINS.tolist(), dom={k: v for k, v in out.items() if not k.startswith("_")}, extra={k: v for k, v in out.items() if k.startswith("_")}), open(OUT_DIR / "out_08_split_distributions.json", "w"), indent=1)
