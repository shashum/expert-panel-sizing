"""
09_supplementary_checks.py — 원고의 보조 검토

  1) 평가자 위치·척도 제거에 따른 α의 변화 (원고 표 4)
  2) 16단위 패널 간 순위상관의 순열검정: 평가자 내 대상 라벨 치환 10,000회 (표 7, 표 9)
  3) 패널 분할의 교환 가능성: 평가자 내 중심화 후 프로파일 상관, 패널 내 쌍 대 패널 간 쌍 (4.5절)
  4) 공유 평가자 제외: 통신·에너지를 함께 평가한 평가자를 뺀 순위상관과 동일 규모 부분패널 분포 (4.5절)
  5) 복수 도메인 평가자의 조건 프로파일의 도메인 간 상관 (4.5절)
  6) 도메인 간 재현성 차이의 부트스트랩: 평가 대상 16개 복원추출 5,000회 (4.5절)
  7) 평가자 배제 조건별 지표 (4.6절, 표 9)

실행
    python code/09_supplementary_checks.py
"""
import itertools
import json
import warnings

import numpy as np
from scipy import stats

from common import (
    OUT_DIR, DOMAINS, ITEMS, load_data, experts_in, rating_matrix,
    krippendorff_alpha_interval, icc_all,
)

warnings.filterwarnings("ignore")
N_PERM, N_BOOT = 10_000, 5_000
SS = np.random.SeedSequence(20261119)
SEED = dict(zip(["perm", "boot"], SS.spawn(2)))
df = load_data()
out = {}


def mat(dom, experts=None):
    """평가자 × 평가 대상 16개 행렬(값은 8문항 평균)."""
    return rating_matrix(df, dom, experts).values


def panels(dom, drop=()):
    ex = [e for e in experts_in(df, dom) if e not in drop]
    first = [e for e in ex if df.loc[df.Expert == e, "Cohort"].iloc[0] == "derivation"]
    return first, [e for e in ex if e not in first]


def rho(A, B):
    """두 패널(평가자 × 대상)의 대상별 평균 간 Spearman 상관."""
    return stats.spearmanr(A.mean(0), B.mean(0)).statistic


def perm_p(A, B, rng, n=N_PERM, pearson=False):
    """평가자마다 16개 대상의 라벨을 독립적으로 치환하는 양측 순열 유의확률."""
    obs = rho(A, B)
    X = np.vstack([A, B]); kA = len(A)
    idx = np.argsort(rng.random((n, X.shape[0], X.shape[1])), axis=2)
    P = np.take_along_axis(np.broadcast_to(X, idx.shape), idx, axis=2)
    a, b = P[:, :kA].mean(1), P[:, kA:].mean(1)
    def corr(x, y):
        x = x - x.mean(1, keepdims=True); y = y - y.mean(1, keepdims=True)
        return (x * y).sum(1) / np.sqrt((x ** 2).sum(1) * (y ** 2).sum(1))
    r = corr(stats.rankdata(a, axis=1), stats.rankdata(b, axis=1))
    if not pearson:
        return float(obs), float((np.abs(r) >= abs(obs) - 1e-12).mean())
    obs_p = stats.pearsonr(A.mean(0), B.mean(0)).statistic
    rp = corr(a, b)                    # 같은 치환에서 계산한 Pearson 상관
    return (float(obs), float((np.abs(r) >= abs(obs) - 1e-12).mean()),
            float(obs_p), float((np.abs(rp) >= abs(obs_p) - 1e-12).mean()))


# ---------------------------------------------------------------- 1) 표 4
# 평가자의 평균과 표준편차는 그 평가자의 전체 응답(모든 도메인)에서 구한다.
# 응답 분산이 0인 평가자는 z정규화 값을 정의할 수 없으므로 결측으로 둔다(α는 결측을 허용한다).
rs = df.groupby("Expert").CQS.agg(["mean", "std"])
dz = df.copy()
mu, sd = dz.Expert.map(rs["mean"]).values, dz.Expert.map(rs["std"]).values
dz["centered"] = dz.CQS - mu
dz["z"] = np.where(sd > 0, (dz.CQS - mu) / np.where(sd > 0, sd, 1), np.nan)
out["table4"] = {}
for label in ["first", "full"]:
    for dom in DOMAINS:
        first, _ = panels(dom)
        ex = first if label == "first" else experts_in(df, dom)
        a = {v: float(krippendorff_alpha_interval(rating_matrix(dz, dom, ex, value=v).values)) for v in ["CQS", "centered", "z"]}
        out["table4"][f"{label}|{dom}"] = dict(raw=a["CQS"], centered=a["centered"], z=a["z"],
                                              share_location=(a["centered"] - a["CQS"]) / (a["z"] - a["CQS"]))

# ---------------------------------------------------------------- 2) 순열검정 (표 7)
rng = np.random.default_rng(SEED["perm"])
out["perm_rho"] = {}
for dom in DOMAINS:
    f, r = panels(dom)
    out["perm_rho"][dom] = dict(zip(["rho", "p", "pearson_r", "pearson_p"], perm_p(mat(dom, f), mat(dom, r), rng, pearson=True)))

# ---------------------------------------------------------------- 3) 교환 가능성
out["exchangeability"] = {}
for dom in DOMAINS:
    f, r = panels(dom)
    ex = f + r
    X = mat(dom, ex); C = X - X.mean(1, keepdims=True)
    ok = C.std(1) > 0
    if not ok.all():
        out["exchangeability"][dom] = "not computed: a rater has zero variance"
        continue
    R = np.corrcoef(C)
    within, between = [], []
    for i, j in itertools.combinations(range(len(ex)), 2):
        (within if (ex[i] in f) == (ex[j] in f) else between).append(R[i, j])
    out["exchangeability"][dom] = dict(within=float(np.mean(within)), between=float(np.mean(between)))

# ---------------------------------------------------------------- 4) 공유 평가자 제외
shared = sorted(set(experts_in(df, "Communication")) & set(experts_in(df, "Energy")), key=lambda e: int(e[1:]))
out["shared_raters_comm_energy"] = len(shared)
out["shared_exclusion"] = {}
for dom in ["Communication", "Energy"]:
    f, r = panels(dom)
    f2, r2 = panels(dom, drop=shared)
    obs = rho(mat(dom, f2), mat(dom, r2))
    vals = [rho(mat(dom, list(a)), mat(dom, list(b))) for a in itertools.combinations(f, len(f2)) for b in itertools.combinations(r, len(r2))]
    vals = np.array(vals)
    # 백분위: 관측값보다 작은 부분패널의 비율(관측 조합 자신은 세지 않음)
    out["shared_exclusion"][dom] = dict(k=f"{len(f2)}/{len(r2)}", rho=float(obs), n_subpanels=len(vals), pct=float((vals < obs - 1e-12).mean() * 100))

# ---------------------------------------------------------------- 5) 복수 도메인 평가자의 프로파일
# (a) 조건 프로파일: 4개 조건(구조 복잡도 × 질의 수준)별 평균, 평가자 내 중심화 — 원고 4.5절의 값
# (b) 대상 프로파일: 16개 평가 대상(모델 × 조건)별 값, 평가자 내 중심화. 같은 평가자의 도메인 간 상관을
#     서로 다른 평가자 쌍의 도메인 간 상관과 비교한다.
multi = [e for e in experts_in(df) if df.loc[df.Expert == e, "Domain"].nunique() > 1]
def profile(e, d, by):
    v = df[(df.Expert == e) & (df.Domain == d)].groupby(by).CQS.mean().sort_index().values
    return v - v.mean()
out["multi_domain_profiles"] = {}
for name, by in [("conditions_4", ["Scenario"]), ("targets_16", ["M", "Scenario"])]:
    same, other = [], []
    for a, b in itertools.combinations(DOMAINS, 2):
        for e in experts_in(df, a):
            for f in experts_in(df, b):
                x, y = profile(e, a, by), profile(f, b, by)
                if x.std() > 0 and y.std() > 0:
                    (same if e == f else other).append(float(np.corrcoef(x, y)[0, 1]))
    t, p = stats.ttest_1samp(same, 0)
    out["multi_domain_profiles"][name] = dict(raters=len(multi), pairs=len(same), mean_r=float(np.mean(same)), t=float(t), df=len(same) - 1, p=float(p),
                                             other_rater_pairs=len(other), other_mean_r=float(np.mean(other)),
                                             p_same_vs_other=float(stats.mannwhitneyu(same, other).pvalue))

# ---------------------------------------------------------------- 6) 도메인 간 차이의 부트스트랩
rng = np.random.default_rng(SEED["boot"])
means = {}
for dom in DOMAINS:
    f, r = panels(dom)
    means[dom] = (mat(dom, f).mean(0), mat(dom, r).mean(0))
diffs = {("Communication", "Healthcare"): [], ("Energy", "Healthcare"): []}
for _ in range(N_BOOT):
    idx = rng.integers(0, 16, 16)
    rr = {d: stats.spearmanr(means[d][0][idx], means[d][1][idx]).statistic for d in DOMAINS}
    for a, b in diffs:
        diffs[(a, b)].append(rr[a] - rr[b])
obs = {d: stats.spearmanr(*means[d]).statistic for d in DOMAINS}
out["domain_difference_bootstrap"] = {f"{a}-{b}": dict(diff=float(obs[a] - obs[b]), lo=float(np.nanpercentile(v, 2.5)), hi=float(np.nanpercentile(v, 97.5)))
                                      for (a, b), v in diffs.items()}

# ---------------------------------------------------------------- 7) 배제 조건 (4.6절, 표 9)
items = df.groupby(["Expert", "Domain"])[ITEMS].apply(lambda g: g.values.ravel())
zero_var = sorted({e for (e, d), v in items.items() if v.std() == 0})
restricted = sorted({(e, d) for (e, d), v in items.items() if set(np.unique(v)) <= {4, 5}}, key=lambda x: (int(x[0][1:]), x[1]))
out["response_patterns"] = dict(zero_variance=zero_var, only_4_and_5=[f"{e}|{d}" for e, d in restricted],
                                sd={f"{e}|{d}": float(items[(e, d)].std()) for e, d in restricted})
rest_first = [(e, d) for e, d in restricted if df.loc[df.Expert == e, "Cohort"].iloc[0] == "derivation"]
rest_rep = [(e, d) for e, d in restricted if (e, d) not in rest_first and e not in zero_var]
conditions = {
    "none": [],
    "zero_variance": [(e, d) for e, d in restricted if e in zero_var],
    "repeat_range_restricted": [(e, d) for e, d in restricted if (e, d) not in rest_first],
    "first_range_restricted": rest_first,
    "all_range_restricted": restricted,
}
rng = np.random.default_rng(SEED["perm"].spawn(1)[0])
out["exclusion"] = {}
for name, drop in conditions.items():
    row = {}
    for dom in DOMAINS:
        dd = [e for e, d in drop if d == dom]
        f, r = panels(dom, drop=dd)
        if name == "none":              # 배제 없음은 2)의 결과(표 7)를 그대로 쓴다
            rh, pp = out["perm_rho"][dom]["rho"], out["perm_rho"][dom]["p"]
        else:
            rh, pp = perm_p(mat(dom, f), mat(dom, r), rng)
        X = mat(dom, f + r).T
        ic = icc_all(X) if X.shape[1] > 1 else {}
        row[dom] = dict(rho=rh, p=pp, k=f"{len(f)}/{len(r)}", alpha=float(krippendorff_alpha_interval(X.T)),
                        icc_c1=float(ic.get("ICC_C1", np.nan)), icc_ck=float(ic.get("ICC_Ck", np.nan)))
    out["exclusion"][name] = row
hc = [e for e in experts_in(df, "Healthcare") if ("Healthcare" not in [d for x, d in restricted if x == e])]
out["healthcare_without_range_restricted"] = dict(raters=len(hc), icc_c1=float(icc_all(mat("Healthcare", hc).T)["ICC_C1"]))

json.dump(out, open(OUT_DIR / "out_09_supplementary_checks.json", "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps(out, indent=1, ensure_ascii=False, default=lambda x: round(float(x), 4)))
