"""
30_reproduce_from_public_data.py — 공개 파생 자료만으로 원고의 주요 수치를 재산출한다(원자료 불필요).

입력: data/derived/mean_squares.csv, data/derived/panel_means.csv
출력: outputs/out_30_public_reproduction.json

  - Krippendorff's α(간격 계량), ICC(C,1), ICC(C,k), ICC(A,1), ICC(A,k)          원고 표 3
  - 분산성분 σt², σr², σe²와 단일 평정자 일관성 r1                                표 1
  - ICC(C,k) ≥ .70, .80에 필요한 평가자 수                                        4.3절
  - 패널 간 순위상관(16개 대상, 4개 모델), 감쇠 예측, 역산 r1                     표 6, 7, 8
  - 기준선/비교 판정의 소요 평가자 비와 95% 구간                                  5.4절
  - 판정 형태별 소요 인원과 정확도(통합 분산성분)                                  표 12

실행
    python code/30_reproduce_from_public_data.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MS = pd.read_csv(ROOT / "data" / "derived" / "mean_squares.csv")
PM = pd.read_csv(ROOT / "data" / "derived" / "panel_means.csv")
out = {"reliability": {}, "variance_components": {}, "required_raters": {}, "cross_panel": {}, "ratio": {}}


def sb(r1, k):
    return k * r1 / (1 + (k - 1) * r1)


for row in MS.itertuples():
    n, k, t, r, e = row.n_targets, row.k_raters, row.ms_target, row.ms_rater, row.ms_resid
    ss_t, ss_r, ss_e = t * (n - 1), r * (k - 1), e * (n - 1) * (k - 1)
    alpha = 1 - (n * k - 1) * (ss_r + ss_e) / (n * (k - 1) * (ss_t + ss_r + ss_e))
    out["reliability"][f"{row.domain}|{row.panel}"] = dict(
        k=k, alpha=alpha,
        icc_c1=(t - e) / (t + (k - 1) * e), icc_ck=(t - e) / t,
        icc_a1=(t - e) / (t + (k - 1) * e + k * (r - e) / n), icc_ak=(t - e) / (t + (r - e) / n))
    if row.panel == "full":
        vt, vr, ve = max((t - e) / k, 0), max((r - e) / n, 0), e
        r1 = vt / (vt + ve)
        out["variance_components"][row.domain] = dict(vt=vt, vr=vr, ve=ve, r1=r1)
        out["required_raters"][row.domain] = {str(g): math.ceil(g * (1 - r1) / (r1 * (1 - g))) for g in (0.70, 0.80)}
        F, d1, d2 = r / e, k - 1, (n - 1) * (k - 1)
        th = [(F / stats.f.ppf(q, d1, d2) - 1) / n for q in (0.975, 0.025)]
        out["ratio"][row.domain] = dict(ratio=(vr + ve) / (2 * ve), lo=(th[0] + 1) / 2, hi=(th[1] + 1) / 2)

for dom, g in PM.groupby("domain", sort=False):
    rho16 = stats.spearmanr(g.first_mean, g.repeat_mean).statistic
    mm = g.groupby("model")[["first_mean", "repeat_mean"]].mean()
    rho4 = stats.spearmanr(mm.first_mean, mm.repeat_mean).statistic
    r1 = out["variance_components"][dom]["r1"]
    kA, kB = int(g.k_first.iloc[0]), int(g.k_repeat.iloc[0])
    pred = lambda x: math.sqrt(sb(x, kA) * sb(x, kB))
    n = len(g)
    moran = lambda r: 6 / (math.pi * (n + 1)) * (math.asin(r) + (n - 2) * math.asin(r / 2))
    from scipy.optimize import brentq
    out["cross_panel"][dom] = dict(rho_16=rho16, rho_4_models=rho4, k=f"{kA}/{kB}",
                                   R_A=sb(r1, kA), R_B=sb(r1, kB), predicted_r=pred(r1), predicted_rho=moran(pred(r1)),
                                   # 표 8의 역산 r1: 감쇠식에 관측 순위상관을 그대로 대입한 값과, Moran 환산을 거쳐 역산한 값
                                   inverse_r1=brentq(lambda x: pred(x) - rho16, 1e-6, 0.99),
                                   inverse_r1_via_moran=brentq(lambda x: moran(pred(x)) - rho16, 1e-6, 0.99))

vc = pd.DataFrame(out["variance_components"]).T
VT, VR, VE = vc.vt.mean(), vc.vr.mean(), vc.ve.mean()
out["pooled"] = dict(vt=VT, vr=VR, ve=VE, ratio=(VR + VE) / (2 * VE), sd_targets=math.sqrt(VT))
table12 = []
for delta in (round(math.sqrt(VT), 2), 0.25, 0.50):
    row = dict(delta=delta)
    for p in (0.80, 0.90):
        z = stats.norm.ppf(p)
        row[f"k_comparative_{p}"] = math.ceil(2 * VE * z ** 2 / delta ** 2)
        row[f"k_threshold_{p}"] = math.ceil((VR + VE) * z ** 2 / delta ** 2)
    for k in (5, 10):
        row[f"acc_comparative_k{k}"] = stats.norm.cdf(delta / math.sqrt(2 * VE / k))
        row[f"acc_threshold_k{k}"] = stats.norm.cdf(delta / math.sqrt((VR + VE) / k))
    table12.append(row)
out["table12"] = table12

(ROOT / "outputs").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "outputs" / "out_30_public_reproduction.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4)))
