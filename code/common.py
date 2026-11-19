"""
common.py — 데이터 로딩, 패널 정의, 신뢰도 지표 공통 함수

원자료 경로는 환경변수 SURVEY_XLSX로 바꿀 수 있다. 기본값은 data/raw/survey.xlsx이다.
원자료는 저장소에 포함되지 않는다(data/README.md).
"""

import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 설정

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = os.environ.get("SURVEY_XLSX", str(ROOT / "data" / "raw" / "survey.xlsx"))
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
SHEET = "종합"

ITEMS = [f"Q{i}" for i in range(1, 9)]

# 4개 평가 차원 (각 2문항)
DIMENSIONS = {
    "Plausibility": ["Q1", "Q2"],
    "TechnicalValidity": ["Q3", "Q4"],
    "Consistency": ["Q5", "Q6"],
    "OperationalValue": ["Q7", "Q8"],
}

# 모델명 축약
MODEL_SHORT = {
    "Qwen3-8B": "Qwen3",
    "LLaMA 3.1 8B Instruct": "LLaMA",
    "Foundation-Sec-8B-Instruct": "Foundation",
    "DeepSeek-R1-0528-Qwen3-8B": "DeepSeek",
}

# 패널 정의 (모집 시점 기준). derivation = 최초 패널, validation = 반복 패널.
# E6는 응답이 불완전하여 분석에서 제외되었으므로 자료에 없다(원고 3.3.3절).
DERIVATION = ["E1", "E2", "E3", "E4", "E5", "E7", "E8", "E9", "E10", "E11"]
VALIDATION = ["E12", "E13", "E14", "E15", "E16", "E17",
              "E18", "E19", "E20", "E21"]

DOMAINS = ["Communication", "Energy", "Healthcare"]

# 응답 품질 기준 배제 후보 (전 문항 동일 응답: 분산 0)
STRAIGHTLINE_CANDIDATES = ["E20"]


# ---------------------------------------------------------------- 로딩

def load_data(path=DATA_PATH, sheet=SHEET, exclude=None):
    """설문 원자료를 읽고 파생 변수를 붙인다.

    반환 컬럼
        Expert, Model, M(축약), Domain, Scenario, Q1..Q8,
        CQS(8문항 평균), 4개 차원 평균, Cohort
    """
    df = pd.read_excel(path, sheet_name=sheet)

    missing = [c for c in ["Expert", "Model", "Domain", "Scenario"] + ITEMS
               if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    if exclude:
        df = df[~df["Expert"].isin(exclude)].copy()

    df["CQS"] = df[ITEMS].mean(axis=1)
    for dim, cols in DIMENSIONS.items():
        df[dim] = df[cols].mean(axis=1)

    df["M"] = df["Model"].map(MODEL_SHORT).fillna(df["Model"])
    df["Cohort"] = np.where(df["Expert"].isin(DERIVATION),
                            "derivation", "validation")
    return df.reset_index(drop=True)


def experts_in(df, domain=None, cohort=None):
    """도메인/코호트별 전문가 ID를 번호순으로 반환."""
    sub = df
    if domain is not None:
        sub = sub[sub["Domain"] == domain]
    if cohort is not None:
        sub = sub[sub["Cohort"] == cohort]
    return sorted(sub["Expert"].unique(), key=lambda e: int(e[1:]))


def rating_matrix(df, domain, experts=None, value="CQS",
                  unit=("Model", "Scenario")):
    """신뢰도 행렬 생성.

    반환: (평가자 × 평가대상) DataFrame.
    행 = rater(전문가), 열 = unit(기본값 모델×시나리오 16개).
    Krippendorff/ICC 함수에 넘길 때 방향에 주의할 것.
    """
    sub = df[df["Domain"] == domain]
    if experts is not None:
        sub = sub[sub["Expert"].isin(experts)]
    piv = sub.pivot_table(index="Expert", columns=list(unit),
                          values=value, aggfunc="mean")
    order = experts if experts is not None else experts_in(sub, domain)
    return piv.reindex(order)


# ---------------------------------------------- Krippendorff's alpha

def krippendorff_alpha_interval(matrix):
    """Krippendorff's alpha (interval metric).

    matrix : array-like, shape = (raters, units). 결측은 np.nan.

    폐형식 사용:
        Do = (1/n) * sum_u [ 2 * m_u * SS_u / (m_u - 1) ]
        De = 2 * SS_total / (n - 1)
        alpha = 1 - Do / De
    여기서 n은 전체 값의 개수, m_u는 unit u의 평정 수,
    SS_u는 unit u 내 편차제곱합이다.
    """
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("2차원 배열이어야 함")

    cnt = (~np.isnan(M)).sum(axis=0)
    keep = cnt >= 2
    if keep.sum() == 0:
        return np.nan

    X = M[:, keep]
    m = cnt[keep].astype(float)
    mean_u = np.nanmean(X, axis=0)
    SS_u = np.nansum((X - mean_u) ** 2, axis=0)

    n = m.sum()
    vals = X[~np.isnan(X)]
    SS_total = ((vals - vals.mean()) ** 2).sum()
    if SS_total == 0 or n < 2:
        return np.nan

    Do = (2.0 / n) * np.sum(m * SS_u / (m - 1.0))
    De = 2.0 * SS_total / (n - 1.0)
    return 1.0 - Do / De




# ------------------------------------------------------------- ICC

def icc_all(matrix):
    """이원 분산분석 기반 급내상관.

    matrix : array-like, shape = (units, raters). 결측 없는 완전자료 가정.

    반환 dict
        ICC_C1 : 일관성, 단일 평정자
        ICC_Ck : 일관성, k명 평균
        ICC_A1 : 절대일치, 단일 평정자
        ICC_Ak : 절대일치, k명 평균
        MSR/MSC/MSE 및 분산성분 비율
    """
    M = np.asarray(matrix, dtype=float)
    if np.isnan(M).any():
        raise ValueError("결측이 있는 경우 완전자료로 정리 후 호출할 것")
    n, k = M.shape
    if n < 2 or k < 2:
        return {key: np.nan for key in
                ("ICC_C1", "ICC_Ck", "ICC_A1", "ICC_Ak",
                 "MSR", "MSC", "MSE", "p_target", "p_rater", "p_resid")}

    gm = M.mean()
    MSR = k * ((M.mean(axis=1) - gm) ** 2).sum() / (n - 1)      # unit(대상)
    MSC = n * ((M.mean(axis=0) - gm) ** 2).sum() / (k - 1)      # rater(평가자)
    SST = ((M - gm) ** 2).sum()
    SSE = SST - MSR * (n - 1) - MSC * (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))

    return {
        "ICC_C1": (MSR - MSE) / (MSR + (k - 1) * MSE),
        "ICC_Ck": (MSR - MSE) / MSR,
        "ICC_A1": (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n),
        "ICC_Ak": (MSR - MSE) / (MSR + (MSC - MSE) / n),
        "MSR": MSR, "MSC": MSC, "MSE": MSE,
        "p_target": MSR * (n - 1) / SST,
        "p_rater": MSC * (k - 1) / SST,
        "p_resid": SSE / SST,
    }


def spearman_brown_k(r1, target):
    """단일 평정자 신뢰도 r1에서 목표 신뢰도 달성에 필요한 평정자 수."""
    if not np.isfinite(r1) or r1 <= 0 or not 0 < target < 1:
        return np.nan
    return target * (1 - r1) / (r1 * (1 - target))


def spearman_brown_rk(r1, k):
    """k명 평균의 예측 신뢰도."""
    return k * r1 / (1 + (k - 1) * r1)


# --------------------------------------------------------- 부분집합

def enumerate_subsets(items, n, cap=2000, seed=0):
    """크기 n의 부분집합. 총 개수가 cap 이하이면 전수 열거.

    반환: (부분집합 리스트, 전수열거 여부)
    """
    N = len(items)
    total = 1
    for i in range(n):
        total = total * (N - i) // (i + 1)
    if total <= cap:
        return [list(c) for c in itertools.combinations(items, n)], True

    rng = np.random.default_rng(seed)
    seen, out = set(), []
    while len(out) < cap:
        key = tuple(sorted(rng.choice(N, n, replace=False)))
        if key in seen:
            continue
        seen.add(key)
        out.append([items[i] for i in key])
    return out, False


def fmt(x, nd=4):
    return "     nan" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:{nd + 4}.{nd}f}"


if __name__ == "__main__":
    d = load_data()
    print(f"관측치 {len(d)}, 고유 전문가 {d.Expert.nunique()}, "
          f"결측 {d[ITEMS].isna().sum().sum()}")
    print(pd.crosstab(d.Domain, d.Cohort))
