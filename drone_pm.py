"""드론 예측정비 실습 - 전체 파이프라인 (README.txt의 [1]~[8] 단계)

실행:  .venv\\Scripts\\python drone_pm.py
결과:  outputs/ 폴더
  - eda_summary.txt              [1] 데이터 살펴보기 결과
  - cleaning_log.csv             [2] 이상치·결측 처리 내역(행 단위)
  - validation_metrics.csv       [5] 검증 성능(모델·기준선)
  - validation_monthly_hits.csv  [5] 검증 기간 월별 20대 적중 수
  - threshold_costs.csv          [6] 확률 기준별 비용(300/600만원)
  - test_predictions_2025.csv    [7] 제출용 시험 기간 예측
  - this_month_predictions.csv   [8] 2026-01 정비 드론 전체(드론별 최근 기록) 확률
  - recheck_plan_2026_01.csv     [8] 2026-01 재점검 대상 20대
  - report.md                    제출물 (2) 분석 보고서 (모든 숫자는 이 스크립트 실행 결과)
  - recheck_plan_2026_01.md      제출물 (3) 2026-01 재점검 계획 · 모델의 한계
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if hasattr(sys.stdout, "reconfigure"):      # 주피터 등에서는 없음
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

TARGET = "failure_within_30days"
FAIL_TYPES = {"사후정비", "긴급정비"}
K = 20                       # 한 달 재점검 가능 대수
COST_MISS = 300              # 놓친 재고장 1건(만원)
COST_MISS_ALT = 600          # 민감도 확인용
COST_CHECK = 60              # 재점검 1대(만원)

VALID_START = pd.Timestamp("2024-07-01")   # 검증 기간 2024-07 ~ 2024-12
PURGE_DAYS = 30                             # 학습 정답 창이 검증 기간과 겹치지 않도록 제외
BOUNDARY_GAP = 0.01                         # 20위·21위 확률 차이가 이보다 작으면 보고서에 경계 주의 문구
                                            # (랜덤 시드만 바꿔도 검증 PR-AUC가 약 0.004 흔들림)

eda_lines: list[str] = []


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    eda_lines.append(msg)


# ----------------------------------------------------------------------------
# 0. 데이터 읽기
# ----------------------------------------------------------------------------
def load():
    tr = pd.read_csv(DATA / "drone_train.csv", encoding="utf-8-sig")
    te = pd.read_csv(DATA / "drone_test.csv", encoding="utf-8-sig")
    tm = pd.read_csv(DATA / "drone_this_month.csv", encoding="utf-8-sig")
    tr["split"], te["split"], tm["split"] = "train", "test", "this_month"
    df = pd.concat([tr, te, tm], ignore_index=True)
    df["maintenance_date"] = pd.to_datetime(df["maintenance_date"])
    df["row_id"] = np.arange(len(df))
    return df


def cost_unit_error(cost: pd.Series) -> pd.Series:
    """100배 단위 오입력: 1만 이상이면서 100의 배수 (정상 최댓값 611, 정상값 중 100의 배수는 100·200·300뿐)"""
    return (cost >= 10000) & (cost % 100 == 0)


def repeat_failure_drones(train: pd.DataFrame) -> list[str]:
    """반복 고장 드론: 학습 데이터에서 정비 간격 중앙값 7일 이하이면서 재고장률 80% 이상"""
    g = train.groupby("equipment_id").agg(gap=("days_since_last_maintenance", "median"), rate=(TARGET, "mean"))
    return g[(g.gap <= 7) & (g.rate >= 0.8)].index.tolist()


# ----------------------------------------------------------------------------
# [1] 데이터 살펴보기
# ----------------------------------------------------------------------------
def explore(df: pd.DataFrame):
    log("=" * 70, "\n[1] 데이터 살펴보기\n" + "=" * 70)
    for s, g in df.groupby("split", sort=False):
        log(f"{s}: {len(g)}행, 드론 {g.equipment_id.nunique()}대, "
            f"{g.maintenance_date.min():%Y-%m-%d} ~ {g.maintenance_date.max():%Y-%m-%d}")
    log("\n결측 수(파일별):")
    for s, g in df.groupby("split", sort=False):
        na = g.drop(columns=[TARGET]).isna().sum()
        log(f"  {s}: {na[na > 0].to_dict()}")
    log("  ※ this_month 파일에는 availability_rate 컬럼 자체가 없음")

    tr = df[df.split == "train"]
    log(f"\n학습 데이터 재고장률: {tr[TARGET].mean():.3f} ({int(tr[TARGET].sum())}/{len(tr)})")

    na_fault = tr[tr.fault_category.isna()].maintenance_type.value_counts().to_dict()
    log(f"fault_category 결측 행의 정비 구분: {na_fault}  -> 예방정비에는 고장이 없어 '원래 값 없음'")
    first_rows = df.sort_values("maintenance_date").groupby("equipment_id").head(1)
    log(f"days_since_last_maintenance 결측 {int(df.days_since_last_maintenance.isna().sum())}건 / "
        f"드론별 첫 기록 {len(first_rows)}건, 그중 결측 {int(first_rows.days_since_last_maintenance.isna().sum())}건"
        " -> 직전 정비가 없어 '원래 값 없음'")

    log("\n수치 컬럼 분포(학습):")
    num_cols = ["equipment_age_years", "severity", "downtime_hours", "repair_duration_hours",
                "parts_replaced_count", "maintenance_cost", "cumulative_operating_hours",
                "days_since_last_maintenance", "availability_rate"]
    log(tr[num_cols].describe(percentiles=[.01, .5, .99]).T.round(2).to_string())

    log("\n이상치 후보:")
    log(f"  downtime_hours == 999 (결측 코드): {(df.downtime_hours >= 999).groupby(df.split).sum().to_dict()}"
        f"  (그 외 최댓값 {df.loc[df.downtime_hours < 999, 'downtime_hours'].max()})")
    big = cost_unit_error(df.maintenance_cost)
    log(f"  maintenance_cost >= 10,000 이면서 100의 배수: {big.groupby(df.split).sum().to_dict()}, "
        f"값 {sorted(df.loc[big, 'maintenance_cost'].tolist())}")
    log(f"    정비 구분별 나머지 최댓값 {df[~big].groupby('maintenance_type').maintenance_cost.max().to_dict()}, "
        f"{df.loc[~big, 'maintenance_cost'].max()}~{df.loc[big, 'maintenance_cost'].min()} 사이 값 "
        f"{int(df.maintenance_cost.between(df.loc[~big, 'maintenance_cost'].max() + 1, df.loc[big, 'maintenance_cost'].min() - 1).sum())}건, "
        f"나머지 중 100의 배수 {int((df.loc[~big, 'maintenance_cost'] % 100 == 0).sum())}건")
    low = tr.availability_rate <= 0.3
    log(f"  availability_rate <= 0.3: {int(low.sum())}건, 이들의 재고장률 {tr.loc[low, TARGET].mean():.2f}"
        f" (그 외 {tr.loc[~low, TARGET].mean():.2f}) -> 정비 뒤 그 달의 고장이 반영된 값(누수)")

    # 일관성 점검
    s = df.sort_values(["equipment_id", "maintenance_date"])
    gap = s.groupby("equipment_id").maintenance_date.diff().dt.days
    log(f"\n일관성: 날짜 차이와 days_since_last 불일치 {int(((gap - s.days_since_last_maintenance).abs() > 0).sum())}건, "
        f"누적비행시간 감소 {int((s.groupby('equipment_id').cumulative_operating_hours.diff() < 0).sum())}건, "
        f"동일 드론-날짜 중복 {int(df.duplicated(['equipment_id', 'maintenance_date']).sum())}건")

    # 반복 고장 드론 (입력 오류가 아니라 패턴이 다른 기체)
    rep = repeat_failure_drones(tr)
    cnt = df.groupby("equipment_id").size()
    for e in rep:
        g = tr[tr.equipment_id == e]
        log(f"\n반복 고장 드론 {e}: 전체 기록 {cnt[e]}건(드론 중앙값 {cnt.median():.0f}건), "
            f"정비 간격 중앙값 {g.days_since_last_maintenance.median():.0f}일, 학습 재고장률 {g[TARGET].mean():.2f} "
            "-> 날짜·비행시간이 일관되어 입력 오류가 아님. 지우지 않음")

    # 기간에 따른 분포 변화
    log("\n기간별 분포(중앙값): " + ", ".join(
        f"{s} 경과연수 {g.equipment_age_years.median():.1f}년 / 누적 {g.cumulative_operating_hours.median():.0f}시간"
        for s, g in df.groupby("split", sort=False)))
    for s in ("test", "this_month"):
        g = df[df.split == s]
        log(f"  {s}: 학습 최댓값보다 오래된 기록(경과연수 > {tr.equipment_age_years.max()}) "
            f"{(g.equipment_age_years > tr.equipment_age_years.max()).mean():.1%}")

    log("\n패턴(학습, 재고장률):")
    for c in ["maintenance_type", "equipment_type", "fault_category"]:
        r = tr.groupby(tr[c].fillna("없음"))[TARGET].agg(["mean", "size"]).round(3)
        log(f"  [{c}]\n" + r.to_string())
    season = tr.maintenance_date.dt.month.map({12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
                                               6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"})
    log("  [계절]\n" + tr.groupby(season)[TARGET].agg(["mean", "size"]).round(3).to_string())
    log("  [계절 x 고장유형: 고장 건수]\n"
        + pd.crosstab(season, tr.fault_category).to_string())
    log("  [드론유형 x 고장유형: 고장 건수]\n"
        + pd.crosstab(tr.equipment_type, tr.fault_category).to_string())


# ----------------------------------------------------------------------------
# [2] 데이터 처리 기준
# ----------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """행은 지우지 않는다. 모든 이상치는 값 수정 + 표시 컬럼으로 남긴다."""
    df = df.copy()
    logs = []

    # (a) downtime_hours 999 -> 결측 코드. 값을 지어내지 않고 '모름'(NaN)으로 두고 표시한다.
    #     (다른 행·다른 기간 값으로 채우지 않으므로 미래 정보가 섞이지 않음)
    bad = df.downtime_hours >= 999
    df["downtime_missing"] = bad.astype(int)
    for i in df.index[bad]:
        logs.append((df.at[i, "split"], df.at[i, "equipment_id"], df.at[i, "maintenance_date"].date(),
                     "downtime_hours", df.at[i, "downtime_hours"], np.nan,
                     "999는 결측 코드. 모름(빈 값)으로 두고 downtime_missing=1 표시"))
    df.loc[bad, "downtime_hours"] = np.nan

    # (b) maintenance_cost >= 10,000 이면서 100의 배수 -> 단위 오입력(원래 값 x100).
    #     100으로 나누면 150~190으로 사후정비 정상 범위이고, 수리시간·부품 수로 본 예상 비용과도 맞음
    bad = cost_unit_error(df.maintenance_cost)
    df["cost_fixed"] = bad.astype(int)
    for i in df.index[bad]:
        logs.append((df.at[i, "split"], df.at[i, "equipment_id"], df.at[i, "maintenance_date"].date(),
                     "maintenance_cost", df.at[i, "maintenance_cost"], df.at[i, "maintenance_cost"] / 100,
                     "1만 이상이면서 100의 배수. 100배 단위 오입력으로 보고 /100"))
    df.loc[bad, "maintenance_cost"] = df.loc[bad, "maintenance_cost"] / 100

    # (c) 구조적 결측: 예방정비는 고장 없음 -> '없음'
    df["fault_category"] = df.fault_category.fillna("없음")
    df["fault_code"] = df.fault_code.fillna("없음")
    # days_since_last_maintenance 결측(첫 기록)은 피처 단계에서 is_first 표시 + 대체

    pd.DataFrame(logs, columns=["split", "equipment_id", "maintenance_date", "column",
                                "before", "after", "reason"]).to_csv(
        OUT / "cleaning_log.csv", index=False, encoding="utf-8-sig")
    print(f"\n[2] 이상치 수정 {len(logs)}건 -> outputs/cleaning_log.csv (삭제한 행 0건)")
    return df


# ----------------------------------------------------------------------------
# [3] 피처 만들기 (드론별 과거 기록만 사용, train-test-this_month를 이어서 계산)
# ----------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["equipment_id", "maintenance_date", "row_id"]).reset_index(drop=True)
    df["is_failure"] = df.maintenance_type.isin(FAIL_TYPES).astype(int)

    hist_cols = {k: np.zeros(len(df)) for k in [
        "fail_cnt_30d", "fail_cnt_90d", "fail_cnt_180d", "fail_cnt_365d", "maint_cnt_90d",
        "same_cat_fail_365d", "unsuccess_cnt_180d", "severity_sum_180d",
        "days_since_prev_failure", "days_since_last_pm", "hours_since_last_pm",
        "n_prior_records", "hours_per_day_recent"]}
    prev_type = np.empty(len(df), dtype=object)

    for _, idx in df.groupby("equipment_id").indices.items():
        g = df.loc[idx]
        dates = g.maintenance_date.values
        fail = g.is_failure.values
        cat = g.fault_category.values
        succ = g.maintenance_success.values
        sev = g.severity.values
        hrs = g.cumulative_operating_hours.values
        mtype = g.maintenance_type.values
        for j, row in enumerate(idx):
            d = dates[j]
            past = slice(0, j + 1)          # 현재 정비 포함(정비 직후 시점에는 알고 있음)
            age_days = (d - dates[past]).astype("timedelta64[D]").astype(int)
            f = fail[past]
            for w in (30, 90, 180, 365):
                hist_cols[f"fail_cnt_{w}d"][row] = f[age_days <= w].sum()
            hist_cols["maint_cnt_90d"][row] = (age_days <= 90).sum()
            hist_cols["same_cat_fail_365d"][row] = (
                ((cat[past] == cat[j]) & (f == 1) & (age_days <= 365)).sum() if fail[j] else 0)
            hist_cols["unsuccess_cnt_180d"][row] = ((succ[past] == 0) & (age_days <= 180)).sum()
            hist_cols["severity_sum_180d"][row] = sev[past][age_days <= 180].sum()

            prev_f = np.where(fail[:j] == 1)[0]  # 현재 제외, 직전 고장
            hist_cols["days_since_prev_failure"][row] = (
                (d - dates[prev_f[-1]]).astype("timedelta64[D]").astype(int) if len(prev_f) else 730)
            pm = np.where(mtype[: j + 1] == "예방정비")[0]
            if len(pm):
                hist_cols["days_since_last_pm"][row] = (d - dates[pm[-1]]).astype("timedelta64[D]").astype(int)
                hist_cols["hours_since_last_pm"][row] = hrs[j] - hrs[pm[-1]]
            else:
                hist_cols["days_since_last_pm"][row] = np.nan
                hist_cols["hours_since_last_pm"][row] = np.nan
            hist_cols["n_prior_records"][row] = j
            if j > 0:
                dd = (d - dates[j - 1]).astype("timedelta64[D]").astype(int)
                hist_cols["hours_per_day_recent"][row] = (hrs[j] - hrs[j - 1]) / max(dd, 1)
            else:
                hist_cols["hours_per_day_recent"][row] = np.nan
            prev_type[row] = mtype[j - 1] if j > 0 else "없음"

    for k, v in hist_cols.items():
        df[k] = v
    df["prev_maintenance_type"] = prev_type

    # 직전 정비 없음(드론별 첫 기록) / 예방정비 이력 없음 -> 이유가 있는 결측.
    # 0('방금 정비함')이나 중앙값('51일 전 정비')으로 채우면 없는 사실이 생기므로 비워 두고 표시만 한다.
    # (트리 모델은 빈 값을 그대로 학습, 로지스틱회귀만 모델 안에서 학습 데이터 중앙값으로 채움)
    df["is_first_maintenance"] = df.days_since_last_maintenance.isna().astype(int)

    m = df.maintenance_date.dt.month
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    return df.sort_values("row_id").reset_index(drop=True)


# ----------------------------------------------------------------------------
# [4] 쓸 컬럼 고르기
# ----------------------------------------------------------------------------
CAT_FEATURES = ["equipment_type", "unit_code", "maintenance_type", "fault_category",
                "prev_maintenance_type"]
NUM_FEATURES = [
    "equipment_age_years", "severity", "downtime_hours", "repair_duration_hours",
    "parts_replaced_count", "maintenance_cost", "maintenance_success",
    "cumulative_operating_hours", "days_since_last_maintenance",
    "downtime_missing", "is_first_maintenance", "is_failure",
    "fail_cnt_30d", "fail_cnt_90d", "fail_cnt_180d", "fail_cnt_365d", "maint_cnt_90d",
    "same_cat_fail_365d", "unsuccess_cnt_180d", "severity_sum_180d",
    "days_since_prev_failure", "days_since_last_pm", "hours_since_last_pm",
    "n_prior_records", "hours_per_day_recent", "month_sin", "month_cos",
]
FEATURES = CAT_FEATURES + NUM_FEATURES

EXCLUDED = {
    "availability_rate": "'그 달' 가동률이라 정비 뒤 같은 달의 고장·중단이 섞임(1장: 0.3 이하이면 재고장률이 매우 높음). "
                         "정비 직후에는 알 수 없고 이번 달 파일에도 없음",
    "failure_within_30days": "예측 대상(정답)",
    "fault_code": "약 40종으로 잘게 나뉘어 학습 건수가 적음. 같은 정보를 fault_category로 사용",
    "equipment_id": "식별자. 드론 고유 성향은 과거 이력 피처로 반영",
    "maintenance_date": "날짜 자체 대신 계절(월 sin/cos)과 경과일 피처로 사용",
    "cost_fixed": "데이터 처리 표시용(입력 오류 여부는 재고장과 무관)",
}

FEATURE_KO = {
    "equipment_type": "드론 유형", "unit_code": "부대", "maintenance_type": "이번 정비 구분",
    "fault_category": "이번 고장 유형", "prev_maintenance_type": "직전 정비 구분",
    "equipment_age_years": "경과연수", "severity": "고장 심각도", "downtime_hours": "운용중단 시간",
    "repair_duration_hours": "수리 시간", "parts_replaced_count": "교체 부품 수",
    "maintenance_cost": "정비 비용", "maintenance_success": "정비 성공 여부",
    "cumulative_operating_hours": "누적 비행시간", "days_since_last_maintenance": "직전 정비 뒤 경과일",
    "downtime_missing": "운용중단 시간 모름", "is_first_maintenance": "첫 정비 기록",
    "is_failure": "이번 정비가 고장정비", "fail_cnt_30d": "최근 30일 고장 수",
    "fail_cnt_90d": "최근 90일 고장 수", "fail_cnt_180d": "최근 180일 고장 수",
    "fail_cnt_365d": "최근 1년 고장 수", "maint_cnt_90d": "최근 90일 정비 횟수",
    "same_cat_fail_365d": "1년 내 같은 유형 고장 수", "unsuccess_cnt_180d": "최근 180일 정비 실패 수",
    "severity_sum_180d": "최근 180일 심각도 합", "days_since_prev_failure": "직전 고장 뒤 경과일",
    "days_since_last_pm": "예방정비 뒤 경과일", "hours_since_last_pm": "예방정비 뒤 비행시간",
    "n_prior_records": "과거 정비 기록 수", "hours_per_day_recent": "최근 하루 비행시간",
    "month": "정비 월(계절)",
}

# 새로 만든 피처: (컬럼, 뜻, 왜 넣었나) — 보고서 3장 표
FEATURE_DOC = [
    ("fail_cnt_30d", "이번 정비 포함 최근 30일 사후 · 긴급정비 수", "고장이 몰려 있으면 곧 또 고장 날 가능성이 큼 (README 예시: 최근 기간 고장 수)"),
    ("fail_cnt_90d", "최근 90일 고장 수", "30일보다 긴 흐름"),
    ("fail_cnt_180d", "최근 180일 고장 수", "반기 흐름"),
    ("fail_cnt_365d", "최근 1년 고장 수", "기체의 장기 고장 성향"),
    ("maint_cnt_90d", "최근 90일 정비 횟수(종류 무관)", "정비가 잦은 기체"),
    ("same_cat_fail_365d", "이번 고장과 같은 유형의 1년 내 고장 수", "같은 계통 반복 고장 = 근본 원인 미해결 신호"),
    ("unsuccess_cnt_180d", "최근 180일 정비 실패 수", "정비가 잘 안 된 기체"),
    ("severity_sum_180d", "최근 180일 고장 심각도 합", "고장 건수에 심각도를 더한 부담"),
    ("days_since_prev_failure", "직전 고장(이번 제외) 뒤 경과일, 고장 이력 없으면 730",
     "고장 간격이 짧을수록 위험 (README 예시: 직전 고장 뒤 경과일)"),
    ("days_since_last_pm", "마지막 예방정비 뒤 경과일", "예방정비 주기가 밀리면 위험"),
    ("hours_since_last_pm", "마지막 예방정비 뒤 비행시간", "README 예시: 예방정비 뒤 비행시간"),
    ("n_prior_records", "과거 정비 기록 수", "기체의 누적 정비 이력"),
    ("hours_per_day_recent", "직전 정비 뒤 하루 평균 비행시간", "많이 쓰는 기체는 빨리 닳음"),
    ("prev_maintenance_type", "직전 정비 구분", "고장정비가 연속되는지"),
    ("is_failure", "이번 정비가 사후 · 긴급정비인지", "고장정비 뒤 재고장률이 높음 (1장)"),
    ("month_sin", "정비 월(계절) — sin · cos 두 개로 표현", "계절별 재고장률 차이 (1장). 12월과 1월이 가깝게 표현됨"),
]


# ----------------------------------------------------------------------------
# [5] 모델과 평가 함수
# ----------------------------------------------------------------------------
def make_models():
    ohe = lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return {
        "로지스틱회귀": Pipeline([
            ("prep", ColumnTransformer([("cat", ohe(), CAT_FEATURES),
                                        # 빈 값(운용중단 시간 모름)은 학습 데이터 중앙값으로 채움(모델 안에서 학습 데이터로만 계산)
                                        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                          ("sc", StandardScaler())]), NUM_FEATURES)])),
            ("clf", LogisticRegression(C=0.3, max_iter=2000))]),
        "랜덤포레스트": Pipeline([
            ("prep", ColumnTransformer([("cat", ohe(), CAT_FEATURES),
                                        ("num", "passthrough", NUM_FEATURES)])),
            ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features=0.3,
                                           random_state=SEED, n_jobs=-1))]),
        "그래디언트부스팅": Pipeline([
            ("prep", ColumnTransformer([("cat", ohe(), CAT_FEATURES),
                                        ("num", "passthrough", NUM_FEATURES)])),
            ("clf", HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_depth=3,
                                                   min_samples_leaf=30, l2_regularization=1.0,
                                                   random_state=SEED))]),
    }


def latest_per_drone_month(d: pd.DataFrame) -> pd.DataFrame:
    """한 달에 같은 드론이 여러 번 정비했으면 가장 최근 기록만 남긴다(README [8])."""
    d = d.assign(ym=d.maintenance_date.dt.to_period("M"))
    return d.sort_values(["maintenance_date", "row_id"]).groupby(["ym", "equipment_id"]).tail(1)


def monthly_topk_hits(d: pd.DataFrame, score: np.ndarray, k=K, reps=1, seed=SEED):
    """매달 점수 상위 k대(드론 기준)를 골랐을 때 실제 재고장 수. 동점은 무작위로 깸(reps번 평균)."""
    d = d.assign(score=score)
    d = latest_per_drone_month(d)
    rng = np.random.default_rng(seed)
    out = {}
    for ym, g in d.groupby("ym"):
        hits = []
        for _ in range(reps):
            jitter = rng.random(len(g)) * 1e-9 if reps > 1 else np.zeros(len(g))
            top = np.argsort(-(g.score.values + jitter), kind="stable")[:k]
            hits.append(g[TARGET].values[top].sum())
        out[str(ym)] = float(np.mean(hits))
    return out


def evaluate(name, d, score, reps=1):
    hits = monthly_topk_hits(d, score, reps=reps)
    keep = ~d.equipment_id.isin(REPEAT_DRONES).values      # 반복 고장 드론을 뺀 민감도 확인
    return {"model": name,
            "PR_AUC": round(average_precision_score(d[TARGET], score), 4),
            "ROC_AUC": round(roc_auc_score(d[TARGET], score), 4),
            "top20_hits_total": round(sum(hits.values()), 2),
            "positives_total": int(d[TARGET].sum()),
            "PR_AUC_excl_repeat": round(average_precision_score(d[TARGET].values[keep], np.asarray(score)[keep]), 4),
            }, hits


REPEAT_DRONES: list[str] = []   # main()에서 학습 데이터로 정함


def threshold_table(y, p, cost_miss):
    rows = []
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        flag = p >= t
        miss = int(((~flag) & (y == 1)).sum())
        rows.append({"threshold": t, "flagged": int(flag.sum()), "caught": int((flag & (y == 1)).sum()),
                     "missed": miss, "cost": miss * cost_miss + int(flag.sum()) * COST_CHECK})
    tab = pd.DataFrame(rows)
    # 비용이 같으면 더 높은 기준(점검 건수 적음)을 택함
    best = tab.sort_values(["cost", "threshold"], ascending=[True, False]).iloc[0].to_dict()
    best = pd.Series({k: (float(v) if k == "threshold" else int(v)) for k, v in best.items()}, dtype=object)
    return tab, best


# ----------------------------------------------------------------------------
# [8] 위험 요인 설명: 피처를 '보통 값'으로 바꿨을 때 확률이 얼마나 내려가는지
# ----------------------------------------------------------------------------
def explain_rows(model, X: pd.DataFrame, ref: pd.DataFrame, top_n=3):
    base = model.predict_proba(X)[:, 1]
    groups = {c: [c] for c in FEATURES if c not in ("month_sin", "month_cos")}
    groups["month"] = ["month_sin", "month_cos"]
    ref_vals = {c: (ref[c].mode()[0] if c in CAT_FEATURES or c in
                    ("is_failure", "maintenance_success", "downtime_missing", "is_first_maintenance")
                    else ref[c].median()) for c in FEATURES}
    ref_vals["month_sin"], ref_vals["month_cos"] = 0.0, 0.0   # 계절 효과 제거
    deltas = {}
    for gname, cols in groups.items():
        Xr = X.copy()
        for c in cols:
            Xr[c] = ref_vals[c]
        deltas[gname] = base - model.predict_proba(Xr)[:, 1]
    deltas = pd.DataFrame(deltas, index=X.index)

    texts = []
    for i in X.index:
        top = deltas.loc[i].sort_values(ascending=False)
        top = top[top > 0.005].head(top_n)
        parts = []
        for g, v in top.items():
            if g == "month":
                val = f"{(np.degrees(np.arctan2(X.at[i, 'month_sin'], X.at[i, 'month_cos'])) / 30) % 12:.0f}월"
            else:
                val = X.at[i, g]
                if isinstance(val, (int, float, np.number)):
                    val = "모름" if pd.isna(val) else f"{round(float(val), 1):g}"
                else:
                    val = str(val)
            parts.append(f"{FEATURE_KO[g]}={val}(+{v * 100:.1f}%p)")
        texts.append("; ".join(parts) if parts else "뚜렷한 단일 요인 없음")
    return texts, deltas


# 가장 크게 위험을 올린 요인 -> 그 요인에 맞는 조치 (권고 앞부분)
FACTOR_ACTION = {
    ("fail_cnt_30d", "fail_cnt_90d", "fail_cnt_180d", "fail_cnt_365d", "severity_sum_180d",
     "days_since_prev_failure", "same_cat_fail_365d", "is_failure", "prev_maintenance_type",
     "maint_cnt_90d", "days_since_last_maintenance"): "[최근 고장 반복] 직전 고장 계통의 수리 품질 재확인.",
    ("days_since_last_pm", "hours_since_last_pm"): "[예방정비 주기 경과] 예방정비를 앞당김.",
    ("cumulative_operating_hours", "equipment_age_years", "n_prior_records"): "[노후 기체] 수명 부품 교체 검토.",
    ("maintenance_success", "unsuccess_cnt_180d"): "[정비 실패 이력] 재정비 후 확인 비행.",
    ("hours_per_day_recent",): "[운용 강도 높음] 비행 일정 조정 검토.",
}

RECOMMEND = {
    "배터리": "배터리 셀 전압·내부저항 측정, 충전기 점검, 필요 시 배터리팩 교체",
    "모터": "모터·ESC 진동·발열 점검, 베어링 상태 확인, 프로펠러 균형 점검",
    "통신": "데이터링크·안테나·커넥터 점검, 통신 거리 시험",
    "기체": "프레임·암 균열 점검, 체결부 토크 재확인",
    "센서": "IMU·GPS·짐벌 캘리브레이션, 센서 커넥터 점검",
    "없음": "예방정비 직후 고위험: 최근 고장 이력 계통 위주 정밀 재점검",
}


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    df = load()
    explore(df)
    (OUT / "eda_summary.txt").write_text("\n".join(eda_lines), encoding="utf-8")

    df = clean(df)
    df = build_features(df)

    train = df[df.split == "train"].copy()
    test = df[df.split == "test"].copy()
    month = df[df.split == "this_month"].copy()

    # [5] 기간 분할: 학습 2023-01 ~ 2024-05(정답 창 30일 겹침 제거), 검증 2024-07 ~ 2024-12
    purge_end = VALID_START - pd.Timedelta(days=PURGE_DAYS)
    fit = train[train.maintenance_date < purge_end]
    val = train[train.maintenance_date >= VALID_START]
    print(f"\n[5] 학습 {fit.maintenance_date.min():%Y-%m-%d}~{fit.maintenance_date.max():%Y-%m-%d} "
          f"{len(fit)}건 / 검증 {val.maintenance_date.min():%Y-%m-%d}~{val.maintenance_date.max():%Y-%m-%d} "
          f"{len(val)}건 (제외한 경계 구간 {len(train) - len(fit) - len(val)}건)")

    REPEAT_DRONES[:] = repeat_failure_drones(train)
    print(f"    반복 고장 드론(학습 데이터 기준): {REPEAT_DRONES}")

    results, monthly = [], {}
    rng = np.random.default_rng(SEED)
    # 기준선 1: 무작위 (1000회 평균)
    r = [evaluate("기준선_무작위", val, rng.random(len(val)))[0] for _ in range(1000)]
    base_rand = {"model": "기준선_무작위(1000회 평균)",
                 "PR_AUC": round(np.mean([x["PR_AUC"] for x in r]), 4),
                 "ROC_AUC": round(np.mean([x["ROC_AUC"] for x in r]), 4),
                 "top20_hits_total": round(np.mean([x["top20_hits_total"] for x in r]), 2),
                 "positives_total": r[0]["positives_total"],
                 "PR_AUC_excl_repeat": round(np.mean([x["PR_AUC_excl_repeat"] for x in r]), 4)}
    results.append(base_rand)
    # 기준선 2: 경과연수 오래된 순 (동점은 무작위, 200회 평균)
    res, hits = evaluate("기준선_경과연수순", val, val.equipment_age_years.values, reps=200)
    results.append(res)
    monthly["기준선_경과연수순"] = hits

    preds_val = {}
    for name, model in make_models().items():
        model.fit(fit[FEATURES], fit[TARGET])
        p = model.predict_proba(val[FEATURES])[:, 1]
        preds_val[name] = p
        res, hits = evaluate(name, val, p)
        results.append(res)
        monthly[name] = hits

    metrics = pd.DataFrame(results)
    metrics.to_csv(OUT / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False))

    model_rows = metrics[~metrics.model.str.startswith("기준선")]
    best_name = model_rows.sort_values(["PR_AUC", "top20_hits_total"], ascending=False).iloc[0]["model"]
    print(f"-> 검증 PR-AUC 최고 모델: {best_name}")

    # 성능 불확실성: 검증 표본을 다시 뽑았을 때(부트스트랩 1000회) + 랜덤 시드만 바꿨을 때
    yv = val[TARGET].values
    rng_b = np.random.default_rng(SEED)
    boot_idx = [rng_b.integers(0, len(yv), len(yv)) for _ in range(1000)]
    p_best, p_age = preds_val[best_name], val.equipment_age_years.values
    pr_best = np.array([average_precision_score(yv[i], p_best[i]) for i in boot_idx])
    pr_age = np.array([average_precision_score(yv[i], p_age[i]) for i in boot_idx])
    seed_pr = []
    for s in range(5):
        m = make_models()[best_name]
        if "random_state" in m.named_steps["clf"].get_params():
            m.set_params(clf__random_state=s)
        m.fit(fit[FEATURES], fit[TARGET])
        seed_pr.append(average_precision_score(yv, m.predict_proba(val[FEATURES])[:, 1]))
    unc = pd.DataFrame([
        {"항목": f"{best_name} PR-AUC 95% 구간 (부트스트랩 1000회)",
         "하한": np.percentile(pr_best, 2.5), "상한": np.percentile(pr_best, 97.5), "표준오차_또는_폭": pr_best.std()},
        {"항목": "경과연수순 PR-AUC 95% 구간 (부트스트랩 1000회)",
         "하한": np.percentile(pr_age, 2.5), "상한": np.percentile(pr_age, 97.5), "표준오차_또는_폭": pr_age.std()},
        {"항목": f"PR-AUC 차이 ({best_name} - 경과연수순) 95% 구간",
         "하한": np.percentile(pr_best - pr_age, 2.5), "상한": np.percentile(pr_best - pr_age, 97.5),
         "표준오차_또는_폭": (pr_best - pr_age).std()},
        {"항목": f"{best_name} PR-AUC 랜덤 시드 0~4", "하한": min(seed_pr), "상한": max(seed_pr),
         "표준오차_또는_폭": max(seed_pr) - min(seed_pr)},
    ]).round(4)
    unc.to_csv(OUT / "validation_uncertainty.csv", index=False, encoding="utf-8-sig")

    # 월별로 가능한 최대 적중(그 달 실제 재고장 드론 수, 최대 20)
    lv = latest_per_drone_month(val)
    ceiling = {str(ym): int(min(K, g[TARGET].sum())) for ym, g in lv.groupby("ym")}
    pd.DataFrame({**monthly, "가능한_최대": ceiling}).to_csv(
        OUT / "validation_monthly_hits.csv", encoding="utf-8-sig")

    # [6] 위험 표시 기준 (검증 기간 예측으로만 결정)
    p_val, y_val = preds_val[best_name], val[TARGET].values
    tab300, best300 = threshold_table(y_val, p_val, COST_MISS)
    tab600, best600 = threshold_table(y_val, p_val, COST_MISS_ALT)
    thr = tab300.merge(tab600[["threshold", "cost"]], on="threshold", suffixes=("_300", "_600"))
    thr.to_csv(OUT / "threshold_costs.csv", index=False, encoding="utf-8-sig")
    no_model_cost = {c: int(y_val.sum()) * c for c in (COST_MISS, COST_MISS_ALT)}
    all_check_cost = len(y_val) * COST_CHECK
    print(f"\n[6] 놓침 300만원 -> 기준 {best300.threshold:.2f}, 표시 {best300.flagged}건, "
          f"놓침 {best300.missed}건, 비용 {best300.cost:,}만원")
    print(f"    놓침 600만원 -> 기준 {best600.threshold:.2f}, 표시 {best600.flagged}건, "
          f"놓침 {best600.missed}건, 비용 {best600.cost:,}만원")
    print(f"    (참고: 아무것도 점검 안 함 300={no_model_cost[COST_MISS]:,} / 600={no_model_cost[COST_MISS_ALT]:,}, "
          f"전부 점검 {all_check_cost:,}만원)")

    # [7] 최종 모델: 학습 데이터 전체(2023-01 ~ 2024-12)로 다시 학습
    final = make_models()[best_name]
    final.fit(train[FEATURES], train[TARGET])
    test["prob"] = final.predict_proba(test[FEATURES])[:, 1]
    sub = test[["equipment_id", "maintenance_date", "prob"]].copy()
    sub["maintenance_date"] = sub.maintenance_date.dt.strftime("%Y-%m-%d")
    sub["prob"] = sub.prob.round(6)
    assert len(sub) == 1241 and sub.prob.between(0, 1).all()
    sub.to_csv(OUT / "test_predictions_2025.csv", index=False, encoding="utf-8-sig")
    print(f"\n[7] 시험 예측 {len(sub)}건 -> outputs/test_predictions_2025.csv "
          f"(확률 평균 {sub.prob.mean():.3f}, 기준 {best300.threshold:.2f} 이상 {int((sub.prob >= best300.threshold).sum())}건)")

    # [8] 2026-01 재점검 20대
    month["prob"] = final.predict_proba(month[FEATURES])[:, 1]
    latest = month.sort_values(["maintenance_date", "row_id"]).groupby("equipment_id").tail(1)
    latest = latest.sort_values("prob", ascending=False).reset_index(drop=True)
    latest[["equipment_id", "equipment_type", "unit_code", "maintenance_date", "maintenance_type",
            "fault_category", "prob"]].assign(
        maintenance_date=latest.maintenance_date.dt.strftime("%Y-%m-%d"), prob=latest.prob.round(6),
        flag_300=(latest.prob >= best300.threshold).astype(int),
        flag_600=(latest.prob >= best600.threshold).astype(int)).to_csv(
        OUT / "this_month_predictions.csv", index=False, encoding="utf-8-sig")
    over300 = int((latest.prob >= best300.threshold).sum())
    over600 = int((latest.prob >= best600.threshold).sum())
    top = latest.head(K).copy()
    reasons, deltas = explain_rows(final, top[FEATURES], train[FEATURES])
    top["위험을 올린 요인"] = reasons
    # 권고 = (가장 크게 위험을 올린 요인에 맞는 조치) + (이번 고장 유형별 점검)
    top_factor = deltas.idxmax(axis=1)
    action = top_factor.map(lambda g: next((v for k, v in FACTOR_ACTION.items() if g in k), ""))
    top["권고"] = [f"{a} " if a else "" for a in action] + top.fault_category.map(RECOMMEND)
    rep = top.equipment_id.isin(REPEAT_DRONES)
    top.loc[rep, "권고"] = "반복 고장 기체: 근본 원인 조사 · 운용 중지 검토. " + top.loc[rep, "권고"]
    plan = pd.DataFrame({
        "순위": np.arange(1, len(top) + 1), "드론": top.equipment_id, "유형": top.equipment_type,
        "부대": top.unit_code, "최근 정비일": top.maintenance_date.dt.strftime("%Y-%m-%d"),
        "정비 구분": top.maintenance_type, "고장 유형": top.fault_category,
        "확률": top.prob.round(3), "기준 이상": np.where(top.prob >= best300.threshold, "예", "아니오"),
        "위험을 올린 요인": top["위험을 올린 요인"], "권고": top["권고"]})
    plan.to_csv(OUT / "recheck_plan_2026_01.csv", index=False, encoding="utf-8-sig")
    overflow = latest.iloc[K:][latest.iloc[K:].prob >= best300.threshold]
    print(f"\n[8] 2026-01 정비 드론 {len(latest)}대 중 기준({best300.threshold:.2f}) 이상 {over300}대, "
          f"20대 초과 {len(overflow)}대 -> outputs/recheck_plan_2026_01.csv")
    print(plan[["순위", "드론", "유형", "부대", "확률", "위험을 올린 요인"]].to_string(index=False))

    write_report(df, fit, val, metrics, monthly, best_name, best300, best600, no_model_cost,
                 all_check_cost, sub, latest, plan, overflow, over300, over600, unc, ceiling, train)


def write_report(df, fit, val, metrics, monthly, best_name, b300, b600, no_model_cost, all_check_cost,
                 sub, latest, plan, overflow, over300, over600, unc, ceiling, train):
    cl = pd.read_csv(OUT / "cleaning_log.csv", encoding="utf-8-sig")
    best = metrics[metrics.model == best_name].iloc[0]
    rnd = metrics[metrics.model.str.contains("무작위")].iloc[0]
    age = metrics[metrics.model.str.contains("경과연수")].iloc[0]
    mh = pd.DataFrame(monthly)[[best_name, "기준선_경과연수순"]]
    L = []
    w = L.append
    tr_all = df[df.split == "train"]
    cnt = df.groupby("equipment_id").size()
    u = unc.set_index("항목")
    u_best = u.loc[f"{best_name} PR-AUC 95% 구간 (부트스트랩 1000회)"]
    u_diff = u.loc[f"PR-AUC 차이 ({best_name} - 경과연수순) 95% 구간"]
    u_seed = u.loc[f"{best_name} PR-AUC 랜덤 시드 0~4"]
    purge_start = VALID_START - pd.Timedelta(days=PURGE_DAYS)
    hits_total, ceil_total = best.top20_hits_total, sum(ceiling.values())

    w("# 드론 예측정비 분석 보고서\n")
    w("> 모든 숫자는 `drone_pm.py` 실행 결과이며, 괄호 안에 출처 파일을 적었다. 교육용 가상 데이터. "
      "재점검 계획은 별도 문서 `outputs/recheck_plan_2026_01.md`.\n")

    # ── 요약 ────────────────────────────────────────────────
    w("## 요약\n")
    w(f"- **목적**: 정비를 마친 드론 중 30일 안에 다시 고장 날 드론을 골라 2026-01 재점검 {K}대를 정한다.")
    w(f"- **모델**: {best_name}. 검증(2024-07~12) PR-AUC **{best.PR_AUC:.3f}** "
      f"(95% 구간 {u_best.하한:.3f}~{u_best.상한:.3f}), 경과연수순 {age.PR_AUC:.3f} · 무작위 {rnd.PR_AUC:.3f}.")
    w(f"- **월 {K}대 적중**: 6개월 합 **{hits_total:g}건** / 가능한 최대 {ceil_total}건 "
      f"(경과연수순 {age.top20_hits_total:g}건, 무작위 {rnd.top20_hits_total:g}건).")
    w(f"- **위험 표시 기준**: 확률 **{b300.threshold:.2f}** 이상 (놓침 600만원이면 {b600.threshold:.2f}).")
    w(f"- **주의**: 반복 고장 드론({', '.join(REPEAT_DRONES)})을 빼면 PR-AUC가 {best.PR_AUC_excl_repeat:.3f}로 내려간다. "
      "그래도 기준선보다 높다.\n")

    w("## 용어\n")
    w("| 용어 | 뜻 |\n|---|---|")
    w(f"| 재고장 | 정비 뒤 30일 안에 사후 · 긴급정비가 있었음 (`{TARGET}` = 1) |")
    w("| PR-AUC | 확률이 높은 순으로 골랐을 때 실제 재고장을 얼마나 앞쪽에 모았는지 (0~1). 아무렇게나 고르면 재고장 비율과 같다 |")
    w(f"| 월 {K}대 적중 | 매달 드론별 최근 정비 기록을 확률 순으로 {K}대 골랐을 때 실제 재고장한 드론 수 |")
    w("| 기준선 | 모델 없이 고르는 방법(무작위, 경과연수 오래된 순). 모델은 이보다 나아야 의미가 있다 |")
    w("| %p | 확률의 차이(퍼센트포인트). 0.30 → 0.40이면 +10%p |\n")

    # ── 1. 데이터 살펴보기 ──────────────────────────────────
    w("## 1. 데이터 살펴보기 (outputs/eda_summary.txt)\n")
    w("| 파일 | 기간 | 건수 | 드론 | 정답 |\n|---|---|---|---|---|")
    for s, name in (("train", "drone_train.csv"), ("test", "drone_test.csv"), ("this_month", "drone_this_month.csv")):
        g = df[df.split == s]
        period = (f"{g.maintenance_date.min():%Y-%m}" if g.maintenance_date.min().to_period("M") == g.maintenance_date.max().to_period("M")
                  else f"{g.maintenance_date.min():%Y-%m} ~ {g.maintenance_date.max():%Y-%m}")
        w(f"| {name} | {period} | {len(g):,} | "
          f"{g.equipment_id.nunique()} | {'있음' if s == 'train' else '없음'} |")
    w(f"\n학습 데이터 재고장률 **{tr_all[TARGET].mean():.1%}** ({int(tr_all[TARGET].sum()):,}/{len(tr_all):,}건).\n")
    w("**수치 컬럼 분포 (학습 데이터, 2장 처리 후)**\n")
    w("| 컬럼 | 최솟값 | 중앙값 | 99% | 최댓값 | 빈 값 |\n|---|---|---|---|---|---|")
    for c in ["equipment_age_years", "cumulative_operating_hours", "days_since_last_maintenance", "severity",
              "downtime_hours", "repair_duration_hours", "parts_replaced_count", "maintenance_cost", "availability_rate"]:
        x = tr_all[c]
        w(f"| {FEATURE_KO.get(c, '그 달 가동률')} (`{c}`) | {x.min():g} | {x.median():g} | {x.quantile(.99):.4g} | "
          f"{x.max():g} | {int(x.isna().sum())} |")
    w("")
    w("**재고장률 패턴 (학습 데이터)**\n")
    season = tr_all.maintenance_date.dt.month.map({12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
                                                   6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"})
    groups = [("정비 구분", tr_all.maintenance_type), ("드론 유형", tr_all.equipment_type),
              ("고장 유형", tr_all.fault_category), ("계절", season)]
    w("| 구분 | 값별 재고장률 (건수) |\n|---|---|")
    for label, key in groups:
        r = tr_all.groupby(key)[TARGET].agg(["mean", "size"]).sort_values("mean", ascending=False)
        w(f"| {label} | " + " · ".join(f"{k} {v['mean']:.0%} ({int(v['size'])})" for k, v in r.iterrows()) + " |")
    mt = tr_all.groupby("maintenance_type")[TARGET].mean()
    sr = tr_all.groupby(season)[TARGET].mean()
    w(f"\n- 고장정비(긴급 {mt['긴급정비']:.0%} · 사후 {mt['사후정비']:.0%}) 뒤의 재고장률이 예방정비({mt['예방정비']:.0%})보다 "
      "훨씬 높다 → 이번 정비 구분과 최근 고장 이력이 핵심 신호.")
    w(f"- 계절은 {sr.idxmax()}({sr.max():.0%})이 가장 높고 {sr.idxmin()}({sr.min():.0%})이 가장 낮다 → 월(계절) 피처를 넣었다.")
    w(f"- 드론 유형별 차이도 있어(최고 {tr_all.groupby('equipment_type')[TARGET].mean().idxmax()}) 유형을 피처로 넣었다.\n")
    w("**결측 · 이상치 · 눈여겨볼 점**\n")
    fault_na = {s: int((df[df.split == s].fault_code == "없음").sum()) for s in ("train", "test", "this_month")}
    w(f"- 고장 코드 · 유형 결측(학습 {fault_na['train']:,} / 시험 {fault_na['test']:,} / 이번 달 {fault_na['this_month']})은 "
      "모두 예방정비 → 고장이 없어 원래 값이 없는 것.")
    first = df[df.is_first_maintenance == 1]
    w(f"- 직전 정비 뒤 경과일 결측 {len(first)}건은 모두 드론별 첫 기록({first.maintenance_date.min():%Y-%m}~"
      f"{first.maintenance_date.max():%Y-%m}, 학습 데이터에만 있음) → 직전 정비가 없어 원래 값이 없는 것.")
    w(f"- 운용중단 시간 999 {int(df.downtime_missing.sum())}건, 정비 비용 1만 이상 · 100의 배수 {int(df.cost_fixed.sum())}건 → 2장.")
    low = tr_all.availability_rate <= 0.3
    w(f"- 가동률 0.3 이하 {int(low.sum())}건의 재고장률 {tr_all.loc[low, TARGET].mean():.0%} (그 외 "
      f"{tr_all.loc[~low, TARGET].mean():.0%}) → 정비 뒤 정보가 섞인 값(4장).")
    for e in REPEAT_DRONES:
        g = tr_all[tr_all.equipment_id == e]
        w(f"- 반복 고장 드론 **{e}**: 기록 {cnt[e]}건(드론 중앙값 {cnt.median():.0f}건), 정비 간격 중앙값 "
          f"{g.days_since_last_maintenance.median():.0f}일, 학습 재고장률 {g[TARGET].mean():.0%}. 날짜 · 비행시간이 일관되어 "
          "입력 오류가 아닌 실제 반복 고장 기체로 판단해 지우지 않았다.")
    tr_max = tr_all.equipment_age_years.max()
    w(f"- 경과연수 중앙값: 학습 {tr_all.equipment_age_years.median():.1f}년 → 시험 "
      f"{df[df.split == 'test'].equipment_age_years.median():.1f}년 → 이번 달 "
      f"{df[df.split == 'this_month'].equipment_age_years.median():.1f}년. 드론이 점점 오래되고 있다(모델의 한계 참고).\n")

    # ── 2. 데이터 처리 ──────────────────────────────────────
    w("## 2. 데이터 처리 기준과 근거 (outputs/cleaning_log.csv)\n")
    w("| 대상 | 판단 | 처리 | 하지 않은 것 | 건수 |\n|---|---|---|---|---|")
    n = cl.groupby("column").size().to_dict()
    normal = df[df.cost_fixed == 0]
    fixed = df[df.cost_fixed == 1]
    big_raw = cl[cl.column == "maintenance_cost"].before
    w(f"| downtime_hours = 999 | 결측 코드 (그 외 최댓값 {df.downtime_hours.max():g}시간) | 비워 두고 `downtime_missing = 1` "
      f"| 다른 행 값으로 추정해 채우지 않음 | {n.get('downtime_hours', 0)} |")
    w(f"| maintenance_cost ≥ 10,000 · 100의 배수 | 100배 단위 오입력. 정상 최댓값 {int(normal.maintenance_cost.max())}과 "
      f"{int(big_raw.min()):,} 사이에 값 없음. 정상값 중 100의 배수는 "
      f"{sorted(normal.loc[normal.maintenance_cost % 100 == 0, 'maintenance_cost'].astype(int).unique().tolist())}뿐 "
      f"| ÷100 → {int(fixed.maintenance_cost.min())}~{int(fixed.maintenance_cost.max())} (사후정비 정상 범위), `cost_fixed = 1` "
      f"| 행 삭제 | {n.get('maintenance_cost', 0)} |")
    w(f"| fault_code · fault_category 결측 | 예방정비라 원래 값 없음 | '없음' 범주 | — | "
      f"{fault_na['train'] + fault_na['test'] + fault_na['this_month']:,} |")
    med = tr_all.days_since_last_maintenance.median()
    w(f"| 직전 정비 뒤 경과일 · 최근 하루 비행시간 결측 | 드론별 첫 기록, 직전 정비 없음 (이유가 있는 결측) "
      f"| 비워 두고 `is_first_maintenance = 1` | 0('방금 정비함'), 중앙값('{med:.0f}일 전 정비')으로 채우지 않음 | {len(first)} |")
    w(f"| 예방정비 뒤 경과일 · 비행시간 결측 | 예방정비 이력 없음 (이유가 있는 결측) | 비워 둠 | 0으로 채우지 않음 "
      f"| {int(df.days_since_last_pm.isna().sum())} |")
    w(f"| availability_rate | 정비 뒤 그 달 정보가 섞임 (0.3 이하 재고장률 {tr_all.loc[low, TARGET].mean():.0%}) "
      "| 컬럼 제외 (4장) | — | 전체 |")
    w("\n- **삭제한 행은 없다.** 고친 값은 표시 컬럼으로 남겨 추적할 수 있다.")
    w("- 빈 값은 트리 모델(랜덤포레스트 · 그래디언트부스팅)이 그대로 학습한다. 빈 값을 다루지 못하는 로지스틱회귀만 "
      "모델 안에서 학습 데이터 중앙값으로 채운다(검증 · 시험 값은 쓰지 않음).\n")

    # ── 3. 피처 ─────────────────────────────────────────────
    w("## 3. 피처 만들기\n")
    w("학습 → 시험 → 이번 달 기록을 드론별로 이어 붙여, **그 정비일까지의 기록만으로** 계산했다. "
      "그 뒤 기록을 지우고 다시 계산해도 같은 값인지 `reports/verify_results.py`(V4)로 확인했다.\n")
    w("| 피처 | 뜻 | 왜 |\n|---|---|---|")
    for c, desc, why in FEATURE_DOC:
        name = "정비 월(계절) (`month_sin`, `month_cos`)" if c == "month_sin" else f"{FEATURE_KO.get(c, c)} (`{c}`)"
        w(f"| {name} | {desc} | {why} |")
    w(f"\n원본 컬럼과 표시 컬럼을 합쳐 모두 {len(FEATURES)}개 (범주 {len(CAT_FEATURES)}, 수치 {len(NUM_FEATURES)}).\n")

    # ── 4. 컬럼 선택 ────────────────────────────────────────
    w("## 4. 쓸 컬럼 고르기\n")
    w("**넣은 원본 컬럼**: 드론 유형 · 부대 · 경과연수 · 누적 비행시간 · 직전 정비 뒤 경과일, 그리고 이번 정비의 구분 · 고장 유형 · "
      "심각도 · 운용중단 시간 · 수리 시간 · 교체 부품 수 · 비용 · 성공 여부. 모두 **정비를 마친 직후 정비 기록에 적혀 있는 값**이라 "
      "예측 시점에 알 수 있다.\n")
    w("**넣지 않은 컬럼**\n")
    w("| 컬럼 | 이유 |\n|---|---|")
    for k, v in EXCLUDED.items():
        w(f"| {k} | {v} |")
    w("")

    # ── 5. 기간 분할과 성능 ─────────────────────────────────
    w("## 5. 기간 분할과 성능 (outputs/validation_metrics.csv, validation_monthly_hits.csv, validation_uncertainty.csv)\n")
    w("| 구간 | 기간 | 건수 | 용도 |\n|---|---|---|---|")
    w(f"| 학습 | {fit.maintenance_date.min():%Y-%m-%d} ~ {fit.maintenance_date.max():%Y-%m-%d} | {len(fit):,} | 모델 학습 |")
    w(f"| 제외 | {purge_start:%Y-%m-%d} ~ {VALID_START - pd.Timedelta(days=1):%Y-%m-%d} "
      f"| {len(train) - len(fit) - len(val)} | 정답 창({PURGE_DAYS}일)이 검증 기간과 겹쳐 뺌 |")
    w(f"| 검증 | {val.maintenance_date.min():%Y-%m-%d} ~ {val.maintenance_date.max():%Y-%m-%d} | {len(val):,} "
      f"(재고장 {int(val[TARGET].sum())}) | 모델 선택 · 위험 기준 결정 |")
    w(f"| 시험 | 2025-01 ~ 2025-12 | {len(sub):,} | 채점용. 기준 결정에 쓰지 않음 |\n")
    w("무작위로 나누지 않고 기간으로 나눴다. 최종 모델은 학습 데이터 전체(2023-01~2024-12)로 다시 학습했다.\n")
    w("| 모델 | PR-AUC | ROC-AUC | 월 20대 적중 합 | PR-AUC (반복 고장 드론 제외) |\n|---|---|---|---|---|")
    for _, r in metrics.iterrows():
        w(f"| {r.model} | {r.PR_AUC:.3f} | {r.ROC_AUC:.3f} | {r.top20_hits_total:g} | {r.PR_AUC_excl_repeat:.3f} |")
    w(f"\n- 선택 모델: **{best_name}** (검증 PR-AUC 최고). 무작위 대비 PR-AUC {best.PR_AUC / rnd.PR_AUC:.1f}배, "
      f"경과연수순 대비 적중 {best.top20_hits_total - age.top20_hits_total:+.1f}건.")
    w(f"- **불확실성**: 검증 표본을 다시 뽑으면 PR-AUC 95% 구간 {u_best.하한:.3f}~{u_best.상한:.3f}. "
      f"경과연수순과의 차이도 95% 구간({u_diff.하한:.3f}~{u_diff.상한:.3f})이 0보다 확실히 크다. "
      f"랜덤 시드만 바꾸면 {u_seed.하한:.4f}~{u_seed.상한:.4f}(폭 {u_seed.표준오차_또는_폭:.4f}) 흔들린다.")
    w(f"- **반복 고장 드론**({', '.join(REPEAT_DRONES)})은 재고장이 너무 잦아 맞히기 쉽다. 빼면 PR-AUC가 "
      f"{best.PR_AUC_excl_repeat:.3f}로 내려가지만 경과연수순({age.PR_AUC_excl_repeat:.3f}) · 무작위"
      f"({rnd.PR_AUC_excl_repeat:.3f})보다 높다. 월 20대 적중은 드론당 1건으로 세므로 영향이 작다.")
    w("- 정확도는 재고장이 적은 데이터에서 착시가 커서 쓰지 않았다.")
    w("- 모델 설정값(나무 수, 깊이 등)은 미리 정한 보수적인 값으로 고정했고 검증 점수로 고르지 않았다. "
      "검증 기간을 모델 선택 · 위험 기준에 이미 쓰고 있어, 설정값까지 맞추면 검증 성능이 더 부풀기 때문이다.\n")
    w(f"**월별 {K}대 적중**\n\n| 월 | {best_name} | 경과연수순 | 가능한 최대 |\n|---|---|---|---|")
    for ym in mh.index:
        w(f"| {ym} | {mh.at[ym, best_name]:g} | {mh.at[ym, '기준선_경과연수순']:.1f} | {ceiling[ym]} |")
    w(f"| **합** | **{hits_total:g}** | {age.top20_hits_total:.1f} | {ceil_total} |")
    w(f"\n가능한 최대 = 그 달 실제로 재고장한 드론 수(최대 {K}). 경과연수순은 같은 연수가 많아 동점을 무작위로 200번 깨서 "
      "평균을 냈기 때문에 소수가 나온다.\n")

    # ── 6. 위험 표시 기준 ───────────────────────────────────
    w("## 6. 위험 표시 기준 (outputs/threshold_costs.csv, 검증 기간에서만 결정)\n")
    w(f"비용 = 놓친 재고장 수 × 놓침 비용 + 기준 이상 건수 × {COST_CHECK}만원. 확률 0.01~0.99를 0.01 간격으로 모두 계산해 "
      "가장 싼 기준을 골랐다.\n")
    w("| 놓침 비용 | 기준 확률 | 표시 건수 | 잡은 재고장 | 놓친 재고장 | 총비용(만원) | 이론값 |\n|---|---|---|---|---|---|---|")
    for c, b in ((COST_MISS, b300), (COST_MISS_ALT, b600)):
        w(f"| {c}만원 | {b.threshold:.2f} | {b.flagged} | {b.caught} | {b.missed} | {b.cost:,} | {COST_CHECK / c:.2f} |")
    w(f"\n- **이론값**: 확률이 정확하다면 '점검비 < 확률 × 놓침 비용'일 때 점검하는 게 이득이므로 기준은 "
      f"점검비 ÷ 놓침 비용 = {COST_CHECK}/{COST_MISS} = {COST_CHECK / COST_MISS:.2f}, {COST_CHECK}/{COST_MISS_ALT} = "
      f"{COST_CHECK / COST_MISS_ALT:.2f}이다. 실제로 고른 기준({b300.threshold:.2f} · {b600.threshold:.2f})은 이론값과 가까워, "
      "모델의 확률이 크게 치우치지 않았다는 근거도 된다.")
    w(f"- 놓침 비용이 {COST_MISS} → {COST_MISS_ALT}만원으로 커지면 기준이 {b300.threshold:.2f} → {b600.threshold:.2f}로 낮아져 "
      f"더 많이 점검한다 (표시 {b300.flagged} → {b600.flagged}건).")
    w(f"- 참고: 아무것도 점검하지 않으면 {no_model_cost[COST_MISS]:,}만원({COST_MISS}) / {no_model_cost[COST_MISS_ALT]:,}만원"
      f"({COST_MISS_ALT}), 전부 점검하면 {all_check_cost:,}만원.\n")

    # ── 7. 시험 예측 ────────────────────────────────────────
    w("## 7. 시험 기간 예측 (outputs/test_predictions_2025.csv)\n")
    w(f"- 학습 데이터 전체로 다시 학습한 최종 모델({best_name})로 시험 {len(sub):,}건 모두 예측.")
    w(f"- 확률 평균 {sub.prob.mean():.3f} (검증 기간 재고장률 {val[TARGET].mean():.3f}), "
      f"기준({b300.threshold:.2f}) 이상 {int((sub.prob >= b300.threshold).sum())}건.")
    w("- 결과 검증: `reports/verify_results.py`, `reports/verify_notebook.ipynb` (결과 재현 · 누수 · 숫자 대조).\n")
    w("---\n2026-01 드론 재점검 계획과 모델의 한계는 별도 문서 `outputs/recheck_plan_2026_01.md`에 있다.")
    report_lines, L = L, []          # 여기부터는 재점검 계획 문서
    w = L.append
    w("# 2026-01 드론 재점검 계획\n")
    w("> 모든 숫자는 `drone_pm.py` 실행 결과이며, 괄호 안에 출처 파일을 적었다. 교육용 가상 데이터. "
      "데이터 처리 · 모델 · 기준을 정한 과정은 `outputs/report.md` 참고.\n")
    w(f"- 사용 모델: {best_name} (2023-01~2024-12 학습)")
    w(f"- 위험 표시 기준: 확률 {b300.threshold:.2f} 이상 (놓침 300만원 · 재점검 60만원 기준, 검증 기간에서 결정)")
    w(f"- 점검 여력: 월 {K}대\n")
    w("## 1. 재점검 대상 20대 (outputs/recheck_plan_2026_01.csv)\n")
    w(f"이번 달 정비한 드론 {len(latest)}대(드론별 최근 기록) 중 위험 표시 기준({b300.threshold:.2f}) 이상 "
      f"**{over300}대** (600만원 기준 {b600.threshold:.2f} 적용 시 {over600}대).\n")
    w("| 순위 | 드론 | 유형 | 부대 | 확률 | 위험을 올린 요인 | 권고 |\n|---|---|---|---|---|---|---|")
    for _, r in plan.iterrows():
        w(f"| {r['순위']} | {r['드론']} | {r['유형']} | {r['부대']} | {r['확률']:.3f} | {r['위험을 올린 요인']} | {r['권고']} |")
    w("\n위험을 올린 요인: 해당 값을 학습 데이터의 보통 값(중앙값·최빈값)으로 바꿨을 때 확률이 얼마나 내려가는지(%p).\n")
    exp_fail = plan["확률"].sum()
    w(f"**기대 효과**: 20대의 확률 합은 {exp_fail:.1f}로, 그대로 두면 약 {exp_fail:.0f}대가 30일 안에 다시 고장 날 것으로 "
      f"예상된다. 재점검으로 이를 막는다면 놓침 비용 약 {exp_fail * COST_MISS:,.0f}만원을 줄이고 점검비 "
      f"{K * COST_CHECK:,}만원이 든다 (README 비용 가정, 확률이 정확하다고 볼 때의 기대값).\n")
    if len(latest) > K:
        a, b = latest.iloc[K - 1], latest.iloc[K]
        gap = a.prob - b.prob
        if gap < BOUNDARY_GAP:
            w(f"**20위 경계 주의**: {K}위 {a.equipment_id}({a.equipment_type}, {a.unit_code}) 확률 {a.prob:.4f}, "
              f"{K + 1}위 {b.equipment_id}({b.equipment_type}, {b.unit_code}, 이번 정비 {b.maintenance_type}) 확률 {b.prob:.4f}. "
              f"차이 {gap:.4f}는 매우 작아(검증 PR-AUC도 랜덤 시드만 바꾸면 {u_seed.표준오차_또는_폭:.4f} 흔들림) "
              "경계의 순위는 사실상 같다. 정비 담당자가 현장 상태를 보고 바꿀 수 있다. "
              "(출처: outputs/this_month_predictions.csv, validation_uncertainty.csv)\n")
    if len(overflow):
        w(f"**20대를 넘는 기준 이상 드론 {len(overflow)}대**: "
          + ", ".join(f"{r.equipment_id}({r.prob:.2f})" for r in overflow.itertuples())
          + "\n\n처리 방안\n"
            "1. 다음 달 초 우선 점검 대상으로 넘긴다 (확률 순).\n"
            "2. 그때까지 비행 전 점검을 강화하고 고위험 임무에서 뺀다.\n"
            "3. 같은 부대 정비반 협조가 가능하면 추가 점검한다. 부대별 초과 대수: "
          + ", ".join(f"{u} {c}대" for u, c in overflow.unit_code.value_counts().items()) + "\n"
          + f"4. 놓침 비용을 {COST_MISS_ALT}만원으로 본다면 기준 이상이 {over600}대로 늘어나므로, "
            "점검 여력 확대를 검토한다.\n")
    else:
        w("기준 이상 드론이 20대 이하이므로 모두 이번 달에 재점검한다.\n")
    w("## 2. 모델의 한계\n")
    w("- 교육용 가상 데이터로 만든 모델이며, 확률은 과거 패턴에 근거한 추정치다.")
    w("- 비용 기준은 재점검하면 재고장을 막는다는 가정 위에서 계산했다.")
    w("- 위험 표시 기준은 검증 모델로 정했고 최종 모델은 더 많은 데이터로 다시 학습해 확률 분포가 조금 다를 수 있다.")
    w("- 모델 선택과 위험 표시 기준을 같은 검증 기간(2024-07~12)에서 정했으므로 검증 성능은 다소 낙관적일 수 있다.")
    w(f"- 비용 기준은 월 {K}대 점검 여력을 따지지 않는다. 검증 기간 기준 이상은 {b300.flagged}건"
      f"(월 평균 {b300.flagged / 6:.0f}건)으로 여력을 넘으므로, 실제로는 확률 순으로 {K}대를 먼저 점검한다.")
    tr_max = tr_all.equipment_age_years.max()
    old_test = (df[df.split == "test"].equipment_age_years > tr_max).mean()
    old_month = (df[df.split == "this_month"].equipment_age_years > tr_max).mean()
    w(f"- 시험 · 이번 달 드론은 학습 때보다 오래되었다. 경과연수가 학습 최댓값({tr_max}년)을 넘는 기록이 "
      f"시험 {old_test:.0%}, 이번 달 {old_month:.0%}다. 트리 모델은 학습 범위 밖 값을 최댓값처럼 다루므로 "
      "오래된 드론의 확률은 덜 정확할 수 있다.")
    w("- 요인 설명은 상관관계이며 고장 원인을 뜻하지 않는다.")
    w("- **최종 재점검 여부는 정비 담당자가 현장 상태·임무 일정을 함께 보고 판단한다.**")
    (OUT / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (OUT / "recheck_plan_2026_01.md").write_text("\n".join(L), encoding="utf-8")
    print("\n분석 보고서 -> outputs/report.md")
    print("재점검 계획 -> outputs/recheck_plan_2026_01.md")


if __name__ == "__main__":
    main()
