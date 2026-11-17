"""드론 예측정비 실습 - 전체 파이프라인 (README.txt의 [1]~[8] 단계)

실행:  .venv\\Scripts\\python drone_pm.py
결과:  outputs/ 폴더
  - eda_summary.txt              [1] 데이터 살펴보기 결과
  - cleaning_log.csv             [2] 이상치·결측 처리 내역(행 단위)
  - validation_metrics.csv       [5] 검증 성능(모델·기준선)
  - validation_monthly_hits.csv  [5] 검증 기간 월별 20대 적중 수
  - threshold_costs.csv          [6] 확률 기준별 비용(300/600만원)
  - test_predictions_2025.csv    [7] 제출용 시험 기간 예측
  - recheck_plan_2026_01.csv     [8] 2026-01 재점검 대상 20대
  - report.md                    보고서 초안(모든 숫자는 이 스크립트 실행 결과)
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    log(f"  downtime_hours == 999: {(df.downtime_hours >= 999).groupby(df.split).sum().to_dict()}"
        f"  (그 외 최댓값 {df.loc[df.downtime_hours < 999, 'downtime_hours'].max()})")
    big = df.maintenance_cost > 2000
    log(f"  maintenance_cost > 2000: {big.groupby(df.split).sum().to_dict()}, 값 {sorted(df.loc[big, 'maintenance_cost'].tolist())}"
        f"  (그 외 최댓값 {df.loc[~big, 'maintenance_cost'].max()})")
    low = tr.availability_rate <= 0.3
    log(f"  availability_rate <= 0.3: {int(low.sum())}건, 이들의 재고장률 {tr.loc[low, TARGET].mean():.2f}"
        f" (그 외 {tr.loc[~low, TARGET].mean():.2f}) -> 정비 뒤 그 달의 고장이 반영된 값(누수)")

    # 일관성 점검
    s = df.sort_values(["equipment_id", "maintenance_date"])
    gap = s.groupby("equipment_id").maintenance_date.diff().dt.days
    log(f"\n일관성: 날짜 차이와 days_since_last 불일치 {int(((gap - s.days_since_last_maintenance).abs() > 0).sum())}건, "
        f"누적비행시간 감소 {int((s.groupby('equipment_id').cumulative_operating_hours.diff() < 0).sum())}건, "
        f"동일 드론-날짜 중복 {int(df.duplicated(['equipment_id', 'maintenance_date']).sum())}건")

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

    # (a) downtime_hours 999 -> 결측 코드. 같은 정비 구분의 '운용중단/수리시간' 중앙 비율로 채움
    bad = df.downtime_hours >= 999
    df["downtime_imputed"] = bad.astype(int)
    ok = ~bad & (df.repair_duration_hours > 0)
    ratio = (df.loc[ok, "downtime_hours"] / df.loc[ok, "repair_duration_hours"]).groupby(
        df.loc[ok, "maintenance_type"]).median()
    new = (df.loc[bad, "repair_duration_hours"] * df.loc[bad, "maintenance_type"].map(ratio)).round(1)
    for i in df.index[bad]:
        logs.append((df.at[i, "split"], df.at[i, "equipment_id"], df.at[i, "maintenance_date"].date(),
                     "downtime_hours", df.at[i, "downtime_hours"], new[i],
                     "999는 결측 코드. 수리시간 x 정비구분별 중앙비율로 대체"))
    df.loc[bad, "downtime_hours"] = new

    # (b) maintenance_cost > 2000 -> 단위 오입력(원래 값 x100). 100으로 나누면 사후정비 정상 범위
    bad = df.maintenance_cost > 2000
    df["cost_fixed"] = bad.astype(int)
    for i in df.index[bad]:
        logs.append((df.at[i, "split"], df.at[i, "equipment_id"], df.at[i, "maintenance_date"].date(),
                     "maintenance_cost", df.at[i, "maintenance_cost"], df.at[i, "maintenance_cost"] / 100,
                     "다른 값과 100배 차이. 단위 오입력으로 보고 /100"))
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

    # 첫 기록 / 예방정비 이력 없음 -> 표시 + 대체(학습 데이터 중앙값)
    df["is_first_record"] = df.days_since_last_maintenance.isna().astype(int)
    tr = df.split == "train"
    for c in ["days_since_last_maintenance", "hours_per_day_recent",
              "days_since_last_pm", "hours_since_last_pm"]:
        df[c] = df[c].fillna(df.loc[tr, c].median())

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
    "downtime_imputed", "is_first_record", "is_failure",
    "fail_cnt_30d", "fail_cnt_90d", "fail_cnt_180d", "fail_cnt_365d", "maint_cnt_90d",
    "same_cat_fail_365d", "unsuccess_cnt_180d", "severity_sum_180d",
    "days_since_prev_failure", "days_since_last_pm", "hours_since_last_pm",
    "n_prior_records", "hours_per_day_recent", "month_sin", "month_cos",
]
FEATURES = CAT_FEATURES + NUM_FEATURES

EXCLUDED = {
    "availability_rate": "'그 달' 가동률이라 정비 뒤 같은 달의 고장·중단이 섞임(<=0.3이면 재고장률 100%). "
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
    "downtime_imputed": "운용중단 시간 대체 여부", "is_first_record": "첫 정비 기록",
    "is_failure": "이번 정비가 고장정비", "fail_cnt_30d": "최근 30일 고장 수",
    "fail_cnt_90d": "최근 90일 고장 수", "fail_cnt_180d": "최근 180일 고장 수",
    "fail_cnt_365d": "최근 1년 고장 수", "maint_cnt_90d": "최근 90일 정비 횟수",
    "same_cat_fail_365d": "1년 내 같은 유형 고장 수", "unsuccess_cnt_180d": "최근 180일 정비 실패 수",
    "severity_sum_180d": "최근 180일 심각도 합", "days_since_prev_failure": "직전 고장 뒤 경과일",
    "days_since_last_pm": "예방정비 뒤 경과일", "hours_since_last_pm": "예방정비 뒤 비행시간",
    "n_prior_records": "과거 정비 기록 수", "hours_per_day_recent": "최근 하루 비행시간",
    "month": "정비 월(계절)",
}


# ----------------------------------------------------------------------------
# [5] 모델과 평가 함수
# ----------------------------------------------------------------------------
def make_models():
    ohe = lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return {
        "로지스틱회귀": Pipeline([
            ("prep", ColumnTransformer([("cat", ohe(), CAT_FEATURES),
                                        ("num", StandardScaler(), NUM_FEATURES)])),
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
    return {"model": name,
            "PR_AUC": round(average_precision_score(d[TARGET], score), 4),
            "ROC_AUC": round(roc_auc_score(d[TARGET], score), 4),
            "top20_hits_total": round(sum(hits.values()), 2),
            "positives_total": int(d[TARGET].sum())}, hits


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
                    ("is_failure", "maintenance_success", "downtime_imputed", "is_first_record")
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
                val = f"{round(float(val), 1):g}" if isinstance(val, (int, float, np.number)) else str(val)
            parts.append(f"{FEATURE_KO[g]}={val}(+{v * 100:.1f}%p)")
        texts.append("; ".join(parts) if parts else "뚜렷한 단일 요인 없음")
    return texts, deltas


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

    results, monthly = [], {}
    rng = np.random.default_rng(SEED)
    # 기준선 1: 무작위 (1000회 평균)
    r = [evaluate("기준선_무작위", val, rng.random(len(val)))[0] for _ in range(1000)]
    base_rand = {"model": "기준선_무작위(1000회 평균)",
                 "PR_AUC": round(np.mean([x["PR_AUC"] for x in r]), 4),
                 "ROC_AUC": round(np.mean([x["ROC_AUC"] for x in r]), 4),
                 "top20_hits_total": round(np.mean([x["top20_hits_total"] for x in r]), 2),
                 "positives_total": r[0]["positives_total"]}
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
    pd.DataFrame(monthly).to_csv(OUT / "validation_monthly_hits.csv", encoding="utf-8-sig")
    print(metrics.to_string(index=False))

    model_rows = metrics[~metrics.model.str.startswith("기준선")]
    best_name = model_rows.sort_values(["PR_AUC", "top20_hits_total"], ascending=False).iloc[0]["model"]
    print(f"-> 검증 PR-AUC 최고 모델: {best_name}")

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
    over300 = int((latest.prob >= best300.threshold).sum())
    over600 = int((latest.prob >= best600.threshold).sum())
    top = latest.head(K).copy()
    reasons, _ = explain_rows(final, top[FEATURES], train[FEATURES])
    top["위험을 올린 요인"] = reasons
    top["권고"] = top.fault_category.map(RECOMMEND)
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
                 all_check_cost, sub, latest, plan, overflow, over300, over600)


def write_report(df, fit, val, metrics, monthly, best_name, b300, b600, no_model_cost, all_check_cost,
                 sub, latest, plan, overflow, over300, over600):
    cl = pd.read_csv(OUT / "cleaning_log.csv", encoding="utf-8-sig")
    best = metrics[metrics.model == best_name].iloc[0]
    rnd = metrics[metrics.model.str.contains("무작위")].iloc[0]
    age = metrics[metrics.model.str.contains("경과연수")].iloc[0]
    mh = pd.DataFrame(monthly)[[best_name, "기준선_경과연수순"]]
    L = []
    w = L.append
    w("# 드론 예측정비 분석 보고서\n")
    w("> 모든 숫자는 `drone_pm.py` 실행 결과이며, 괄호 안에 출처 파일을 적었다. 교육용 가상 데이터.\n")
    w("## 1. 데이터 처리 기준과 근거 (outputs/eda_summary.txt, outputs/cleaning_log.csv)\n")
    w("| 대상 | 판단 | 처리 | 건수 |\n|---|---|---|---|")
    n = cl.groupby("column").size().to_dict()
    w(f"| downtime_hours = 999 | 다른 최댓값과 동떨어진 결측 코드(입력 실수) | 수리시간 × 정비구분별 중앙 비율로 대체, `downtime_imputed` 표시 | {n.get('downtime_hours', 0)} |")
    w(f"| maintenance_cost 15,000~19,000 | 100배 단위 오입력(나누면 사후정비 정상 범위) | ÷100, `cost_fixed` 표시 | {n.get('maintenance_cost', 0)} |")
    w("| fault_code·fault_category 결측 | 모두 예방정비 → 고장이 없어 원래 값 없음 | '없음' 범주 | 학습 1,707 / 시험 876 / 이번 달 62 |")
    w("| days_since_last_maintenance 결측 | 드론별 첫 기록 → 직전 정비 없음 | `is_first_record` 표시 + 학습 중앙값 | 150 |")
    w("| availability_rate ≤ 0.3 | 정비 뒤 고장이 반영된 값(재고장률 100%) | 컬럼 자체를 제외 | - |")
    w("\n**삭제한 행은 없다.** 이상치는 값만 고치고 표시 컬럼으로 남겨 추적할 수 있게 했다.\n")
    w("## 2. 넣지 않은 컬럼과 이유\n")
    w("| 컬럼 | 이유 |\n|---|---|")
    for k, v in EXCLUDED.items():
        w(f"| {k} | {v} |")
    w(f"\n사용 피처 {len(FEATURES)}개: 범주 {CAT_FEATURES}, 수치 {len(NUM_FEATURES)}개(과거 이력 피처 포함). "
      "이력 피처는 train → test → this_month 기록을 드론별로 이어 붙여 그 정비일까지의 기록만으로 계산했다.\n")
    w("## 3. 기간 분할과 성능 (outputs/validation_metrics.csv, outputs/validation_monthly_hits.csv)\n")
    w(f"- 학습: {fit.maintenance_date.min():%Y-%m-%d} ~ {fit.maintenance_date.max():%Y-%m-%d} ({len(fit)}건)")
    w(f"- 검증: {val.maintenance_date.min():%Y-%m-%d} ~ {val.maintenance_date.max():%Y-%m-%d} ({len(val)}건, 재고장 {int(val[TARGET].sum())}건)")
    w(f"- 2024-06의 {PURGE_DAYS}일은 정답 창(30일)이 검증 기간과 겹쳐 학습에서 뺐다.")
    w("- 월 20대 적중: 매달 드론별 최근 기록 기준 확률 상위 20대 중 실제 재고장 수(6개월 합).\n")
    w("| 모델 | PR-AUC | ROC-AUC | 월 20대 적중 합 |\n|---|---|---|---|")
    for _, r in metrics.iterrows():
        w(f"| {r.model} | {r.PR_AUC:.3f} | {r.ROC_AUC:.3f} | {r.top20_hits_total:g} |")
    w(f"\n선택 모델: **{best_name}** (검증 PR-AUC 최고). 무작위 대비 PR-AUC {best.PR_AUC / rnd.PR_AUC:.1f}배, "
      f"경과연수순 대비 적중 {best.top20_hits_total - age.top20_hits_total:+g}건.\n")
    w(f"월별 적중(선택 모델 / 경과연수순):\n\n| 월 | {best_name} | 경과연수순 |\n|---|---|---|")
    for ym, r in mh.iterrows():
        w(f"| {ym} | {r[best_name]:g} | {r['기준선_경과연수순']:g} |")
    w("")
    w("## 4. 위험 표시 기준 (outputs/threshold_costs.csv, 검증 기간에서만 결정)\n")
    w("비용 = 놓친 재고장 수 × 놓침 비용 + 기준 이상 건수 × 60만원\n")
    w("| 놓침 비용 | 기준 확률 | 표시 건수 | 잡은 재고장 | 놓친 재고장 | 총비용(만원) |\n|---|---|---|---|---|---|")
    for c, b in ((COST_MISS, b300), (COST_MISS_ALT, b600)):
        w(f"| {c}만원 | {b.threshold:.2f} | {b.flagged} | {b.caught} | {b.missed} | {b.cost:,} |")
    w(f"\n참고: 아무것도 점검하지 않으면 {no_model_cost[COST_MISS]:,}만원(300) / {no_model_cost[COST_MISS_ALT]:,}만원(600), "
      f"전부 점검하면 {all_check_cost:,}만원. 놓침 비용이 커지면 기준이 "
      f"{'낮아져' if b600.threshold < b300.threshold else '그대로거나 높아져'} 더 많이 점검하는 쪽이 유리해진다.\n")
    w("## 5. 시험 기간 예측 (outputs/test_predictions_2025.csv)\n")
    w(f"- 최종 모델은 학습 데이터 전체(2023-01~2024-12)로 다시 학습. 시험 {len(sub)}건 모두 예측.")
    w(f"- 확률 평균 {sub.prob.mean():.3f}, 기준({b300.threshold:.2f}) 이상 {int((sub.prob >= b300.threshold).sum())}건.\n")
    w("## 6. 2026-01 드론 재점검 계획 (outputs/recheck_plan_2026_01.csv)\n")
    w(f"이번 달 정비한 드론 {len(latest)}대(드론별 최근 기록) 중 위험 표시 기준({b300.threshold:.2f}) 이상 "
      f"**{over300}대** (600만원 기준 {b600.threshold:.2f} 적용 시 {over600}대).\n")
    w("| 순위 | 드론 | 유형 | 부대 | 확률 | 위험을 올린 요인 | 권고 |\n|---|---|---|---|---|---|---|")
    for _, r in plan.iterrows():
        w(f"| {r['순위']} | {r['드론']} | {r['유형']} | {r['부대']} | {r['확률']:.3f} | {r['위험을 올린 요인']} | {r['권고']} |")
    w("\n위험을 올린 요인: 해당 값을 학습 데이터의 보통 값(중앙값·최빈값)으로 바꿨을 때 확률이 얼마나 내려가는지(%p).\n")
    if len(overflow):
        w(f"**20대를 넘는 기준 이상 드론 {len(overflow)}대**: "
          + ", ".join(f"{r.equipment_id}({r.prob:.2f})" for r in overflow.itertuples())
          + "\n- 처리 방안: 다음 달 초 우선 점검 대상으로 넘기고, 그때까지 비행 전 점검 강화·고위험 임무 배제, "
            "같은 부대 정비반 협조 가능 시 추가 점검.\n")
    else:
        w("기준 이상 드론이 20대 이하이므로 모두 이번 달에 재점검한다.\n")
    w("## 7. 모델의 한계\n")
    w("- 교육용 가상 데이터로 만든 모델이며, 확률은 과거 패턴에 근거한 추정치다.")
    w("- 비용 기준은 재점검하면 재고장을 막는다는 가정 위에서 계산했다.")
    w("- 위험 표시 기준은 검증 모델로 정했고 최종 모델은 더 많은 데이터로 다시 학습해 확률 분포가 조금 다를 수 있다.")
    w("- 요인 설명은 상관관계이며 고장 원인을 뜻하지 않는다.")
    w("- **최종 재점검 여부는 정비 담당자가 현장 상태·임무 일정을 함께 보고 판단한다.**")
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n보고서 초안 -> outputs/report.md")


if __name__ == "__main__":
    main()
