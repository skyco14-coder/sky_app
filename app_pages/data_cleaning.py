"""[2]~[4] 데이터 처리 기준, 피처 만들기, 쓸 컬럼 고르기"""
import pandas as pd
import streamlit as st

from app_utils import require_output
from drone_pm import CAT_FEATURES, EXCLUDED, FEATURE_KO, NUM_FEATURES

st.title("데이터 처리 · 컬럼 선택")
st.caption("README [2]~[4] · 기준은 drone_pm.py의 clean() · build_features(), 처리 내역은 outputs/cleaning_log.csv")

# ── [2] 처리 기준 ───────────────────────────────────────────
st.header("데이터 처리 기준")
st.markdown("**삭제한 행은 없습니다.** 이상치는 값만 고치거나 '모름'으로 두고, 표시 컬럼을 남겨 추적할 수 있게 했습니다.")
rules = pd.DataFrame([
    ("downtime_hours = 999", "결측 코드 (입력 실수)",
     "값을 지어내지 않고 '모름'(빈 값)으로 둠 + `downtime_missing` 표시. 로지스틱회귀만 모델 안에서 학습 데이터 중앙값으로 채움"),
    ("maintenance_cost ≥ 10,000이면서 100의 배수", "100배 단위 오입력 (정상 최댓값 611과 15,000 사이 값 없음)",
     "÷100 (150~190, 사후정비 정상 범위) + `cost_fixed` 표시"),
    ("반복 고장 드론 (DR-0034, DR-0115)", "입력 오류 아님 (날짜 · 비행시간 일관)", "지우지 않음. PR-AUC를 이 드론 빼고도 따로 계산"),
    ("fault_code · fault_category 결측", "원래 값 없음 (모두 예방정비)", "'없음' 범주"),
    ("days_since_last_maintenance 결측", "원래 값 없음 (드론별 첫 기록)", "`is_first_record` 표시 + 학습 데이터 중앙값"),
    ("availability_rate", "정비 뒤 정보가 섞인 값 (누수)", "컬럼 자체를 모델에서 제외"),
], columns=["대상", "판단", "처리"])
st.dataframe(rules, hide_index=True, width="stretch")

log = require_output("cleaning_log.csv")
st.subheader(f"실제로 고친 행 ({len(log)}건)")
st.dataframe(log.rename(columns={"before": "처리 전", "after": "처리 후", "reason": "이유"}),
             hide_index=True, width="stretch")

# ── [3] 피처 ────────────────────────────────────────────────
st.header("피처 만들기")
st.markdown(
    "- 학습 → 시험 → 이번 달 기록을 **드론별로 이어 붙여**, 그 정비일까지의 기록만으로 계산합니다.\n"
    "- 그 정비일 뒤의 기록은 쓰지 않습니다. (검증: `reports/verify_results.py` V4 — 미래 기록을 지우고 다시 계산해도 같은 값)\n"
    "- 이번 정비 자체는 '정비 직후'에 알 수 있으므로 포함합니다.")
HISTORY = {
    "fail_cnt_30d": "현재 포함 30일 안의 사후 · 긴급정비 수",
    "fail_cnt_90d": "현재 포함 90일 안의 고장 수", "fail_cnt_180d": "현재 포함 180일 안의 고장 수",
    "fail_cnt_365d": "현재 포함 1년 안의 고장 수", "maint_cnt_90d": "90일 안의 정비 횟수(종류 무관)",
    "same_cat_fail_365d": "이번 고장과 같은 유형의 1년 안 고장 수",
    "unsuccess_cnt_180d": "180일 안의 정비 실패 수", "severity_sum_180d": "180일 안의 고장 심각도 합",
    "days_since_prev_failure": "직전 고장(이번 제외) 뒤 경과일, 없으면 730",
    "days_since_last_pm": "마지막 예방정비 뒤 경과일", "hours_since_last_pm": "마지막 예방정비 뒤 비행시간",
    "n_prior_records": "과거 정비 기록 수", "hours_per_day_recent": "직전 정비 뒤 하루 평균 비행시간",
    "prev_maintenance_type": "직전 정비 구분", "is_failure": "이번 정비가 사후 · 긴급정비인지",
    "month_sin": "정비 월(계절) — sin", "month_cos": "정비 월(계절) — cos",
}
feat = pd.DataFrame({
    "피처": CAT_FEATURES + NUM_FEATURES,
    "이름": [FEATURE_KO.get(c, FEATURE_KO["month"] if c.startswith("month") else c) for c in CAT_FEATURES + NUM_FEATURES],
    "종류": ["범주"] * len(CAT_FEATURES) + ["수치"] * len(NUM_FEATURES),
})
feat["구분"] = feat.피처.map(lambda c: "새로 만든 이력 · 계절 피처" if c in HISTORY else "원본 · 처리 표시")
feat["설명"] = feat.피처.map(HISTORY).fillna("")
c1, c2 = st.columns(2)
c1.metric("사용 피처", f"{len(feat)}개")
c2.metric("새로 만든 피처", f"{int((feat.구분 != '원본 · 처리 표시').sum())}개")
only_new = st.toggle("새로 만든 피처만 보기", value=True)
st.dataframe(feat[feat.구분 != "원본 · 처리 표시"] if only_new else feat, hide_index=True, width="stretch")

# ── [4] 넣지 않은 컬럼 ──────────────────────────────────────
st.header("넣지 않은 컬럼과 이유")
st.markdown("예측 시점(정비 직후)에 알 수 없는 값은 넣지 않습니다.")
st.dataframe(pd.DataFrame(list(EXCLUDED.items()), columns=["컬럼", "이유"]), hide_index=True, width="stretch")
