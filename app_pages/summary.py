"""요약 페이지: 분석 목적, 업무 조건, 데이터, 핵심 결과"""
import streamlit as st

from app_utils import (COST_CHECK, COST_MISS, COST_MISS_ALT, DATA_FILES, MONTHLY_CAPACITY,
                       best_threshold, read_data, require_output)

st.title("드론 예측정비 요약")
st.caption("※ 교육용 가상 데이터입니다. 실제 부대 · 기종 정보는 없습니다.")

# ── 분석 목적 ───────────────────────────────────────────────
st.header("분석 목적")
st.markdown(
    "정비를 마친 드론 가운데 **30일 안에 다시 고장 날 드론**을 미리 골라, "
    f"**이번 달(2026-01) 재점검할 드론 {MONTHLY_CAPACITY}대**를 정한다."
)
c1, c2, c3, c4 = st.columns(4)
c1.markdown("**사용자**  \n드론 정비반장")
c2.markdown("**예측 대상**  \n`failure_within_30days`  \n정비 1건마다 30일 안 재고장 확률을 예측  \n"
            "(30일 안에 사후 · 긴급정비가 있으면 1)")
c3.markdown("**예측 시점**  \n정비를 마친 직후")
c4.markdown(f"**결정**  \n드론별 이번 달 최근 정비의 확률로 재점검 {MONTHLY_CAPACITY}대 선정")

st.subheader("업무 조건")
k1, k2, k3 = st.columns(3)
k1.metric("월 재점검 가능", f"{MONTHLY_CAPACITY}대")
k2.metric("재고장 놓침 1건", f"{COST_MISS}만원", help=f"민감도 확인: {COST_MISS_ALT}만원")
k3.metric("재점검 1대", f"{COST_CHECK}만원")

# ── 제공 데이터 ─────────────────────────────────────────────
st.header("제공 데이터")
rows = []
for name, role in DATA_FILES.items():
    df = read_data(name)
    rows.append({
        "파일": name,
        "용도": role,
        "기간": f"{df.maintenance_date.min():%Y-%m} ~ {df.maintenance_date.max():%Y-%m}",
        "건수": len(df),
        "드론 수": df.equipment_id.nunique(),
    })
st.dataframe(rows, hide_index=True, width="stretch")
st.caption("한 행이 정비 1건이며, 같은 드론의 정비 기록이 날짜순으로 여러 행 있다. (출처: data/*.csv)")

# ── 핵심 결과 ───────────────────────────────────────────────
st.header("핵심 결과")
metrics = require_output("validation_metrics.csv")
costs = require_output("threshold_costs.csv")
month = require_output("this_month_predictions.csv")

models = metrics[~metrics.model.str.startswith("기준선")]
best = models.sort_values(["PR_AUC", "top20_hits_total"], ascending=False).iloc[0]
rand = metrics[metrics.model.str.contains("무작위")].iloc[0]
age = metrics[metrics.model.str.contains("경과연수")].iloc[0]
t300 = best_threshold(costs, "cost_300")
t600 = best_threshold(costs, "cost_600")
over300 = int(month.flag_300.sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"검증 PR-AUC ({best.model})", f"{best.PR_AUC:.3f}",
          f"{best.PR_AUC - age.PR_AUC:+.3f} vs 경과연수순")
m2.metric("월 20대 적중 (검증 6개월 합)", f"{best.top20_hits_total:g}건",
          f"{best.top20_hits_total - age.top20_hits_total:+.1f} vs 경과연수순")
m3.metric("위험 표시 기준 (300만원)", f"{t300.threshold:.2f}",
          f"600만원이면 {t600.threshold:.2f}", delta_color="off")
m4.metric("2026-01 기준 이상 드론", f"{over300}대",
          f"{over300 - MONTHLY_CAPACITY:+d}대 vs 점검 여력" if over300 != MONTHLY_CAPACITY else None,
          delta_color="inverse")

st.markdown("**기준선과 비교 (검증 기간 2024-07 ~ 12)**")
st.dataframe(
    metrics.rename(columns={"model": "모델", "PR_AUC": "PR-AUC", "ROC_AUC": "ROC-AUC",
                            "top20_hits_total": "월 20대 적중 합", "positives_total": "실제 재고장"}),
    hide_index=True, width="stretch")
st.caption("출처: outputs/validation_metrics.csv, threshold_costs.csv, this_month_predictions.csv "
           "(drone_pm.py 실행 결과)")

if over300 > MONTHLY_CAPACITY:
    st.warning(f"이번 달 위험 표시 기준을 넘은 드론이 {over300}대로, 점검 여력 {MONTHLY_CAPACITY}대보다 "
               f"{over300 - MONTHLY_CAPACITY}대 많습니다. 확률 순으로 {MONTHLY_CAPACITY}대를 먼저 점검합니다.")

# ── 바로가기 ────────────────────────────────────────────────
st.header("자세히 보기")
l1, l2, l3 = st.columns(3)
l1.page_link("app_pages/model_performance.py", label="모델 성능", icon="📈")
l2.page_link("app_pages/risk_threshold.py", label="위험 표시 기준", icon="⚖️")
l3.page_link("app_pages/recheck_plan.py", label="2026-01 재점검 계획", icon="✅")

st.divider()
st.caption("모델의 확률은 과거 패턴에 근거한 추정치입니다. "
           "**최종 재점검 여부는 정비 담당자가 현장 상태와 임무 일정을 함께 보고 판단합니다.**")
