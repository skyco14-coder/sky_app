"""[7]~[8] 2025 시험 예측과 2026-01 재점검 계획"""
import altair as alt
import streamlit as st

from app_utils import C_BASE, C_MAIN, MONTHLY_CAPACITY, OUT, best_threshold, require_output

st.title("재점검 계획 (2026-01)")
st.caption("README [7]~[8] · outputs/recheck_plan_2026_01.csv, this_month_predictions.csv, test_predictions_2025.csv")

plan = require_output("recheck_plan_2026_01.csv")
month = require_output("this_month_predictions.csv")
costs = require_output("threshold_costs.csv")
sub = require_output("test_predictions_2025.csv")
t300 = best_threshold(costs, "cost_300").threshold
t600 = best_threshold(costs, "cost_600").threshold

over = month[month.flag_300 == 1]
overflow = month.iloc[MONTHLY_CAPACITY:][month.iloc[MONTHLY_CAPACITY:].flag_300 == 1]

# ── 현황 ────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("이번 달 정비 드론", f"{len(month)}대", help="한 드론이 여러 번 정비했으면 가장 최근 기록으로 판단")
c2.metric(f"기준({t300:.2f}) 이상", f"{len(over)}대")
c3.metric("점검 여력", f"{MONTHLY_CAPACITY}대")
c4.metric("여력 초과", f"{len(overflow)}대", delta_color="inverse")
st.caption(f"놓침 비용을 600만원으로 보면 기준 {t600:.2f}, 기준 이상 {int(month.flag_600.sum())}대")

# ── 20대 ────────────────────────────────────────────────────
st.header(f"재점검 대상 {MONTHLY_CAPACITY}대")
show = plan[["순위", "드론", "유형", "부대", "확률", "위험을 올린 요인", "권고", "최근 정비일", "정비 구분", "고장 유형"]]
st.dataframe(show, hide_index=True, width="stretch",
             column_config={"확률": st.column_config.ProgressColumn("확률", min_value=0, max_value=1, format="%.3f"),
                            "위험을 올린 요인": st.column_config.TextColumn(width="large"),
                            "권고": st.column_config.TextColumn(width="large")})
st.caption("위험을 올린 요인: 그 값을 학습 데이터의 보통 값(중앙값 · 최빈값)으로 바꿨을 때 확률이 얼마나 내려가는지(%p). "
           "상관관계이며 고장 원인을 뜻하지 않습니다.")

# ── 분포 ────────────────────────────────────────────────────
st.header("이번 달 드론별 재고장 확률")
m = month.reset_index(drop=True).assign(순위=lambda d: d.index + 1)
m["구분"] = m.순위.map(lambda r: "재점검 20대" if r <= MONTHLY_CAPACITY else "그 외")
bars = alt.Chart(m).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
    x=alt.X("equipment_id:N", sort=None, title=None, axis=alt.Axis(labels=False, ticks=False)),
    y=alt.Y("prob:Q", title="재고장 확률", scale=alt.Scale(domain=[0, 1])),
    color=alt.Color("구분:N", scale=alt.Scale(domain=["재점검 20대", "그 외"], range=[C_MAIN, C_BASE]),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=[alt.Tooltip("순위:Q"), alt.Tooltip("equipment_id:N", title="드론"),
             alt.Tooltip("equipment_type:N", title="유형"), alt.Tooltip("unit_code:N", title="부대"),
             alt.Tooltip("prob:Q", title="확률", format=".3f")])
rule = alt.Chart(m.head(1).assign(t=t300)).mark_rule(strokeDash=[4, 4], color="#52514e").encode(y="t:Q")
text = alt.Chart(m.head(1).assign(t=t300, label=f"위험 표시 기준 {t300:.2f}")).mark_text(
    align="right", dy=-8, color="#52514e").encode(x=alt.value(780), y="t:Q", text="label:N")
st.altair_chart((bars + rule + text).properties(height=280), width="stretch")
st.caption("확률 높은 순 정렬. 막대에 마우스를 올리면 드론 정보가 보입니다.")

# ── 20대를 넘는 드론 ────────────────────────────────────────
st.header("20대를 넘는 기준 이상 드론")
if len(overflow):
    st.dataframe(overflow.rename(columns={"equipment_id": "드론", "equipment_type": "유형", "unit_code": "부대",
                                          "maintenance_date": "최근 정비일", "maintenance_type": "정비 구분",
                                          "fault_category": "고장 유형", "prob": "확률"})
                 .drop(columns=["flag_300", "flag_600"]), hide_index=True, width="stretch")
    st.markdown(
        f"**처리 방안** (기준 이상 {len(over)}대 중 여력 밖 {len(overflow)}대)\n"
        "- 다음 달 초 우선 점검 대상으로 넘긴다.\n"
        "- 그때까지 비행 전 점검을 강화하고 고위험 임무에서 뺀다.\n"
        "- 같은 부대 정비반 협조가 가능하면 추가 점검한다.")
else:
    st.success("기준 이상 드론이 20대 이하이므로 모두 이번 달에 재점검합니다.")

# ── 시험 예측 ───────────────────────────────────────────────
st.header("2025 시험 기간 예측 (제출용)")
c1, c2, c3 = st.columns(3)
c1.metric("예측 건수", f"{len(sub):,}건")
c2.metric("확률 평균", f"{sub.prob.mean():.3f}")
c3.metric(f"기준({t300:.2f}) 이상", f"{int((sub.prob >= t300).sum())}건")
d1, d2 = st.columns(2)
d1.download_button("test_predictions_2025.csv 받기", (OUT / "test_predictions_2025.csv").read_bytes(),
                   "test_predictions_2025.csv", "text/csv", width="stretch")
d2.download_button("recheck_plan_2026_01.csv 받기", (OUT / "recheck_plan_2026_01.csv").read_bytes(),
                   "recheck_plan_2026_01.csv", "text/csv", width="stretch")

# ── 한계 ────────────────────────────────────────────────────
st.header("모델의 한계")
st.markdown(
    "- 교육용 가상 데이터로 만든 모델이며, 확률은 과거 패턴에 근거한 추정치입니다.\n"
    "- 비용 기준은 재점검하면 재고장을 막는다는 가정 위에서 계산했습니다.\n"
    "- 모델 선택과 위험 기준을 같은 검증 기간에서 정해 검증 성능이 다소 낙관적일 수 있습니다.\n"
    "- 위험 기준은 검증 모델로 정했고, 최종 모델은 더 많은 데이터로 다시 학습해 확률 분포가 조금 다를 수 있습니다.\n"
    "- 요인 설명은 상관관계이며 고장 원인을 뜻하지 않습니다.")
st.warning("**최종 재점검 여부는 정비 담당자가 현장 상태 · 임무 일정을 함께 보고 판단합니다.**")
