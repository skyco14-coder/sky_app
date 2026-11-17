"""[6] 위험 표시 기준: 비용이 가장 작은 확률 기준 (검증 기간에서만 결정)"""
import altair as alt
import pandas as pd
import streamlit as st

from app_utils import (C_COMPARE, C_MAIN, COST_CHECK, COST_MISS, COST_MISS_ALT, MONTHLY_CAPACITY,
                       best_threshold, require_output)

st.title("위험 표시 기준")
st.caption("README [6] · outputs/threshold_costs.csv (검증 기간 2024-07~12 예측으로만 계산)")

costs = require_output("threshold_costs.csv")
b300 = best_threshold(costs, "cost_300")
b600 = best_threshold(costs, "cost_600")
positives = int(costs.caught.iloc[0] + costs.missed.iloc[0])
n_rows = int(costs.flagged.iloc[0])

st.markdown(f"**비용 = 놓친 재고장 수 × 놓침 비용 + 기준 이상 건수 × {COST_CHECK}만원.** "
            "확률 0.01~0.99를 0.01 간격으로 모두 계산해 비용이 가장 작은 기준을 골랐습니다. "
            "**2025 시험 데이터로는 기준을 고르지 않았습니다.**")

# ── 고른 기준 ───────────────────────────────────────────────
st.header("고른 기준")
c1, c2 = st.columns(2)
for col, miss, b in ((c1, COST_MISS, b300), (c2, COST_MISS_ALT, b600)):
    with col.container(border=True):
        st.markdown(f"**놓침 {miss}만원**")
        a, b_, c = st.columns(3)
        a.metric("기준 확률", f"{b.threshold:.2f}")
        b_.metric("표시 건수", f"{int(b.flagged)}건", f"재고장 {int(b.caught)}건 잡음", delta_color="off")
        c.metric("총비용", f"{int(b[f'cost_{miss}']):,}만원", f"놓침 {int(b.missed)}건", delta_color="off")
st.markdown(
    f"놓침 비용이 {COST_MISS}만원 → {COST_MISS_ALT}만원으로 커지면 기준이 **{b300.threshold:.2f} → {b600.threshold:.2f}**로 "
    f"낮아집니다. 재고장을 놓치는 것이 더 비싸지므로 더 많이 점검하는 쪽이 유리해집니다.")
st.caption(f"참고: 아무것도 점검하지 않으면 {positives * COST_MISS:,}만원({COST_MISS}) / "
           f"{positives * COST_MISS_ALT:,}만원({COST_MISS_ALT}), 전부 점검하면 {n_rows * COST_CHECK:,}만원")

# ── 비용 곡선 ───────────────────────────────────────────────
st.header("기준별 총비용")
long = costs.melt(id_vars=["threshold", "flagged", "caught", "missed"], value_vars=["cost_300", "cost_600"],
                  var_name="놓침 비용", value_name="총비용")
long["놓침 비용"] = long["놓침 비용"].map({"cost_300": f"{COST_MISS}만원", "cost_600": f"{COST_MISS_ALT}만원"})
scale = alt.Scale(domain=[f"{COST_MISS}만원", f"{COST_MISS_ALT}만원"], range=[C_MAIN, C_COMPARE])
hover = alt.selection_point(fields=["threshold"], nearest=True, on="pointerover", empty=False)
base = alt.Chart(long).encode(
    x=alt.X("threshold:Q", title="확률 기준"), y=alt.Y("총비용:Q", title="총비용(만원)"),
    color=alt.Color("놓침 비용:N", scale=scale, legend=alt.Legend(orient="top", title="놓침 비용")))
lines = base.mark_line(strokeWidth=2)
points = base.mark_point(size=80, filled=True).encode(
    opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    tooltip=[alt.Tooltip("threshold:Q", title="기준", format=".2f"), "놓침 비용",
             alt.Tooltip("총비용:Q", format=","), alt.Tooltip("flagged:Q", title="표시 건수"),
             alt.Tooltip("missed:Q", title="놓친 재고장")]).add_params(hover)
rule = alt.Chart(long).mark_rule(color="#9a9893").encode(x="threshold:Q").transform_filter(hover)
best_pts = pd.DataFrame({"threshold": [b300.threshold, b600.threshold],
                         "총비용": [b300.cost_300, b600.cost_600],
                         "놓침 비용": [f"{COST_MISS}만원", f"{COST_MISS_ALT}만원"]})
marks = alt.Chart(best_pts).mark_point(size=160, shape="diamond", filled=True, stroke="white", strokeWidth=2).encode(
    x="threshold:Q", y="총비용:Q", color=alt.Color("놓침 비용:N", scale=scale, legend=None))
labels = alt.Chart(best_pts).mark_text(dy=-16, fontWeight="bold").encode(
    x="threshold:Q", y="총비용:Q", text=alt.Text("threshold:Q", format=".2f"))
st.altair_chart((lines + rule + points + marks + labels).properties(height=340), width="stretch")
st.caption("◆ 표시가 각 놓침 비용에서 비용이 가장 작은 기준입니다. 선 위에 마우스를 올리면 값이 보입니다.")

# ── 직접 바꿔 보기 ──────────────────────────────────────────
st.header("기준을 직접 바꿔 보기")
t = st.slider("확률 기준", 0.01, 0.99, float(b300.threshold), 0.01)
row = costs.loc[(costs.threshold - t).abs().idxmin()]
a, b_, c, d = st.columns(4)
a.metric("표시 건수 (6개월)", f"{int(row.flagged)}건", f"월 평균 {row.flagged / 6:.0f}건", delta_color="off")
b_.metric("잡은 재고장", f"{int(row.caught)}/{positives}건")
c.metric(f"총비용 ({COST_MISS}만원)", f"{int(row.cost_300):,}만원",
         f"{int(row.cost_300 - b300.cost_300):+,} vs 최적", delta_color="inverse")
d.metric(f"총비용 ({COST_MISS_ALT}만원)", f"{int(row.cost_600):,}만원",
         f"{int(row.cost_600 - b600.cost_600):+,} vs 최적", delta_color="inverse")
if row.flagged / 6 > MONTHLY_CAPACITY:
    st.warning(f"이 기준이면 월 평균 {row.flagged / 6:.0f}건이 표시되어 점검 여력 {MONTHLY_CAPACITY}대를 넘습니다. "
               "README의 비용 공식은 점검 여력을 따지지 않으므로, 실제로는 확률 순으로 20대를 먼저 점검합니다.")

with st.expander("전체 표 보기"):
    st.dataframe(costs.rename(columns={"threshold": "기준", "flagged": "표시", "caught": "잡음", "missed": "놓침",
                                       "cost_300": f"비용({COST_MISS})", "cost_600": f"비용({COST_MISS_ALT})"}),
                 hide_index=True, width="stretch")
