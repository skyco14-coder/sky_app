"""[5] 모델 학습과 검증: 기간 분할, PR-AUC, 월 20대 적중, 기준선 비교"""
import altair as alt
import pandas as pd
import streamlit as st

from app_utils import C_BASE, C_COMPARE, C_MAIN, best_model, require_output
from drone_pm import PURGE_DAYS, VALID_START

st.title("모델 성능")
st.caption("README [5] · outputs/validation_metrics.csv, outputs/validation_monthly_hits.csv")

metrics = require_output("validation_metrics.csv")
monthly = require_output("validation_monthly_hits.csv").rename(columns={"Unnamed: 0": "월"})
best = best_model(metrics)
age = metrics[metrics.model.str.contains("경과연수")].iloc[0]
rand = metrics[metrics.model.str.contains("무작위")].iloc[0]

# ── 기간 분할 ───────────────────────────────────────────────
st.header("기간 분할")
purge_start = VALID_START - pd.Timedelta(days=PURGE_DAYS)
split = pd.DataFrame([
    ("학습", "2023-01-01", (purge_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "모델 학습"),
    ("제외", purge_start.strftime("%Y-%m-%d"), (VALID_START - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
     f"정답 창({PURGE_DAYS}일)이 검증 기간과 겹쳐 뺌"),
    ("검증", VALID_START.strftime("%Y-%m-%d"), "2024-12-31", "모델 선택 · 위험 기준 결정"),
    ("시험", "2025-01-01", "2025-12-31", "채점용 (기준 결정에 쓰지 않음)"),
], columns=["구간", "시작", "끝", "용도"])
color = alt.Scale(domain=["학습", "제외", "검증", "시험"], range=[C_MAIN, C_BASE, C_COMPARE, "#c3c2b7"])
st.altair_chart(
    alt.Chart(split).mark_bar(height=26, cornerRadius=4)
    .encode(x=alt.X("시작:T", title=None, axis=alt.Axis(format="%Y-%m")), x2="끝:T",
            y=alt.Y("구간:N", sort=None, title=None), color=alt.Color("구간:N", scale=color, legend=None),
            tooltip=["구간", "시작", "끝", "용도"])
    .properties(height=170), width="stretch")
st.markdown("무작위로 나누지 않고 **기간으로** 나눴습니다. 최종 모델은 학습 데이터 전체(2023-01~2024-12)로 다시 학습했습니다.")

# ── 성능 ────────────────────────────────────────────────────
st.header("검증 성능과 기준선 비교")
c1, c2, c3 = st.columns(3)
c1.metric(f"PR-AUC · {best.model}", f"{best.PR_AUC:.3f}", f"{best.PR_AUC - age.PR_AUC:+.3f} vs 경과연수순")
c2.metric("월 20대 적중 (6개월 합)", f"{best.top20_hits_total:g}건",
          f"{best.top20_hits_total - age.top20_hits_total:+.1f} vs 경과연수순")
c3.metric("검증 기간 실제 재고장", f"{int(best.positives_total)}건", help="검증 2024-07~12, 643건 중")

m = metrics.assign(구분=metrics.model.map(
    lambda x: "기준선" if x.startswith("기준선") else ("선택 모델" if x == best.model else "다른 모델")))
role = alt.Scale(domain=["선택 모델", "다른 모델", "기준선"], range=[C_MAIN, "#8fb8ec", C_BASE])
left, right = st.columns(2)
for col, field, title, fmt in ((left, "PR_AUC", "PR-AUC", ".3f"),
                               (right, "top20_hits_total", "월 20대 적중 합", ".1f")):
    col.altair_chart(
        alt.Chart(m).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(y=alt.Y("model:N", sort=None, title=None), x=alt.X(f"{field}:Q", title=title),
                color=alt.Color("구분:N", scale=role, legend=alt.Legend(orient="bottom", title=None)),
                tooltip=[alt.Tooltip("model:N", title="모델"), alt.Tooltip(f"{field}:Q", title=title, format=fmt)])
        .properties(height=220, title=title), width="stretch")
st.dataframe(metrics.rename(columns={"model": "모델", "PR_AUC": "PR-AUC", "ROC_AUC": "ROC-AUC",
                                     "top20_hits_total": "월 20대 적중 합", "positives_total": "실제 재고장",
                                     "PR_AUC_excl_repeat": "PR-AUC (반복 고장 드론 제외)"}),
             hide_index=True, width="stretch")
if "PR_AUC_excl_repeat" in metrics:
    st.info(f"**반복 고장 드론(DR-0034, DR-0115)을 빼면** 선택 모델 PR-AUC가 {best.PR_AUC:.3f} → "
            f"**{best.PR_AUC_excl_repeat:.3f}**로 내려갑니다. 이 두 드론은 재고장이 너무 잦아 맞히기 쉽기 때문입니다. "
            f"그래도 경과연수순({age.PR_AUC_excl_repeat:.3f}) · 무작위({rand.PR_AUC_excl_repeat:.3f})보다 높습니다. "
            "월 20대 적중은 드론당 1건으로 세므로 영향이 작습니다.")
st.caption(f"무작위 PR-AUC({rand.PR_AUC:.3f})는 재고장 비율과 같은 값입니다. "
           "정확도는 재고장이 적은 데이터에서 착시가 커서 쓰지 않았습니다.")

# ── 월별 적중 ───────────────────────────────────────────────
st.header("월별 20대 적중")
st.caption("매달 드론별 최근 기록 기준, 확률 상위 20대 중 실제로 30일 안에 재고장한 수")
long = monthly.melt(id_vars="월", value_vars=[best.model, "기준선_경과연수순"], var_name="방법", value_name="적중")
long["방법"] = long.방법.replace({"기준선_경과연수순": "경과연수순"})
pair = alt.Scale(domain=[best.model, "경과연수순"], range=[C_MAIN, C_COMPARE])
base = alt.Chart(long).encode(
    x=alt.X("월:N", title=None), y=alt.Y("적중:Q", title="적중 수 (20대 중)"),
    color=alt.Color("방법:N", scale=pair, legend=alt.Legend(orient="top", title=None)))
st.altair_chart(
    (base.mark_line(strokeWidth=2) + base.mark_point(size=70, filled=True)
     .encode(tooltip=["월", "방법", alt.Tooltip("적중:Q", format=".1f")])).properties(height=280),
    width="stretch")
st.dataframe(monthly.set_index("월"), width="stretch")
st.caption("경과연수순은 동점이 많아 무작위로 200번 순서를 깨서 평균을 냈습니다(소수점이 나오는 이유).")
