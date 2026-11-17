"""[1] 데이터 살펴보기: 분포, 결측, 이상치, 패턴"""
import altair as alt
import pandas as pd
import streamlit as st

from app_utils import C_MAIN, DATA_FILES, read_data, read_text
from drone_pm import cost_unit_error, repeat_failure_drones

TARGET = "failure_within_30days"
SEASON = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
          6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}
NUM_COLS = {
    "equipment_age_years": "경과연수(년)", "severity": "고장 심각도", "downtime_hours": "운용중단 시간",
    "repair_duration_hours": "수리 시간", "parts_replaced_count": "교체 부품 수",
    "maintenance_cost": "정비 비용(만원)", "cumulative_operating_hours": "누적 비행시간",
    "days_since_last_maintenance": "직전 정비 뒤 경과일", "availability_rate": "그 달 가동률",
}

st.title("데이터 살펴보기")
st.caption("README [1] · 원본 data/*.csv를 그대로 읽어 계산 (처리 전 값)")

files = {name: read_data(name) for name in DATA_FILES}
train = files["drone_train.csv"]

# ── 결측 ────────────────────────────────────────────────────
st.header("결측")
rows = []
for name, df in files.items():
    for col in ["fault_code", "fault_category", "days_since_last_maintenance", "availability_rate"]:
        rows.append({"파일": name, "컬럼": col,
                     "결측 수": int(df[col].isna().sum()) if col in df else None,
                     "비고": "" if col in df else "컬럼 없음"})
miss = pd.DataFrame(rows).pivot(index="컬럼", columns="파일", values="결측 수")[list(DATA_FILES)]
st.dataframe(miss, width="stretch")

na_type = train[train.fault_category.isna()].maintenance_type.value_counts()
first = train.sort_values("maintenance_date").groupby("equipment_id").head(1)
c1, c2 = st.columns(2)
c1.markdown(f"**고장 코드 · 유형 결측 → 원래 값 없음**  \n결측 {int(train.fault_category.isna().sum()):,}건이 "
            f"모두 `{', '.join(na_type.index)}`입니다. 예방정비는 고장이 없으므로 값이 없는 것이 정상입니다.")
c2.markdown(f"**직전 정비 뒤 경과일 결측 → 원래 값 없음**  \n결측 {int(train.days_since_last_maintenance.isna().sum())}건 = "
            f"드론별 첫 기록 {len(first)}건. 직전 정비가 없어서 생긴 결측입니다.")
st.info("`availability_rate`는 이번 달 파일(drone_this_month.csv)에 컬럼 자체가 없습니다.")

# ── 이상치 ──────────────────────────────────────────────────
st.header("이상치")
all_rows = pd.concat([df.assign(파일=n) for n, df in files.items()])
o1 = all_rows[all_rows.downtime_hours >= 999]
o2 = all_rows[cost_unit_error(all_rows.maintenance_cost)]
low = train.availability_rate <= 0.3
c1, c2, c3 = st.columns(3)
c1.metric("운용중단 시간 = 999", f"{len(o1)}건",
          help=f"999를 뺀 최댓값 {all_rows.loc[all_rows.downtime_hours < 999, 'downtime_hours'].max()}시간")
c2.metric("정비 비용 ≥ 1만 · 100의 배수", f"{len(o2)}건",
          help=f"그 외 최댓값 {all_rows.loc[~cost_unit_error(all_rows.maintenance_cost), 'maintenance_cost'].max()}만원")
c3.metric("가동률 ≤ 0.3의 재고장률", f"{train.loc[low, TARGET].mean():.0%}",
          f"그 외 {train.loc[~low, TARGET].mean():.0%}", delta_color="off")
with st.expander("이상치 행 보기"):
    cols = ["파일", "equipment_id", "maintenance_date", "maintenance_type", "downtime_hours",
            "repair_duration_hours", "maintenance_cost"]
    st.dataframe(pd.concat([o1, o2])[cols], hide_index=True, width="stretch")
st.caption("999는 다른 값과 동떨어진 결측 코드. 가동률은 정비 뒤 그 달의 고장이 섞인 값이라 모델에서 제외 "
           "→ '데이터 처리 · 컬럼 선택' 페이지")

st.subheader("정비 비용 100배 오입력 근거")
normal = all_rows[~cost_unit_error(all_rows.maintenance_cost)]
rng = normal.groupby("maintenance_type").maintenance_cost.agg(["min", "median", "max"]).astype(int)
rng.columns = ["최솟값", "중앙값", "최댓값"]
c1, c2 = st.columns([2, 3])
c1.dataframe(rng, width="stretch")
mult100 = sorted(normal.loc[normal.maintenance_cost % 100 == 0, "maintenance_cost"].unique().tolist())
gap = int(all_rows.maintenance_cost.between(normal.maintenance_cost.max() + 1, o2.maintenance_cost.min() - 1).sum())
c2.markdown(
    f"- 정상값 최댓값 **{normal.maintenance_cost.max()}**, 오입력 최솟값 **{o2.maintenance_cost.min():,}** — 그 사이 값 **{gap}건**\n"
    f"- 오입력 {len(o2)}건은 모두 100의 배수, 정상값 {len(normal):,}건 중 100의 배수는 {mult100}뿐\n"
    f"- ÷100 하면 {sorted((o2.maintenance_cost / 100).astype(int).tolist())} → 모두 "
    f"{', '.join(o2.maintenance_type.unique())} 정상 범위\n"
    "- 기준: **1만 이상이면서 100의 배수 → ÷100**")

# ── 반복 고장 드론 · 기간별 변화 ────────────────────────────
st.header("눈여겨볼 점")
rep = repeat_failure_drones(train)
cnt = all_rows.groupby("equipment_id").size()
st.subheader("반복 고장 드론")
st.caption("학습 데이터에서 정비 간격 중앙값 7일 이하이면서 재고장률 80% 이상인 드론")
st.dataframe(pd.DataFrame([{
    "드론": e, "유형": train.loc[train.equipment_id == e, "equipment_type"].iloc[0],
    "부대": train.loc[train.equipment_id == e, "unit_code"].iloc[0],
    "전체 기록 수": int(cnt[e]), "정비 간격 중앙값(일)": train.loc[train.equipment_id == e, "days_since_last_maintenance"].median(),
    "학습 재고장률": f"{train.loc[train.equipment_id == e, TARGET].mean():.0%}"} for e in rep]),
    hide_index=True, width="stretch")
st.markdown(f"드론별 기록 수 중앙값은 {cnt.median():.0f}건, 전체 재고장률은 {train[TARGET].mean():.0%}입니다. "
            "날짜 · 비행시간이 일관되어 입력 오류가 아니라 **실제로 고장이 반복되는 기체**로 보고 지우지 않았습니다. "
            "다만 맞히기 쉬운 재고장이 많아 PR-AUC를 높여 보이게 하므로, 이 드론을 뺀 PR-AUC도 함께 봅니다(모델 성능 페이지).")

st.subheader("기간별 드론 노후화")
tmax = train.equipment_age_years.max()
drift = pd.DataFrame([{
    "파일": n, "경과연수 중앙값": round(df.equipment_age_years.median(), 1),
    "누적 비행시간 중앙값": round(df.cumulative_operating_hours.median()),
    f"학습 최댓값({tmax}년)보다 오래된 기록": f"{(df.equipment_age_years > tmax).mean():.0%}"}
    for n, df in files.items()])
st.dataframe(drift, hide_index=True, width="stretch")
st.caption("시험 · 이번 달 드론이 학습 때보다 오래되었습니다. 트리 모델은 학습 범위 밖 값을 최댓값처럼 다루므로 "
           "오래된 드론의 확률은 덜 정확할 수 있습니다.")

# ── 분포 ────────────────────────────────────────────────────
st.header("수치 컬럼 분포 (학습 데이터)")
col = st.selectbox("컬럼", list(NUM_COLS), format_func=lambda c: f"{NUM_COLS[c]} ({c})")
clip = st.toggle("이상치(999, 비용 100배 오입력) 빼고 보기", value=True)
s = train[col].dropna()
if clip and col == "downtime_hours":
    s = s[s < 999]
if clip and col == "maintenance_cost":
    s = s[~cost_unit_error(s)]
st.altair_chart(
    alt.Chart(pd.DataFrame({col: s})).mark_bar(color=C_MAIN, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
    .encode(x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40), title=NUM_COLS[col]),
            y=alt.Y("count():Q", title="건수"),
            tooltip=[alt.Tooltip(f"{col}:Q", bin=alt.Bin(maxbins=40), title="구간"),
                     alt.Tooltip("count():Q", title="건수")])
    .properties(height=260), width="stretch")
st.dataframe(train[list(NUM_COLS)].describe().T.round(2), width="stretch")

# ── 패턴 ────────────────────────────────────────────────────
st.header("재고장률 패턴 (학습 데이터)")
st.caption(f"전체 재고장률 {train[TARGET].mean():.1%} ({int(train[TARGET].sum()):,}/{len(train):,}건)")
t = train.assign(계절=train.maintenance_date.dt.month.map(SEASON),
                 고장유형=train.fault_category.fillna("없음(예방정비)"))
groups = {"정비 구분": "maintenance_type", "드론 유형": "equipment_type", "고장 유형": "고장유형", "계절": "계절"}
tabs = st.tabs(list(groups))
for tab, (label, g) in zip(tabs, groups.items()):
    r = t.groupby(g)[TARGET].agg(재고장률="mean", 건수="size").reset_index().rename(columns={g: label})
    tab.altair_chart(
        alt.Chart(r).mark_bar(color=C_MAIN, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(y=alt.Y(f"{label}:N", sort="-x", title=None),
                x=alt.X("재고장률:Q", axis=alt.Axis(format="%"), title="재고장률"),
                tooltip=[label, alt.Tooltip("재고장률:Q", format=".1%"), "건수"])
        .properties(height=40 * len(r) + 30), width="stretch")

def heatmap(ct: pd.DataFrame, row_name: str, row_order=None):
    long = ct.reset_index().melt(id_vars=ct.index.name, var_name="고장 유형", value_name="건수")
    long = long.rename(columns={ct.index.name: row_name})
    base = alt.Chart(long).encode(
        x=alt.X("고장 유형:N", title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y(f"{row_name}:N", sort=row_order, title=None))
    cells = base.mark_rect(cornerRadius=3, stroke="white", strokeWidth=2).encode(
        color=alt.Color("건수:Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="고장 건수")),
        tooltip=[row_name, "고장 유형", "건수"])
    text = base.mark_text(fontSize=13).encode(
        text="건수:Q",
        color=alt.condition(alt.datum.건수 > long.건수.max() * 0.6, alt.value("white"), alt.value("#0b0b0b")))
    return (cells + text).properties(height=48 * len(ct) + 30)


c1, c2 = st.columns(2)
with c1:
    st.subheader("계절 × 고장 유형")
    order = ["봄", "여름", "가을", "겨울"]
    st.altair_chart(heatmap(pd.crosstab(t.계절, train.fault_category).reindex(order), "계절", order),
                    width="stretch")
with c2:
    st.subheader("드론 유형 × 고장 유형")
    st.altair_chart(heatmap(pd.crosstab(train.equipment_type, train.fault_category), "드론 유형"),
                    width="stretch")

eda = read_text("eda_summary.txt")
if eda:
    with st.expander("drone_pm.py가 출력한 전체 요약 (outputs/eda_summary.txt)"):
        st.code(eda, language=None)
