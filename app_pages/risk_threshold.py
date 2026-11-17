"""위험 표시 기준 페이지 (뼈대)"""
import streamlit as st

from app_utils import coming_soon

st.title("위험 표시 기준")
coming_soon("[6]", [
    "확률 기준별 비용 그래프 (outputs/threshold_costs.csv)",
    "놓침 300만원 · 600만원일 때 고른 기준 비교",
    "기준을 고른 방법 (검증 기간만 사용)",
])
