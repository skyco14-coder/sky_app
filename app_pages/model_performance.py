"""모델 성능 페이지 (뼈대)"""
import streamlit as st

from app_utils import coming_soon

st.title("모델 성능")
coming_soon("[5]", [
    "학습 · 검증 기간 분할",
    "PR-AUC와 월 20대 적중 수 (outputs/validation_metrics.csv)",
    "기준선(무작위 · 경과연수순)과 비교, 월별 적중 (outputs/validation_monthly_hits.csv)",
])
