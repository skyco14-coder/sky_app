"""재점검 계획 페이지 (뼈대)"""
import streamlit as st

from app_utils import coming_soon

st.title("재점검 계획")
coming_soon("[7]~[8]", [
    "2026-01 재점검 대상 20대와 위험 요인 · 권고 (outputs/recheck_plan_2026_01.csv)",
    "기준 이상 드론 수와 20대 초과분 처리 방안",
    "2025 시험 예측 다운로드 (outputs/test_predictions_2025.csv)",
])
