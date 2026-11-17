"""데이터 처리 · 컬럼 선택 페이지 (뼈대)"""
import streamlit as st

from app_utils import coming_soon

st.title("데이터 처리 · 컬럼 선택")
coming_soon("[2]~[4]", [
    "이상치 · 결측 처리 기준과 근거 (outputs/cleaning_log.csv)",
    "드론별 과거 이력 피처 설명",
    "넣지 않은 컬럼과 이유",
])
