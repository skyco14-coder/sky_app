"""데이터 살펴보기 페이지 (뼈대)"""
import streamlit as st

from app_utils import coming_soon

st.title("데이터 살펴보기")
coming_soon("[1]", [
    "컬럼별 분포 · 이상치 · 결측 확인 (outputs/eda_summary.txt)",
    "결측의 원인 (입력 실수 / 원래 값 없음)",
    "드론 유형 · 계절 · 고장 유형별 재고장률 패턴",
])
