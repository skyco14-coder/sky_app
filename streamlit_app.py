"""드론 예측정비 대시보드 - 시작 파일

실행:  .venv\\Scripts\\streamlit run streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="드론 예측정비", page_icon="🛠️", layout="wide")

PAGES = {
    "개요": [
        st.Page("app_pages/summary.py", title="요약", icon="📋", default=True),
    ],
    "분석 과정": [
        st.Page("app_pages/data_explore.py", title="데이터 살펴보기", icon="🔍"),
        st.Page("app_pages/data_cleaning.py", title="데이터 처리 · 컬럼 선택", icon="🧹"),
        st.Page("app_pages/model_performance.py", title="모델 성능", icon="📈"),
        st.Page("app_pages/risk_threshold.py", title="위험 표시 기준", icon="⚖️"),
    ],
    "결과": [
        st.Page("app_pages/recheck_plan.py", title="재점검 계획", icon="✅"),
    ],
}

st.navigation(PAGES).run()
