"""Streamlit 페이지들이 함께 쓰는 데이터 읽기 함수.

앱은 분석을 다시 돌리지 않고, drone_pm.py가 만든 outputs/ 파일만 읽는다.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

# README.txt의 업무 조건
MONTHLY_CAPACITY = 20   # 한 달 재점검 가능 대수
COST_MISS = 300         # 놓친 재고장 1건(만원)
COST_MISS_ALT = 600     # 민감도 확인용(만원)
COST_CHECK = 60         # 재점검 1대(만원)

DATA_FILES = {
    "drone_train.csv": "학습용 (정답 있음)",
    "drone_test.csv": "시험용 (정답 없음, 채점)",
    "drone_this_month.csv": "이번 달 재점검용 (정답 없음)",
}


@st.cache_data
def read_data(name: str) -> pd.DataFrame:
    """data/ 폴더의 원본 CSV"""
    return pd.read_csv(DATA / name, encoding="utf-8-sig", parse_dates=["maintenance_date"])


@st.cache_data
def read_output(name: str) -> pd.DataFrame | None:
    """outputs/ 폴더의 결과 CSV. 없으면 None (drone_pm.py를 먼저 실행해야 함)"""
    path = OUT / name
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def require_output(name: str) -> pd.DataFrame:
    """결과 파일이 없으면 안내를 띄우고 페이지를 멈춘다."""
    df = read_output(name)
    if df is None:
        st.warning(f"`outputs/{name}` 파일이 없습니다. 먼저 `python drone_pm.py`를 실행하세요.")
        st.stop()
    return df


def best_threshold(costs: pd.DataFrame, cost_col: str) -> pd.Series:
    """threshold_costs.csv에서 비용이 가장 작은 기준(동점이면 높은 기준) — drone_pm.py와 같은 규칙"""
    return costs.sort_values([cost_col, "threshold"], ascending=[True, False]).iloc[0]


def coming_soon(step: str, items: list[str]):
    """뼈대 페이지용 '준비 중' 안내"""
    st.info(f"README {step} 내용을 담을 페이지입니다. 아직 준비 중입니다.")
    st.markdown("**들어갈 내용**\n" + "\n".join(f"- {i}" for i in items))
