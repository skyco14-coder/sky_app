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


# 캐시는 파일 이름 + 수정 시각으로 구분한다. 이름만 쓰면 결과 파일이 바뀌어도(새로 푸시해도)
# 앱이 다시 시작될 때까지 예전 값을 계속 보여 준다.
@st.cache_data
def _read_csv(path: str, mtime: float, **kw) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


def read_data(name: str) -> pd.DataFrame:
    """data/ 폴더의 원본 CSV"""
    path = DATA / name
    return _read_csv(str(path), path.stat().st_mtime, parse_dates=["maintenance_date"])


def read_output(name: str) -> pd.DataFrame | None:
    """outputs/ 폴더의 결과 CSV. 없으면 None (drone_pm.py를 먼저 실행해야 함)"""
    path = OUT / name
    if not path.exists():
        return None
    return _read_csv(str(path), path.stat().st_mtime)


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


# 차트 색: 팔레트 고정 순서(1 파랑, 2 주황) + 기준선은 회색
C_MAIN = "#2a78d6"      # 선택 모델 / 단일 계열
C_COMPARE = "#eb6834"   # 비교 대상(경과연수순, 600만원 등)
C_BASE = "#9a9893"      # 기준선 · 참고선


def read_text(name: str) -> str | None:
    path = OUT / name
    return path.read_text(encoding="utf-8") if path.exists() else None


def best_model(metrics: pd.DataFrame) -> pd.Series:
    """validation_metrics.csv에서 선택 모델(기준선 제외, PR-AUC 최고) — drone_pm.py와 같은 규칙"""
    models = metrics[~metrics.model.str.startswith("기준선")]
    return models.sort_values(["PR_AUC", "top20_hits_total"], ascending=False).iloc[0]


def coming_soon(step: str, items: list[str]):
    """뼈대 페이지용 '준비 중' 안내"""
    st.info(f"README {step} 내용을 담을 페이지입니다. 아직 준비 중입니다.")
    st.markdown("**들어갈 내용**\n" + "\n".join(f"- {i}" for i in items))
