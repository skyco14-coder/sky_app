# sky_app — 드론 예측정비 실습

정비를 마친 드론 가운데 **30일 안에 다시 고장 날 드론**을 미리 골라, 2026-01에 재점검할 드론 20대를 정한다.
과제 안내는 [README.txt](README.txt), 작업 규칙은 [AGENTS.md](AGENTS.md)에 있다. 데이터는 교육용 가상 데이터다.

## 제출물

| 제출물 | 파일 |
|---|---|
| (1) 시험 기간 예측 (채점용) | [outputs/test_predictions_2025.csv](outputs/test_predictions_2025.csv) |
| (2) 분석 보고서 | [outputs/report.md](outputs/report.md) |
| (3) 2026-01 재점검 계획 | [outputs/recheck_plan_2026_01.md](outputs/recheck_plan_2026_01.md) · 표 [outputs/recheck_plan_2026_01.csv](outputs/recheck_plan_2026_01.csv) |

## 실행 방법 (Windows)

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

.venv\Scripts\python drone_pm.py                  # 1. 분석 전체 실행 → outputs/ 다시 생성 (약 30초)
.venv\Scripts\python reports\verify_results.py    # 2. 결과 검증 → reports/verification_result.txt
.venv\Scripts\streamlit run streamlit_app.py      # 3. 대시보드 보기 (브라우저가 열림)
```

검증 노트북 [reports/verify_notebook.ipynb](reports/verify_notebook.ipynb)는 VS Code에서 열고 커널로 `.venv`를 고른 뒤
Run All 한다. 처음 한 번 `.venv\Scripts\python -m pip install ipykernel`이 필요하다.

같은 데이터로 다시 실행하면 같은 결과가 나온다 (난수 시드 42 고정).

## 분석 흐름 (README.txt 3장 단계)

| 단계 | 내용 | 코드 (`drone_pm.py`) | 결과 |
|---|---|---|---|
| [1] 데이터 살펴보기 | 분포 · 결측 · 이상치 · 패턴 | `explore()` | `outputs/eda_summary.txt`, 보고서 1장 |
| [2] 데이터 처리 | 999 → 모름, 비용 ÷100, 이유 있는 결측은 비워 둠. 행 삭제 없음 | `clean()` | `outputs/cleaning_log.csv`, 보고서 2장 |
| [3] 피처 만들기 | 드론별 과거 기록으로 이력 피처 (그 정비일까지만) | `build_features()` | 보고서 3장 |
| [4] 컬럼 고르기 | 정비 직후 알 수 없는 `availability_rate` 등 제외 | `FEATURES`, `EXCLUDED` | 보고서 4장 |
| [5] 학습 · 검증 | 기간 분할, PR-AUC · 월 20대 적중, 기준선 비교 | `main()` | `outputs/validation_*.csv`, 보고서 5장 |
| [6] 위험 표시 기준 | 비용 최소 확률 기준 (300 · 600만원) | `threshold_table()` | `outputs/threshold_costs.csv`, 보고서 6장 |
| [7] 시험 예측 | 1,241건 확률 | `main()` | `outputs/test_predictions_2025.csv` |
| [8] 재점검 20대 | 드론별 최근 기록, 위험 요인 설명 | `explain_rows()` | `outputs/recheck_plan_2026_01.*` |

## 폴더

```
data/        원본 CSV (고치지 않음)
drone_pm.py  분석 전체 파이프라인
outputs/     분석 결과 · 제출물 (drone_pm.py가 만듦)
reports/     결과 검증 스크립트 · 노트북
streamlit_app.py, app_pages/, app_utils.py   대시보드 (outputs/를 읽기만 함)
```
