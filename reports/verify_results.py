"""drone_pm.py 결과 검증

실행:  .venv\\Scripts\\python reports\\verify_results.py
결과:  reports/verification_result.txt

outputs/ 결과는 읽기만 한다. 다시 계산이 필요한 부분은 임시 폴더에서 계산한다.
가능한 한 drone_pm.py의 함수를 쓰지 않고 따로 계산해서 대조한다.
(모델 재학습과 피처 계산은 같은 결과가 나오는지 보는 것이 목적이라 drone_pm.py 함수를 쓴다.)
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import drone_pm as pm  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "outputs"
RESULT = Path(__file__).parent / "verification_result.txt"
TARGET = "failure_within_30days"
VALID_START, VALID_END = pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")

lines: list[str] = []
summary: list[tuple[str, bool, str]] = []


def say(msg=""):
    print(msg)
    lines.append(str(msg))


def check(name: str, ok: bool, detail: str = ""):
    summary.append((name, bool(ok), detail))
    say(f"  [{'통과' if ok else '실패'}] {name}" + (f" — {detail}" if detail else ""))


def read(path, **kw):
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


# ---------------------------------------------------------------------------
def v1_submission():
    say("\n[V1] 제출 파일 형식 (test_predictions_2025.csv)")
    sub = read(OUT / "test_predictions_2025.csv", dtype=str)
    test = read(DATA / "drone_test.csv", dtype=str)
    check("열 이름", list(sub.columns) == ["equipment_id", "maintenance_date", "prob"], str(list(sub.columns)))
    check("행 수 1,241", len(sub) == 1241 == len(test), f"{len(sub)}행")
    check("드론·날짜가 시험 데이터와 같은 순서로 모두 일치",
          (sub[["equipment_id", "maintenance_date"]].values == test[["equipment_id", "maintenance_date"]].values).all())
    check("날짜 형식 YYYY-MM-DD", sub.maintenance_date.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all())
    p = pd.to_numeric(sub.prob, errors="coerce")
    check("확률이 0~1, 빈 값 없음", p.notna().all() and p.between(0, 1).all(), f"{p.min():.4f} ~ {p.max():.4f}")


# ---------------------------------------------------------------------------
def v2_label_definition():
    say("\n[V2] 정답 정의 확인 — 정비 뒤 30일 안에 사후·긴급정비가 있으면 1")
    tr = read(DATA / "drone_train.csv", parse_dates=["maintenance_date"])
    te = read(DATA / "drone_test.csv", parse_dates=["maintenance_date"])
    al = pd.concat([tr, te]).sort_values(["equipment_id", "maintenance_date"])
    fail_dates = al[al.maintenance_type.isin(["사후정비", "긴급정비"])].groupby("equipment_id").maintenance_date.apply(np.array)
    derived = []
    for r in tr.itertuples():
        fd = fail_dates.get(r.equipment_id, np.array([], dtype="datetime64[ns]"))
        gap = (fd - np.datetime64(r.maintenance_date)) / np.timedelta64(1, "D")
        derived.append(int(((gap > 0) & (gap <= 30)).any()))
    agree = (np.array(derived) == tr[TARGET].values).mean()
    check("기록으로 다시 계산한 정답과 일치율", agree > 0.99, f"{agree:.4f}")
    say("    ※ 정답 정의를 바르게 이해했는지 확인하는 용도. 시험 정답을 만드는 데 쓰지 않는다.")


# ---------------------------------------------------------------------------
def v3_split():
    say("\n[V3] 기간 분할")
    tr = read(DATA / "drone_train.csv", parse_dates=["maintenance_date"])
    purge_end = pm.VALID_START - pd.Timedelta(days=pm.PURGE_DAYS)
    fit = tr[tr.maintenance_date < purge_end]
    val = tr[tr.maintenance_date >= pm.VALID_START]
    check("검증 시작일 2024-07-01", pm.VALID_START == VALID_START)
    check("검증 기간 2024-07 ~ 12", val.maintenance_date.between(VALID_START, VALID_END).all(),
          f"{val.maintenance_date.min():%Y-%m-%d} ~ {val.maintenance_date.max():%Y-%m-%d}, {len(val)}건")
    last_window = fit.maintenance_date.max() + pd.Timedelta(days=30)
    check("학습 행의 정답 창(30일)이 검증 기간에 닿지 않음", last_window < VALID_START,
          f"학습 마지막 {fit.maintenance_date.max():%Y-%m-%d} + 30일 = {last_window:%Y-%m-%d}")
    check("학습·검증에 2025 시험 데이터 없음", tr.maintenance_date.max() < TEST_START)
    check("무작위 분할 아님(학습이 모두 검증보다 앞)", fit.maintenance_date.max() < val.maintenance_date.min())


# ---------------------------------------------------------------------------
def v4_leakage(df_full: pd.DataFrame, n_sample=40, seed=0):
    say(f"\n[V4] 미래 정보 누수 — 그 정비일 뒤 기록을 모두 지우고 피처를 다시 계산해도 같은지 ({n_sample}건 표본)")
    raw = pm.load()
    rng = np.random.default_rng(seed)
    cand = raw[raw.maintenance_date >= pd.Timestamp("2024-07-01")]
    sample = cand.sample(n_sample, random_state=seed).row_id.tolist()
    hist_cols = ["fail_cnt_30d", "fail_cnt_90d", "fail_cnt_180d", "fail_cnt_365d", "maint_cnt_90d",
                 "same_cat_fail_365d", "unsuccess_cnt_180d", "severity_sum_180d", "days_since_prev_failure",
                 "days_since_last_pm", "hours_since_last_pm", "n_prior_records", "hours_per_day_recent",
                 "prev_maintenance_type", "days_since_last_maintenance"]
    full = df_full.set_index("row_id")
    bad = []
    for rid in sample:
        d = raw.loc[raw.row_id == rid, "maintenance_date"].iloc[0]
        cut = raw[(raw.maintenance_date < d) | ((raw.maintenance_date == d) & (raw.row_id <= rid))]
        f = pm.build_features(pm.clean(cut)).set_index("row_id").loc[rid, hist_cols]
        g = full.loc[rid, hist_cols]
        for c in hist_cols:
            a, b = f[c], g[c]
            same = (a == b) if isinstance(a, str) else np.isclose(float(a), float(b), equal_nan=True)
            if not same:
                bad.append((rid, c, a, b))
    check("이력 피처가 미래 기록 없이도 똑같이 계산됨", not bad,
          f"{len(sample)}건 x {len(hist_cols)}개 피처, 불일치 {len(bad)}개")
    for b in bad[:10]:
        say(f"    불일치 row {b[0]} {b[1]}: 잘라서 {b[2]} / 전체 {b[3]}")
    check("모델 입력에 availability_rate·정답 없음",
          "availability_rate" not in pm.FEATURES and TARGET not in pm.FEATURES)
    # 비용 단위 오입력 규칙
    cost = raw.set_index("row_id").maintenance_cost
    rule = (cost >= 10000) & (cost % 100 == 0)
    fixed = full.cost_fixed == 1
    check("비용 ÷100 대상 = '1만 이상이면서 100의 배수' 행과 정확히 같음",
          set(rule[rule].index) == set(fixed[fixed].index), f"{int(rule.sum())}건")
    check("예전 기준(> 2,000)과도 같은 행 (기준을 바꿔도 결과 동일)", set((cost > 2000)[cost > 2000].index) == set(rule[rule].index))
    normal = raw[~rule.values]
    typ_max = normal.groupby("maintenance_type").maintenance_cost.max()
    fixed_rows = full.loc[fixed[fixed].index]
    check("÷100 한 값이 같은 정비 구분의 정상 범위 안",
          all(r.maintenance_cost <= typ_max[r.maintenance_type] for r in fixed_rows.itertuples()),
          f"{sorted(fixed_rows.maintenance_cost.tolist())}, 정상 최댓값 {typ_max.to_dict()}")
    check("정상 최댓값과 오입력 최솟값 사이에 값 없음",
          int(cost.between(normal.maintenance_cost.max() + 1, cost[rule].min() - 1).sum()) == 0,
          f"{normal.maintenance_cost.max()} ~ {cost[rule].min()}")
    # 직전 정비 없음 / 예방정비 이력 없음 -> 비워 두고 표시 (0 · 중앙값으로 채우지 않음)
    first = full.is_first_maintenance == 1
    check("직전 정비 없는 기록은 경과일 · 하루 비행시간이 빈 값 + is_first_maintenance=1",
          full.loc[first, ["days_since_last_maintenance", "hours_per_day_recent"]].isna().all().all()
          and (raw.set_index("row_id").days_since_last_maintenance.isna() == first).all(),
          f"{int(first.sum())}건, 분할별 {full[first].split.value_counts().to_dict()}")
    check("직전 정비 있는 기록은 빈 값 없음",
          full.loc[~first, ["days_since_last_maintenance", "hours_per_day_recent"]].notna().all().all())
    check("0으로 채운 값 없음 (경과일 최솟값 > 0)", full.days_since_last_maintenance.min() > 0,
          f"최솟값 {full.days_since_last_maintenance.min()}")
    raw999 = raw.set_index("row_id").downtime_hours >= 999
    after = full.loc[raw999[raw999].index]
    check("downtime_hours 999는 '모름'(빈 값)으로 남고 표시됨",
          after.downtime_hours.isna().all() and (after.downtime_missing == 1).all()
          and full.loc[raw999[~raw999].index, "downtime_hours"].notna().all(),
          f"999 {int(raw999.sum())}건 → 빈 값 {int(after.downtime_hours.isna().sum())}건")


# ---------------------------------------------------------------------------
def monthly_hits(val: pd.DataFrame, score, k=20):
    d = val.assign(score=score, ym=val.maintenance_date.dt.to_period("M"))
    d = d.sort_values(["maintenance_date", "row_id"]).groupby(["ym", "equipment_id"]).tail(1)
    return {str(ym): int(g.nlargest(k, "score")[TARGET].sum()) for ym, g in d.groupby("ym")}, d


def v5_validation(df: pd.DataFrame):
    say("\n[V5] 검증 성능 재현 (validation_metrics.csv와 대조)")
    metrics = read(OUT / "validation_metrics.csv").set_index("model")
    train = df[df.split == "train"]
    purge_end = pm.VALID_START - pd.Timedelta(days=pm.PURGE_DAYS)
    fit = train[train.maintenance_date < purge_end]
    val = train[train.maintenance_date >= pm.VALID_START]

    preds = {}
    for name, model in pm.make_models().items():
        model.fit(fit[pm.FEATURES], fit[TARGET])
        p = model.predict_proba(val[pm.FEATURES])[:, 1]
        preds[name] = p
        pr = average_precision_score(val[TARGET], p)
        hits, _ = monthly_hits(val, p)
        check(f"{name}: PR-AUC·월 20대 적중 재현",
              abs(pr - metrics.at[name, "PR_AUC"]) < 1e-3 and sum(hits.values()) == metrics.at[name, "top20_hits_total"],
              f"PR-AUC {pr:.4f} (파일 {metrics.at[name, 'PR_AUC']}), 적중 {sum(hits.values())} (파일 {metrics.at[name, 'top20_hits_total']:g})")

    # 기준선: 무작위 기대값은 식으로 계산(매달 20 x 그 달 재고장 비율)
    _, latest = monthly_hits(val, np.zeros(len(val)))
    exp_rand = sum(min(20, len(g)) * g[TARGET].mean() for _, g in latest.groupby("ym"))
    file_rand = metrics.loc[metrics.index.str.contains("무작위"), "top20_hits_total"].iloc[0]
    check("무작위 기준선 적중 = 식으로 구한 기대값", abs(exp_rand - file_rand) < 0.5,
          f"기대값 {exp_rand:.2f} / 파일(1000회 평균) {file_rand}")
    file_rand_pr = metrics.loc[metrics.index.str.contains("무작위"), "PR_AUC"].iloc[0]
    check("무작위 PR-AUC ≈ 재고장 비율", abs(file_rand_pr - val[TARGET].mean()) < 0.02,
          f"재고장 비율 {val[TARGET].mean():.4f} / 파일 {file_rand_pr}")
    age_pr = average_precision_score(val[TARGET], val.equipment_age_years)
    check("경과연수순 PR-AUC 재현", abs(age_pr - metrics.loc[metrics.index.str.contains("경과연수"), "PR_AUC"].iloc[0]) < 1e-3,
          f"{age_pr:.4f}")

    models = metrics[~metrics.index.str.startswith("기준선")]
    best = models.PR_AUC.idxmax()
    repeat = pm.repeat_failure_drones(train)
    keep = ~val.equipment_id.isin(repeat).values
    pr_ex = average_precision_score(val[TARGET].values[keep], preds[best][keep])
    check("반복 고장 드론을 뺀 PR-AUC 재현", abs(pr_ex - metrics.at[best, "PR_AUC_excl_repeat"]) < 1e-3,
          f"{repeat}, {pr_ex:.4f} (파일 {metrics.at[best, 'PR_AUC_excl_repeat']})")
    base_ex = metrics.loc[metrics.index.str.startswith("기준선"), "PR_AUC_excl_repeat"].max()
    check("반복 고장 드론을 빼도 기준선보다 높음", pr_ex > base_ex, f"{pr_ex:.3f} > {base_ex:.3f}")
    check("선택 모델이 검증 PR-AUC 최고", best == "랜덤포레스트", best)
    check("선택 모델이 두 기준선보다 PR-AUC·적중 모두 높음",
          all(models.loc[best, c] > metrics.loc[metrics.index.str.startswith("기준선"), c].max()
              for c in ["PR_AUC", "top20_hits_total"]))
    return val, preds[best], best


# ---------------------------------------------------------------------------
def v6_threshold(val, p_val):
    say("\n[V6] 위험 표시 기준 — 검증 기간 예측만으로 다시 계산 (threshold_costs.csv와 대조)")
    costs = read(OUT / "threshold_costs.csv")
    y = val[TARGET].values
    res = {}
    for miss in (300, 600):
        best_t, best_c = None, None
        for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
            flag = p_val >= t
            c = int(((~flag) & (y == 1)).sum()) * miss + int(flag.sum()) * 60
            if best_c is None or c < best_c or (c == best_c and t > best_t):
                best_t, best_c = t, c
        file_row = costs.sort_values([f"cost_{miss}", "threshold"], ascending=[True, False]).iloc[0]
        check(f"놓침 {miss}만원 기준 재현", best_t == file_row.threshold and best_c == file_row[f"cost_{miss}"],
              f"다시 계산 {best_t:.2f}/{best_c:,}만원, 파일 {file_row.threshold:.2f}/{int(file_row[f'cost_{miss}']):,}만원")
        res[miss] = best_t
    check("놓침 비용이 커지면 기준이 같거나 낮아짐", res[600] <= res[300], f"300→{res[300]:.2f}, 600→{res[600]:.2f}")
    check("기준을 고른 데이터에 2025 시험 데이터 없음", val.maintenance_date.max() < TEST_START,
          f"검증 마지막 {val.maintenance_date.max():%Y-%m-%d}")
    return res


# ---------------------------------------------------------------------------
def v7_final(df, best_name, thr):
    say("\n[V7] 최종 예측 재현 (학습 전체로 다시 학습)")
    train, test, month = (df[df.split == s] for s in ("train", "test", "this_month"))
    model = pm.make_models()[best_name]
    model.fit(train[pm.FEATURES], train[TARGET])
    p_test = model.predict_proba(test[pm.FEATURES])[:, 1]
    sub = read(OUT / "test_predictions_2025.csv")
    diff = np.abs(p_test - sub.prob.values).max()
    check("시험 예측 재현(소수 6자리 반올림 오차 이내)", diff < 1e-5, f"최대 차이 {diff:.2e}")
    check("최종 학습 데이터가 2024-12까지만", train.maintenance_date.max() < TEST_START)

    p_m = model.predict_proba(month[pm.FEATURES])[:, 1]
    latest = month.assign(prob=p_m).sort_values(["maintenance_date", "row_id"]).groupby("equipment_id").tail(1)
    latest = latest.sort_values("prob", ascending=False)
    plan = read(OUT / "recheck_plan_2026_01.csv")
    check("재점검 20대 = 드론별 최근 기록의 확률 상위 20대",
          latest.equipment_id.head(20).tolist() == plan["드론"].tolist())
    check("재점검 20대 중복 드론 없음", plan["드론"].nunique() == 20)
    repeat = pm.repeat_failure_drones(train)
    rep_rows = plan[plan["드론"].isin(repeat)]
    check("반복 고장 드론 권고에 근본 원인 조사 문구", rep_rows["권고"].str.startswith("반복 고장 기체").all()
          and not plan[~plan["드론"].isin(repeat)]["권고"].str.startswith("반복 고장 기체").any(),
          f"{rep_rows['드론'].tolist()}")
    over = int((latest.prob >= thr[300]).sum())
    mp = read(OUT / "this_month_predictions.csv")
    check("기준 이상 드론 수 재현", over == int(mp.flag_300.sum()), f"{over}대 (2026-01 정비 드론 {len(latest)}대 중)")
    return over


# ---------------------------------------------------------------------------
def v8_report(thr, over):
    say("\n[V8] 보고서 숫자 대조 (report.md ↔ outputs/*.csv)")
    rep = (OUT / "report.md").read_text(encoding="utf-8")
    metrics = read(OUT / "validation_metrics.csv")
    for r in metrics.itertuples():
        row = re.search(rf"\| {re.escape(r.model)} \| ([\d.]+) \| [\d.]+ \| ([\d.]+) \|", rep)
        ok = row and abs(float(row.group(1)) - r.PR_AUC) < 0.001 and float(row.group(2)) == r.top20_hits_total
        check(f"보고서 성능표: {r.model}", bool(ok), row.group(0) if row else "표에서 못 찾음")
    for miss in (300, 600):
        row = re.search(rf"\| {miss}만원 \| ([\d.]+) \|", rep)
        check(f"보고서 기준({miss}만원)", bool(row) and float(row.group(1)) == thr[miss],
              row.group(0) if row else "못 찾음")
    check("분석 보고서에 재점검 계획 문서 안내가 있음", "outputs/recheck_plan_2026_01.md" in rep)

    say("\n[V9] 재점검 계획 문서 대조 (recheck_plan_2026_01.md ↔ outputs/*.csv)")
    path = OUT / "recheck_plan_2026_01.md"
    check("재점검 계획 문서가 있음", path.exists())
    rep = path.read_text(encoding="utf-8") if path.exists() else ""
    for need in ["## 1. 재점검 대상 20대", "## 2. 모델의 한계", "처리 방안", "정비 담당자"]:
        check(f"계획 문서에 '{need}'", need in rep)
    m = re.search(r"기준\(([\d.]+)\) 이상 \*\*(\d+)대\*\*", rep)
    check("계획 문서 기준 이상 드론 수", bool(m) and int(m.group(2)) == over and float(m.group(1)) == thr[300],
          m.group(0) if m else "못 찾음")
    mp = read(OUT / "this_month_predictions.csv")
    a, b = mp.iloc[19], mp.iloc[20]
    m = re.search(r"20위 (DR-\d+).*? 확률 ([\d.]+), 21위 (DR-\d+).*? 확률 ([\d.]+)\. 차이 ([\d.]+)", rep)
    if a.prob - b.prob < pm.BOUNDARY_GAP:
        check("보고서 20위 경계 문구가 결과 파일과 같음",
              bool(m) and m.group(1) == a.equipment_id and m.group(3) == b.equipment_id
              and abs(float(m.group(2)) - a.prob) < 1e-4 and abs(float(m.group(4)) - b.prob) < 1e-4,
              m.group(0) if m else "문구 없음")
    else:
        check("20위 경계 차이가 커서 문구 없음", m is None, f"차이 {a.prob - b.prob:.4f}")
    plan = read(OUT / "recheck_plan_2026_01.csv")
    check("계획 문서 재점검 표 20대가 계획 CSV와 같음",
          all(f"| {r['순위']} | {r['드론']} |" in rep for _, r in plan.iterrows()))


# ---------------------------------------------------------------------------
def main():
    say("drone_pm.py 결과 검증")
    say(f"대상: {OUT}")
    v1_submission()
    v2_label_definition()
    v3_split()

    with tempfile.TemporaryDirectory() as tmp:
        pm.OUT = Path(tmp)            # clean()이 쓰는 로그를 outputs/ 대신 임시 폴더에 쓰게 함
        df = pm.build_features(pm.clean(pm.load()))
        v4_leakage(df)
        val, p_val, best = v5_validation(df)
        thr = v6_threshold(val, p_val)
        over = v7_final(df, best, thr)
    v8_report(thr, over)

    n_ok = sum(ok for _, ok, _ in summary)
    say(f"\n결과: {n_ok}/{len(summary)} 통과")
    for name, ok, detail in summary:
        if not ok:
            say(f"  실패: {name} — {detail}")
    RESULT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {RESULT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
