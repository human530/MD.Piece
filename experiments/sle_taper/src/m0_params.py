# -*- coding: utf-8 -*-
"""M0：參數載入、出處檢核、佔位標示。

thresholds.json 是全案唯一數值真相；程式其他地方不得硬寫數字（tests/test_params.py 會掃 src/ 內的常數）。
status: anchored / pending_extraction / study_defined。pending_extraction 的 value 為 null，
其 placeholder 子欄位提供佔位值——啟動時列印警告、寫入 assumptions.md、介面標示三處同步。"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_DIR = os.path.join(ROOT, "params")
STATUSES = ("anchored", "pending_extraction", "study_defined")


def load(name="thresholds.json"):
    with open(os.path.join(PARAMS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def value(P, key):
    """取值：pending_extraction 且 value 為 null 時回傳 placeholder（呼叫端須知這是佔位）。"""
    row = P[key]
    if row.get("status") == "pending_extraction":                     # 佔位值（value 為 null 或含 null 的結構）
        return row["placeholder"]
    return row["value"]


def check(P, verbose=True):
    """回傳 dict(missing_source, bad_status, placeholders, counts)。"""
    missing, bad, placeholders = [], [], []
    counts = {s: 0 for s in STATUSES}
    for k, row in P.items():
        if k.startswith("_"):
            continue
        if not isinstance(row, dict) or "status" not in row:
            bad.append(k); continue
        if row["status"] not in STATUSES:
            bad.append(k)
        else:
            counts[row["status"]] += 1
        if not row.get("source"):
            missing.append(k)
        if row["status"] == "pending_extraction":
            placeholders.append(dict(key=k, target=row.get("source"), placeholder=row.get("placeholder")))
    if verbose:
        print(f"[M0] 界線值 {sum(counts.values())} 列：anchored {counts['anchored']}、pending_extraction {counts['pending_extraction']}、study_defined {counts['study_defined']}")
        for p in placeholders:
            print(f"[M0] ※ 佔位值：{p['key']} ← 目標來源：{p['target']}；佔位 {p['placeholder']}")
        for m in missing:
            print(f"[M0] 警告：{m} 缺 source")
        for b in bad:
            print(f"[M0] 警告：{b} 缺 status 或 status 不合法")
    return dict(missing_source=missing, bad_status=bad, placeholders=placeholders, counts=counts)


def check_cohort(C, verbose=True):
    missing = []
    for grp in ("baseline_features",):
        for k, row in C[grp].items():
            if not row.get("source") or row.get("status") not in STATUSES:
                missing.append(f"{grp}.{k}")
    for k in ("susceptibility_link", "biopsy_score", "observation_scenarios"):
        if not C[k].get("source"):
            missing.append(k)
    if verbose and missing:
        print("[M0] 警告：cohort.json 缺 source/status：", missing)
    return missing


def assumptions_markdown(P, C, calib=None, gates=None):
    """assumptions.md：所有非 anchored 的參數、佔位值、校準達成值、閘門數值。"""
    L = ["# assumptions.md — SLE 減藥翻轉預警", "",
         "## 界線值總表狀態", "", "| 名稱 | status | 值 | 出處 |", "|---|---|---|---|"]
    for k, row in P.items():
        if k.startswith("_"):
            continue
        v = row.get("value")
        vs = json.dumps(v, ensure_ascii=False) if v is not None else f"null（佔位 {json.dumps(row.get('placeholder'), ensure_ascii=False)}）"
        L.append(f"| {k} | {row['status']} | {vs} | {row.get('source','')} |")
    L += ["", "## 世代生成設定（cohort.json）", "", "| 名稱 | status | 設定 | 出處 |", "|---|---|---|---|"]
    for k, row in C["baseline_features"].items():
        L.append(f"| {k} | {row['status']} | {json.dumps({a: b for a, b in row.items() if a not in ('source', 'status')}, ensure_ascii=False)} | {row['source']} |")
    ph = [k for k, r in P.items() if not k.startswith("_") and r.get("status") == "pending_extraction"]
    L += ["", f"## ※ 佔位參數（{len(ph)} 項）：{', '.join(ph)}", "", "本次結果使用佔位參數，不得對外引用。"]
    if calib:
        L += ["", "## 校準三步達成值", "", "| 步驟 | 目標 | 達成 | 備註 |", "|---|---|---|---|"]
        for row in calib:
            L.append(f"| {row['step']} | {row['target']} | {row['achieved']} | {row.get('note','')} |")
    if gates:
        L += ["", "## 品質關卡（v2 六道）", "", "| 關卡 | 值 | 條件 | 結果 |", "|---|---|---|---|"]
        for g in gates:
            L.append(f"| {g['gate']} | {g['value']} | {g['criterion']} | {'通過' if g['passed'] else '未過：' + g['fail_means']} |")
    return "\n".join(L) + "\n"
