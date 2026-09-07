# -*- coding: utf-8 -*-
"""參數載入與出處檢核（指示 一之七：無文獻依據的參數一律留 null，啟動時列印警告，不得自行填值）。

params.json 是全案唯一數值真相；status ∈ {anchored, pending_extraction, study_defined}。
pending_extraction 的 value 為 null，取值時回傳 placeholder，並在啟動、報告、介面三處標示。"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_DIR = os.path.join(ROOT, "params")
STATUSES = ("anchored", "pending_extraction", "study_defined")


def load(name="params.json"):
    with open(os.path.join(PARAMS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def _row(P, key):
    node = P
    for part in key.split("."):
        node = node[part]
    return node


def value(P, key):
    """取值；pending_extraction 回傳 placeholder（呼叫端須知這是佔位假設）。"""
    row = _row(P, key)
    if row.get("status") == "pending_extraction":
        return row["placeholder"]
    return row["value"]


def _walk(node, prefix=""):
    if isinstance(node, dict) and "status" in node and "source" in node:
        yield prefix, node
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if not k.startswith("_"):
                yield from _walk(v, f"{prefix}.{k}" if prefix else k)


def check(P, verbose=True):
    rows = dict(_walk(P))
    counts = {s: 0 for s in STATUSES}
    bad, placeholders = [], []
    for k, row in rows.items():
        st = row.get("status")
        if st not in STATUSES:
            bad.append(k)
            continue
        counts[st] += 1
        if st == "pending_extraction":
            if row.get("value") is not None:
                bad.append(f"{k}（pending 但 value 非 null）")
            placeholders.append(k)
        if st != "pending_extraction" and not row.get("source"):
            bad.append(f"{k}（缺 source）")
    if verbose:
        print(f"[參數] {sum(counts.values())} 列：anchored {counts['anchored']}、pending_extraction {counts['pending_extraction']}、study_defined {counts['study_defined']}")
        for k in placeholders:
            print(f"[參數] ※ 警告：{k} 無文獻依據（value=null），本次以 placeholder 佔位執行——所產生之數字為該假設的產物，不得作為對真實族群的推論。")
            print(f"        待補來源：{rows[k]['source']}")
        for b in bad:
            print(f"[參數] 錯誤：{b}")
    return dict(counts=counts, placeholders=placeholders, bad=bad, rows=list(rows))


def provenance_table(P):
    """供報告與介面使用的出處表。"""
    out = []
    for k, row in _walk(P):
        out.append(dict(key=k, status=row["status"], source=row.get("source"),
                        value=row.get("value") if row.get("status") != "pending_extraction" else None,
                        has_placeholder=row.get("status") == "pending_extraction"))
    return out
