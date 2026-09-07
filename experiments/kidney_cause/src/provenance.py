# -*- coding: utf-8 -*-
"""真實資料出處帳本——「不自製個案、不虛擬資料」的程式層保證。

規則：
  * 任何進入管線的檔案都必須先經 record()（下載當下記 URL、SHA256、位元組數、UTC 時間）。
  * require_real(path) 會重算 SHA256 與帳本比對；未登錄或雜湊不符即 raise——改過/來路不明的檔案跑不動。
  * 帳本存 results/provenance.json，隨結果一起交付，任何人可逐檔驗證。"""
import hashlib
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "results", "provenance.json")


def _sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                return h.hexdigest()
            h.update(b)


def _load():
    if os.path.exists(LEDGER):
        with open(LEDGER, encoding="utf-8") as f:
            return json.load(f)
    return {"_note": "每筆＝一個真實來源檔案；require_real() 以 SHA256 驗證。", "files": {}}


def record(path, url, extra=None):
    led = _load()
    key = os.path.basename(path)
    led["files"][key] = dict(url=url, sha256=_sha256(path), bytes=os.path.getsize(path),
                             downloaded_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             **(extra or {}))
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "w", encoding="utf-8") as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    return led["files"][key]


def require_real(path):
    """回傳帳本紀錄；未登錄或雜湊不符即 raise（鐵則一）。"""
    led = _load()
    key = os.path.basename(path)
    if key not in led["files"]:
        raise RuntimeError(f"鐵則一違反：{key} 不在真實資料出處帳本內（不得使用自製/虛擬資料）")
    rec = led["files"][key]
    h = _sha256(path)
    if h != rec["sha256"]:
        raise RuntimeError(f"鐵則一違反：{key} 內容與下載當下不符（sha256 {h[:12]}… ≠ {rec['sha256'][:12]}…）")
    return rec


def summary():
    led = _load()
    return dict(n_files=len(led["files"]),
                files={k: dict(url=v["url"], sha256=v["sha256"][:16], bytes=v["bytes"]) for k, v in led["files"].items()})
