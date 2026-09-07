# -*- coding: utf-8 -*-
"""下載 params/manifest.json 列出的真實公開檔案 → data/raw/，逐檔寫出處帳本（URL＋SHA256＋位元組數）。
    python fetch_data.py [--only key1,key2]
manifest 由資料查證工作流的結果產生（results/data_scout.json → make_manifest.py）；
本腳本**只下載、不生成**——任何本地創造的資料都進不了帳本，也就過不了 require_real()。"""
import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from provenance import record  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
UA = {"User-Agent": "Mozilla/5.0 (research; kidney-cause pipeline; contact: local)"}


def fetch(key, url, retries=3):
    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, key)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[fetch] 已存在 {key}（{os.path.getsize(dest)/1024:.0f} KB）——不重抓，帳本沿用")
        return dest
    for k in range(retries):
        try:
            # CDC 對非瀏覽器 UA 回 403（www 主機全擋、wwwn 接受瀏覽器 UA）——用 curl 帶瀏覽器標頭
            import subprocess
            r = subprocess.run(["curl", "-sfL", "--max-time", "300", "-A",
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
                                "-H", "Referer: https://wwwn.cdc.gov/nchs/nhanes/",
                                "-o", dest + ".part", url])
            if r.returncode != 0 or not os.path.exists(dest + ".part") or os.path.getsize(dest + ".part") < 1024:
                raise RuntimeError(f"curl exit {r.returncode}")
            # 內容驗證：CDC 某些路徑對不存在的檔案回 200＋20KB HTML 錯誤頁，只看大小會把它存成資料。
            # SAS XPORT 檔必以 "HEADER RECORD" 開頭（2026-08-30 稽核加入）。
            with open(dest + ".part", "rb") as _f:
                magic = _f.read(13)
            if magic != b"HEADER RECORD":
                os.remove(dest + ".part")
                raise RuntimeError(f"回應不是 SAS XPORT（開頭為 {magic!r}）——可能是錯誤頁")
            os.replace(dest + ".part", dest)
            rec = record(dest, url)
            print(f"[fetch] {key}  {rec['bytes']/1024:.0f} KB  sha256 {rec['sha256'][:16]}…")
            return dest
        except Exception as e:
            print(f"[fetch] {key} 第 {k+1} 次失敗：{e}")
            time.sleep(2 * (k + 1))
    print(f"[fetch] ✗ {key} 放棄（{url}）——照實記錄，不以任何替代資料填補")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    a = ap.parse_args()
    man_p = os.path.join(ROOT, "params", "manifest.json")
    if not os.path.exists(man_p):
        raise SystemExit("尚無 params/manifest.json——先由 results/data_scout.json 產生（make_manifest.py）")
    man = json.load(open(man_p, encoding="utf-8"))
    only = set(a.only.split(",")) if a.only else None
    ok, fail = [], []
    for key, row in man["files"].items():
        if only and key not in only:
            continue
        (ok if fetch(key, row["url"]) else fail).append(key)
    print(f"[fetch] 完成 {len(ok)}；失敗 {len(fail)}" + (f"：{fail}" if fail else ""))
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
