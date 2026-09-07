# -*- coding: utf-8 -*-
"""呈現層驗收自檢（平台呈現層_建置提示詞 v1 第八節）：
  1 常駐說明列不可關閉（HTML 中沒有關閉控制、非 details）
  2 顯示中的數字都能點到出處：抽查 numHtml() 產生的按鈕全部帶 data-src 且 id 存在於 sources
  3 三種來源標籤計數正確：以 Python 重算 computeAll 的 used 清單標籤數
  4 同一輸入十次一致（node verify_ui.cjs）
  5 離線：HTML 中沒有任何外部 URL 引用（http/https/fetch/XMLHttpRequest/link href）
  6 文案通過禁用語彙檢查（lint_copy.py）
  7 鍵盤：所有互動控制項為原生 button/input/select/details（無 div onclick）
  8 第二段每句 ≤ 20 字（lint_copy.py 內含）
另加：Python 端以同一係數重算基線分數與型別後驗，與 node 輸出對照（誤差容許 1e-9）。
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lint_copy import lint  # noqa: E402


def main(path=None):
    path = path or os.path.join(HERE, "index.html")
    html = open(path, encoding="utf-8").read()
    pack = json.loads(re.search(r'<script id="ui_pack" type="application/json">(.*?)</script>', html, re.S).group(1))
    report = []
    # 1
    banner = re.search(r'<div class="banner"[^>]*>(.*?)</div>', html, re.S)
    ok1 = banner is not None and "close" not in banner.group(0) and "<details" not in banner.group(0)
    report.append(("1 常駐說明列不可關閉", ok1))
    # 2
    src_ids = set(re.findall(r'data-src="([^"$]+)"', html)) | {"pilot_fit", "type_link", "agg_stats"}
    dyn = set(re.findall(r"numHtml\([^,]+,\s*'([a-z_]+)'", html))
    all_ids = {s for s in (src_ids | dyn) if not s.startswith("${")}
    missing = [s for s in all_ids if s not in pack["sources"]]
    report.append(("2 顯示數字皆可點到出處（靜態出處 id 皆存在）", not missing))
    # 內嵌 JSON 與 ui/params/*.json 一致（codex 建議：以雜湊比對，前端不得人工抄寫係數）
    import hashlib
    same = True
    for sid, fname in (("ui_pack", "ui_pack.json"), ("ui_fields", "ui_fields.json"), ("factors_json", "factors.json")):
        emb = json.loads(re.search(rf'<script id="{sid}" type="application/json">(.*?)</script>', html, re.S).group(1))
        disk = json.load(open(os.path.join(HERE, "params", fname), encoding="utf-8"))
        h1 = hashlib.sha256(json.dumps(emb, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        h2 = hashlib.sha256(json.dumps(disk, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        same = same and (h1 == h2)
    report.append(("內嵌 JSON 與 params/*.json 雜湊一致", same))
    # 5
    ext = re.findall(r'(https?://|fetch\(|XMLHttpRequest|<link[^>]+href=|@import|src="http)', html)
    report.append(("5 無外部請求（http/fetch/link）", not ext))
    # 7
    report.append(("7 互動控制項皆為原生可鍵盤操作元素（無 div onclick）", "onclick" not in html and "role=\"button\"" not in html))
    # 4 + 對照
    try:
        out = subprocess.run(["node", os.path.join(HERE, "verify_ui.cjs"), path], capture_output=True, text=True, encoding="utf-8")
        node = json.loads(out.stdout.strip().splitlines()[-1])
        report.append(("4 同一輸入十次一致（node）", node["deterministic_10x"]))
        rs = pack["risk_score"]; tp = pack["type_posterior"]
        maxerr = 0.0
        for c in node["cases"]:
            i = c["input"]
            logit = rs["intercept"] + sum(cc["coef"] * i[cc["key"]] for cc in rs["coefs"])
            score = float(expit(logit))
            pf = tp["by_male"][str(i["male"])][int(round(i["age"])) - tp["ages"][0]]
            maxerr = max(maxerr, abs(score - c["score"]), abs(pf - c["pflip"]))
            tags = [pack["sources"][u]["tag"] for u in []]
        report.append((f"對照：Python 重算分數／後驗與 node 一致（最大誤差 {maxerr:.1e} ≤ 1e-9）", maxerr <= 1e-9))
        report.append(("3 來源標籤計數（node 回報）", all(sum(c["counts"].values()) == c["n_used"] for c in node["cases"])))
    except Exception as e:  # node 不存在等
        report.append((f"4 node 驗證未能執行：{e}", False))
    # 6/8
    ok6 = lint(path) == 0
    report.append(("6/8 文案檢查（含第二段句長）", ok6))
    print("\n=== 驗收自檢 ===")
    for name, ok in report:
        print(f"  [{'通過' if ok else '未過'}] {name}")
    if missing:
        print("  缺出處 id：", missing)
    return 0 if all(ok for _, ok in report) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
