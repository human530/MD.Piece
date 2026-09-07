# -*- coding: utf-8 -*-
"""視覺化匯出：讀 results/{report,m0_baseline,thresholds,models}.json＋feature_importance.csv＋params 出處表，
嵌入 template.html 產生單一離線 ui/index.html。前端不硬寫任何係數／門檻——一律來自本資料包。
    python ui/export_viz.py"""
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import params_io as PIO  # noqa: E402

FORBIDDEN = re.compile(r"你的|您的|我的|紅燈|綠燈|黃燈|警示燈|等第|評級")


def main():
    res = os.path.join(ROOT, "results")
    R = json.load(open(os.path.join(res, "report.json"), encoding="utf-8"))
    M0 = json.load(open(os.path.join(res, "m0_baseline.json"), encoding="utf-8"))
    TH = json.load(open(os.path.join(res, "thresholds.json"), encoding="utf-8"))
    MD = json.load(open(os.path.join(res, "models.json"), encoding="utf-8"))
    P = PIO.load()
    imp = list(csv.DictReader(open(os.path.join(res, "feature_importance.csv"), encoding="utf-8-sig")))
    top_imp = {}
    for row in imp:
        key = (row["stage"], row["klass"])
        top_imp.setdefault(key, []).append((row["feature"], float(row["coef"])))
    imp_pack = {}
    for (s, k), rows in top_imp.items():
        rows.sort(key=lambda r: -abs(r[1]))
        imp_pack.setdefault(s, {})[k] = [dict(feature=f, coef=round(c, 4)) for f, c in rows[:8]]
    pack = dict(report=R, m0=M0, thresholds={k: v for k, v in TH.items() if k != "curves"}, curves=TH["curves"],
                models=MD, importance=imp_pack, provenance=PIO.provenance_table(P),
                meta=dict(date="2026-08-28", generator_warning=(
                    "本次建置無真實資料：世代由合成生成器產生，生成器全部參數無文獻依據（params.json generator=null，佔位假設執行）。"
                    "本頁所有數字僅示範管線與方法學，不構成對任何真實族群的證據。")))
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html"), encoding="utf-8").read()
    bad = FORBIDDEN.findall(re.sub(r"<script[\s\S]*?</script>", "", tpl))
    if bad:
        raise SystemExit(f"[lint] 模板含禁用語彙：{sorted(set(bad))}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    open(out, "w", encoding="utf-8").write(tpl.replace("__PACK_JSON__", json.dumps(pack, ensure_ascii=False, separators=(",", ":"))))
    print(f"[ui] index.html {os.path.getsize(out)/1024:.0f} KB")


if __name__ == "__main__":
    main()
