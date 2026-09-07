# -*- coding: utf-8 -*-
"""打包 report.json ＋ 出處帳本 ＋ 歸因等級（src/anchor_map）＋ 疊代測試 ＋ 免疫標籤修正 → 單一離線 ui/index.html。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from anchor_map import as_dict as anchors_as_dict, LEVELS   # noqa: E402


def load(name, sub="results"):
    p = os.path.join(ROOT, sub, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


R = load("report.json")
prov = load("provenance.json")["files"]
lab = load("lab_report.json")
hold = load("holdout_eval.json")
prox = load("nephritis_proxy_repeated_eval.json")
proxy_summary = None
if prox:
    proxy_summary = dict(
        cohort=prox["cohort"], protocol={k: prox["protocol"][k] for k in ("repeats", "outer_folds", "threshold_selection")},
        models={mk: {k: v["patient_pooled"][k] for k in ("auroc", "auprc", "balanced_accuracy")}
                | {"ci": v["patient_pooled"]["auroc_ci95_patient_bootstrap"]}
                for mk, v in prox["evaluations"].items()},
        artifact=prox.get("artifact", {}))

pack = dict(report=R,
            provenance_files={k: dict(url=v["url"], sha256=v["sha256"][:16], bytes=v["bytes"]) for k, v in sorted(prov.items())},
            anchors=anchors_as_dict(), anchor_levels=LEVELS,
            lab=lab, holdout=hold, proxy=proxy_summary)
tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
out = tpl.replace("__PACK_JSON__", json.dumps(pack, ensure_ascii=False, separators=(",", ":")))
open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write(out)
print(f"[ui] index.html {os.path.getsize(os.path.join(HERE, 'index.html')) / 1024:.0f} KB")
