# -*- coding: utf-8 -*-
"""執行 ExWAS：全體成人、197 個暴露、四層調整、對照檢查優先。
    python run_exwas.py [--seed 20260830]

## 執行順序（**對照檢查在看任何結果之前**）

1. 建全體成人暴露世代（`src/exposure_cohort.py`）
2. 補齊種族變數（DEMO 的 RIDRETH1／RIDRETH3——先前以 0 佔位，是實質缺口）
3. 逐暴露掃描：M0 粗關聯 → M1 人口學 → M2 ＋體位抽菸 → M3 ＋糖尿病高血壓
4. BH-FDR 校正（在 M3 的 p 值上）
5. **對照檢查**——決定整批結果可不可信
6. 輸出帶證據等級的清單

## 報告紀律

* 陽性對照掃不出 → 整批標記為「方法無偵測力」，**不報個別發現**
* 陰性對照顯著 → 整批降級為「疑似就醫行為混淆」
* 血中金屬即使顯著也標「反向因果高風險」，不列為因果證據
* 尿液暴露的未校正版與 `_percr` 版並列，方向不一致者標記
"""
import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from exposure_cohort import build as build_exposure  # noqa: E402
from nhanes_cohort import RAW, _read                 # noqa: E402
import exwas                                         # noqa: E402
import pipeline as PL                                # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def attach_race(df, man, verbose=True):
    """補齊種族——RIDRETH1（1999–2018 皆有）／RIDRETH3（2011+）。

    先前以 0 佔位使 M1 的「調整種族」形同虛設。種族在腎病分析中同時關聯
    eGFR 公式沿革、鉛暴露與就醫可及性，是實質混淆項。"""
    frames = []
    for key, v in man["files"].items():
        if not key.upper().startswith("DEMO"):
            continue
        if not os.path.exists(os.path.join(RAW, key)):
            continue
        try:
            f = _read(key)
        except Exception:
            continue
        col = next((c for c in ("RIDRETH1", "RIDRETH3") if c in f.columns), None)
        if col:
            frames.append(f[["SEQN", col]].rename(columns={col: "RIDRETH"}))
    if not frames:
        if verbose:
            print("[種族] ❌ 找不到 RIDRETH——維持 0 佔位，M1 的種族調整無效，須於報告標明")
        return df, False
    r = pd.concat(frames, ignore_index=True).drop_duplicates("SEQN")
    df = df.drop(columns=[c for c in ("race_black", "race_hisp") if c in df.columns])
    df = df.merge(r, on="SEQN", how="left")
    # RIDRETH1: 1=墨西哥裔 2=其他西語裔 3=非西語裔白人 4=非西語裔黑人 5=其他
    df["race_black"] = (df["RIDRETH"] == 4).astype(float)
    df["race_hisp"] = df["RIDRETH"].isin([1, 2]).astype(float)
    if verbose:
        print(f"[種族] ✅ 已補齊：黑人 {int(df['race_black'].sum()):,}／"
              f"西語裔 {int(df['race_hisp'].sum()):,}／有值 {int(df['RIDRETH'].notna().sum()):,}")
    return df, True


def run(seed=20260830, verbose=True):
    man = json.load(open(os.path.join(ROOT, "params", "manifest.json"), encoding="utf-8"))
    P = json.load(open(os.path.join(ROOT, "params", "design.json"), encoding="utf-8"))
    E = build_exposure(P, verbose=verbose)
    df, exposures = E["cohort"], E["exposures"]
    df, race_ok = attach_race(df, man, verbose)

    # 藥物暴露另加「總用藥數」調整層（反向因果防線）
    drug_expo = [e for e in exposures if e.startswith("藥")]
    if "藥_總用藥數" in df.columns:
        exwas.ADJ_SETS["M4_藥物＋總用藥數"] = (
            exwas.ADJ_SETS["M3_＋糖尿病高血壓"] + ["藥_總用藥數"])

    if verbose:
        print(f"\n[掃描] {len(exposures)} 個暴露 × {len(exwas.ADJ_SETS)} 層調整"
              f"（結果＝腎損傷，全體成人 {int(df['kidney_damage'].notna().sum()):,} 人可判定）")
        print(f"        其中藥物暴露 {len(drug_expo)} 個將額外跑 M4（調整總用藥數）")
    t0 = time.time()
    rows = exwas.scan(df, exposures, outcome="kidney_damage", seed=seed, verbose=verbose)

    # ── 對照檢查：在報告任何個別發現之前
    verdict = exwas.control_check(rows, verbose=verbose)

    # 砷內建陰性對照（砷貝他因＝海鮮來源的無毒砷形式）
    ab = next((r for r in rows if r["exposure"] == "URXUAB"), None)
    as_tox = [r for r in rows if r["exposure"].startswith("URXUAS") or
              r["exposure"] in ("URXUDMA", "URXUMMA")]
    arsenic_check = None
    if ab and as_tox:
        # 2026-08-31 修正：先前只看顯著性不看**方向**。
        # 毒性砷形式若呈「保護」方向（OR<1），那與毒性無關——
        # 那是尿液排泄減少的假象（eGFR↓ → 尿中排出減少 → 濃度看似較低）。
        tox_sig = [r["exposure"] for r in as_tox if r["significant_fdr05"]]
        tox_harmful = [r["exposure"] for r in as_tox
                       if r["significant_fdr05"] and r["headline"]["or_per_sd"] > 1]
        tox_protective = [r["exposure"] for r in as_tox
                          if r["significant_fdr05"] and r["headline"]["or_per_sd"] < 1]
        arsenic_check = dict(
            arsenobetaine_significant=bool(ab["significant_fdr05"]),
            arsenobetaine_q=ab["q_bh"], toxic_species_significant=tox_sig,
            toxic_species_harmful_direction=tox_harmful,
            toxic_species_protective_direction=tox_protective,
            interpretation=(
                "⚠️ 砷貝他因（無毒海鮮形式）也顯著——砷的訊號可能是飲食混淆而非毒性"
                if ab["significant_fdr05"] else
                "✅ 砷貝他因不顯著、毒性形式呈有害方向——支持真實毒性效應" if tox_harmful else
                "❌ 毒性形式雖顯著但呈**保護**方向——與毒性不符，"
                "係尿液排泄減少之假象（eGFR↓→尿中濃度↓），不可解讀為砷無害或有益"
                if tox_protective else
                "毒性形式未達顯著——無可判讀之砷訊號"))
        if verbose:
            print(f"\n[砷內建對照] {arsenic_check['interpretation']}")

    # ── 尿液校正一致性：未校正 vs _percr 方向不一致者標記
    inconsistent = []
    for r in rows:
        e = r["exposure"]
        if e.endswith("_percr"):
            continue
        m = next((x for x in rows if x["exposure"] == f"{e}_percr"), None)
        if m and r["significant_fdr05"] != m["significant_fdr05"]:
            inconsistent.append(dict(exposure=e,
                                     raw_sig=r["significant_fdr05"], raw_q=r["q_bh"],
                                     percr_sig=m["significant_fdr05"], percr_q=m["q_bh"]))

    sig = [r for r in rows if r["significant_fdr05"] and r["survives_full_adjustment"]]
    out = dict(
        analysis="ExWAS：腎損傷的上游暴露掃描", seed=seed,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cohort=dict(**E["counts"], race_adjustment_valid=race_ok),
        design=dict(population="全體成人（**不對結果條件化**）",
                    outcome="kidney_damage＝eGFR<60 或 ACR≥30",
                    adjustment_layers=list(exwas.ADJ_SETS),
                    fdr="Benjamini-Hochberg，於 M3 的雙尾 p 值上"),
        control_check=verdict, arsenic_builtin_control=arsenic_check,
        urinary_correction_inconsistent=inconsistent,
        n_scanned=len(rows), n_significant_fdr05=sum(1 for r in rows if r["significant_fdr05"]),
        n_surviving_full_adjustment=len(sig),
        results=rows,
        reporting_rule=("陽性對照未命中→整批不報個別發現；陰性對照命中→整批降級；"
                        "血中金屬即使顯著亦標反向因果高風險，不列為因果證據"))
    PL._dump(out, "exwas.json")
    with open(os.path.join(RESULTS, "runs_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(kind="EXWAS", at=out["created"], n_exposures=len(rows),
                                n_sig=out["n_significant_fdr05"],
                                n_survive=len(sig), control=verdict["interpretation"]),
                           ensure_ascii=False) + "\n")
    if verbose:
        print(f"\n[掃描完成] {len(rows)} 個暴露、{time.time()-t0:.0f} 秒")
        print(f"  FDR<0.05：{out['n_significant_fdr05']}｜其中通過完整調整：{len(sig)}")
        if verdict["has_power"] and verdict["bias_free"]:
            print(f"\n  前 15 個通過完整調整者：")
            for r in sig[:15]:
                h = r["headline"]
                print(f"   {r['exposure']:22s} OR {h['or_per_sd']:.3f} "
                      f"[{h['ci'][0]:.3f},{h['ci'][1]:.3f}]  q={r['q_bh']:.2e}  "
                      f"反向風險={r['reverse_causation_risk']}")
        else:
            print(f"\n  ⚠️ 對照檢查未通過，依報告紀律**不列出個別發現**。")
            print(f"     {verdict['interpretation']}")
        print("\n[完成] results/exwas.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260830)
    run(ap.parse_args().seed)
