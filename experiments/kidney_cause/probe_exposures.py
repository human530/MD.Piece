# -*- coding: utf-8 -*-
"""探測 NHANES 暴露通道在十個週期的可得性，產出可併入 manifest 的清單。
    python probe_exposures.py

## 為什麼要探測而不是直接猜檔名

NHANES 的檔名跨週期不一致（PFAS 早年叫 PFC、重金屬早年在 LAB06/L06），
且 CDC 對不存在的檔案在某些路徑會回 **HTTP 200＋20KB HTML 錯誤頁**。
故一律以 `HEADER RECORD`（SAS XPORT 魔術位元組）驗證內容，不看狀態碼、不看大小。

## 探測的通道與為什麼要它

| 通道 | 為什麼是「原因」而非「結果」 |
|---|---|
| 處方藥物 RXQ_RX | 藥物性腎損傷（NSAIDs／PPI／鋰鹽／胺基醣苷）是可介入的主要病因 |
| 重金屬 PBCD／血鉛鎘汞硒 | 鎘腎病為教科書等級因果；鉛與 CKD 有 MR 證據 |
| 尿砷 UAS | 砷暴露與腎損傷；含物種分析可分辨有機／無機 |
| PFAS／PFC | 新興腎毒物 |
| 塑化劑 PHTHTE | 新興腎毒物 |
| 職業 OCQ | 職業暴露為上游變項 |
| 血壓 BPX／體位 BMX | 高血壓為腎病主因之一；本專案先前竟未納入 |
| 抽菸 SMQ／酒精 ALQ | 已知風險因子，同時為混淆變項 |
| 腎病問卷 KIQ | 自述腎病史——可作為標籤驗證與時序線索 |

血液生化不在此列——那是**下游狀態**，本專案已窮盡（見 TRAINING_SUMMARY §8）。"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
CYCLES = [("1999-2000", "1999", ""), ("2001-2002", "2001", "_B"), ("2003-2004", "2003", "_C"),
          ("2005-2006", "2005", "_D"), ("2007-2008", "2007", "_E"), ("2009-2010", "2009", "_F"),
          ("2011-2012", "2011", "_G"), ("2013-2014", "2013", "_H"), ("2015-2016", "2015", "_I"),
          ("2017-2018", "2017", "_J")]

# stem 依序嘗試（跨週期改名者列多個）
CHANNELS = {
    "藥物":     dict(stems=["RXQ_RX"], role="exposure_drug"),
    "血金屬":   dict(stems=["PBCD", "LAB06", "L06", "L06BMT"], role="exposure_metal"),
    "尿金屬":   dict(stems=["UHM", "UM", "L06UHM"], role="exposure_metal"),
    "尿砷":     dict(stems=["UAS", "UASS"], role="exposure_metal"),
    "PFAS":     dict(stems=["PFAS", "PFC", "SSPFC", "L24PFC"], role="exposure_chem"),
    "塑化劑":   dict(stems=["PHTHTE", "PHPYPA", "EPH", "L24PH"], role="exposure_chem"),
    "農藥":     dict(stems=["PP", "UPHOPM", "L26PP"], role="exposure_chem"),
    "職業":     dict(stems=["OCQ"], role="exposure_occ"),
    "血壓":     dict(stems=["BPX"], role="covariate"),
    "體位":     dict(stems=["BMX"], role="covariate"),
    "抽菸":     dict(stems=["SMQ"], role="covariate"),
    "酒精":     dict(stems=["ALQ"], role="covariate"),
    "腎病問卷": dict(stems=["KIQ_U", "KIQ"], role="label_check"),
    "醫療狀況": dict(stems=["MCQ"], role="label_check"),
}


def is_xport(url):
    """只認 SAS XPORT 魔術位元組——CDC 對不存在的檔案可能回 200＋HTML。"""
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "25", "-r", "0-12", "-A", UA,
                            "-H", "Referer: https://wwwn.cdc.gov/nchs/nhanes/", url],
                           capture_output=True)
        return r.stdout[:13] == b"HEADER RECORD"
    except Exception:
        return False


def main():
    found, missing = {}, []
    for ch, spec in CHANNELS.items():
        found[ch] = dict(role=spec["role"], files={})
        for cyc, year, sfx in CYCLES:
            hit = None
            for stem in spec["stems"]:
                for name in (f"{stem}{sfx}.xpt", f"{stem}{sfx}.XPT"):
                    url = BASE.format(year=year, name=name)
                    if is_xport(url):
                        hit = (name, url)
                        break
                if hit:
                    break
            if hit:
                found[ch]["files"][cyc] = dict(key=hit[0], url=hit[1], cycle=cyc,
                                               role=spec["role"], channel=ch)
            else:
                missing.append(f"{ch}｜{cyc}")
        n = len(found[ch]["files"])
        print(f"  {ch:10s} {n:2d}/10 週期" + ("" if n else "  ← 全無"))
    out = dict(created_note="以 HEADER RECORD 魔術位元組驗證，非狀態碼／大小",
               channels=found, missing=missing,
               n_files=sum(len(v["files"]) for v in found.values()))
    json.dump(out, open(os.path.join(ROOT, "params", "exposure_probe.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n合計可得檔案 {out['n_files']} 個；未找到 {len(missing)} 個組合")
    print("[完成] params/exposure_probe.json")


if __name__ == "__main__":
    main()
