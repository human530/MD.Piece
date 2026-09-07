# -*- coding: utf-8 -*-
"""合成世代生成器（本次建置無真實資料，故依「二、資料契約」的欄位規格生成）。

生成兩張表：
  long_df  一列一次同日成套：patient_id, panel_no, day_from_index, <各通道>；未開立者為 NaN（絕不填 0）
  pat_df   病人層級：patient_id, n_panels, ord_*（M0 的唯一特徵來源）, y_acute/y_site/y_pheno/y_box

刻意重現的兩個現實現象：
  1. **開立行為取決於臨床懷疑**（ord_* 的機率由真實標籤驅動）→ 這正是 M0 洩漏的來源，不是缺陷。
  2. **尿沉渣形態欄位多數不可得**（病人層級可得比例低）→ 觸發指示 七節的代理特徵路徑與子群比較。

封存集（抗體、免疫固定電泳、游離輕鏈、IgG4）**只生成「是否被開立」，不生成數值**——
它們在現實中被消耗於定義參考標準，本檔連值都不產生，通道消耗規則因此在資料層即成立。
整組生成器參數無文獻依據（params.json 的 generator 列 value=null），所有由此而來的數字都是該假設的產物。"""
import numpy as np
import pandas as pd

from params_io import value
from seeding import module_rng

SITES = ("腎絲球", "腎小管間質", "血管")
PHENOS = ("腎病症候群", "腎炎症候群", "孤立血尿", "慢性")


def box(acute, site, pheno):
    """Stage 4 分流格（指示 五節；R 格為合併格，不得再細分）。"""
    if site == "血管" or (acute and site == "腎絲球"):
        return "R"
    if site == "腎小管間質":
        return "A" if acute else "A2"
    if pheno == "腎病症候群":
        return "C"
    if pheno == "腎炎症候群":
        return "D"
    if pheno == "孤立血尿":
        return "F"
    return "E"


def _logit_p(rule, site, pheno, acute, base):
    z = base + rule.get("site", {}).get(site, 0.0) + rule.get("pheno", {}).get(pheno, 0.0) + (rule.get("acute", 0.0) if acute else 0.0)
    return 1.0 / (1.0 + np.exp(-z))


def simulate(P, seed=None, n=None):
    G = value(P, "generator")
    seed = int(value(P, "method.seed")) if seed is None else int(seed)
    n = int(value(P, "method.n_patients")) if n is None else int(n)
    L1, L2, ARCH = value(P, "channels_L1"), value(P, "channels_L2"), value(P, "channels_archive")
    binch = set(value(P, "binary_channels"))
    ch_cfg, eff, slp = G["channels"], G["effects"], G["slope_effects"]
    pan = G["panels"]

    r_arch = module_rng(seed, "archetype")
    r_ch = module_rng(seed, "channels")
    r_pan = module_rng(seed, "panels")
    r_ord = module_rng(seed, "ordering")
    r_mor = module_rng(seed, "morphology")

    arche = G["archetypes"]
    shares = np.array([a["share"] for a in arche], float); shares /= shares.sum()
    idx = r_arch.choice(len(arche), size=n, p=shares)
    site = np.array([arche[i]["site"] for i in idx])
    acute = np.array([arche[i]["acute"] for i in idx], int)
    pheno = np.array([arche[i]["pheno"] for i in idx])
    y_box = np.array([box(a, s, p) for a, s, p in zip(acute, site, pheno)])

    # 病人層級的真實水準與斜率
    level, slope, bin_p = {}, {}, {}
    for ch in L1 + L2:
        c, e = ch_cfg[ch], eff[ch]
        add = np.array([e["site"].get(s, 0.0) for s in site]) + np.array([e["pheno"].get(p, 0.0) for p in pheno]) + e["acute"] * acute
        if ch in binch:
            z = np.log(c["p_base"] / (1 - c["p_base"])) + add
            bin_p[ch] = 1.0 / (1.0 + np.exp(-z))
        else:
            level[ch] = c["base"] + add + r_ch.normal(0.0, c["sd"], n)
        s_cfg = slp.get(ch)
        if s_cfg:
            base_s = np.where(acute == 1, s_cfg["acute"], np.where(pheno == "慢性", s_cfg["chronic"], s_cfg["other"]))
            slope[ch] = base_s + r_ch.normal(0.0, 0.06, n)
        else:
            slope[ch] = r_ch.normal(0.0, 0.03, n)

    # 專項檢驗開立（M0 的唯一特徵來源；機率取決於真實標籤＝臨床懷疑）
    ordr = {}
    for t, rule in G["ordering"]["rules"].items():
        p = np.array([_logit_p(rule, s, ph, a, G["ordering"]["base"]) for s, ph, a in zip(site, pheno, acute)])
        ordr[t] = (r_ord.random(n) < p).astype(int)

    morph_avail = (r_mor.random(n) < ch_cfg["uRBC_dys"]["avail_patient"]).astype(int)

    # 成套（panels）
    n_panels = np.clip(1 + r_pan.poisson(pan["lam"], n), 1, pan["max"])
    rows = []
    wsd, msd = G["within_patient_sd"], G["measurement_sd"]
    for i in range(n):
        k = int(n_panels[i])
        days = np.unique(np.concatenate([[0], r_pan.integers(pan["day_range"][0], pan["day_range"][1], k - 1)])) if k > 1 else np.array([0])
        for j, d in enumerate(np.sort(days)):
            rec = dict(patient_id=i, panel_no=j + 1, day_from_index=int(d))
            for ch in L1 + L2:
                c = ch_cfg[ch]
                if ch in binch:
                    if morph_avail[i] and r_ch.random() < 0.95:
                        rec[ch] = float(r_ch.random() < bin_p[ch][i])
                    else:
                        rec[ch] = np.nan                       # 未做形態學＝NaN，不是 0
                    continue
                measured = (r_ch.random() < c["avail"]) if ch in L1 else (ordr.get(ch, np.zeros(n, int))[i] == 1)
                if not measured:
                    rec[ch] = np.nan                            # 未開立 ≠ 陰性（指示 一之五）
                    continue
                v = level[ch][i] + slope[ch][i] * (d / 365.0) + r_ch.normal(0.0, wsd) + r_ch.normal(0.0, msd)
                rec[ch] = float(v if c.get("linear") else np.exp(v))
            rows.append(rec)
    long_df = pd.DataFrame(rows)

    pat = dict(patient_id=np.arange(n), n_panels=[int(x) for x in long_df.groupby("patient_id").size().reindex(range(n)).values],
               morph_available=morph_avail, archetype=[arche[i]["name"] for i in idx],
               y_acute=acute, y_site=site, y_pheno=pheno, y_box=y_box)
    for t in list(ARCH) + list(L2):
        pat[f"ord_{t}"] = ordr.get(t, np.zeros(n, int))
    pat_df = pd.DataFrame(pat)
    assert not any(c in long_df.columns for c in ARCH), "封存集不得出現在縱向檢驗表（連數值都不生成）"
    return long_df, pat_df
