# -*- coding: utf-8 -*-
"""平台呈現層：Python 端匯出查表包（平台呈現層_建置提示詞 v1 第四節 甲案）。

前端不得實作模型、不得硬寫任何係數／欄位／因子，所以這支腳本把介面會用到的每一個數字
連同來源標籤（抄錄自文獻／由文獻推算／本研究設定）一起匯出：
  ui/params/ui_fields.json   輸入欄位（由 params.json 產生）
  ui/params/factors.json     疾病因子（params.json 目前沒有附出處的因子 → 空清單＋說明）
  ui/params/ui_pack.json     係數、標準化統計、分層切點、各層統計、型別後驗查表、代表性序列與五個時點、τ 掃描
  ui/index.html              由 index.template.html 嵌入上述三個 JSON 產生（單一檔、離線）

執行：python ui/export_ui_pack.py [--results results/results.json] [--fast]
（--fast 跳過重算首次警報日，五個時點中的首次警報日留空——僅供版面預覽）
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from cohort import load_params, _val, make_cohort, derived_scale                     # noqa: E402
from risk import risk_score, stratify                                                # noqa: E402
from warning import run_warning                                                      # noqa: E402
from scipy.special import expit                                                      # noqa: E402

TAG_LIT, TAG_DER, TAG_SET = "抄錄自文獻", "由文獻推算", "本研究設定"


def tag_of(derived_from):
    """params.json 的 derived_from → 三分法（規格第三節第三層）。"""
    d = str(derived_from or "")
    if d in ("", "literature") or d.startswith("literature"):
        return TAG_LIT
    if d.startswith(("renormalized", "kdigo_anchors", "text_description", "figure", "matched_baseline")):
        return TAG_DER
    return TAG_SET                                                                   # assumption / calibrated_* / spec_decision / fade_default


class Sources:
    """出處登記簿：每個顯示中的數字都要指到這裡的一個 id。"""

    def __init__(self):
        self.items = {}

    def add(self, sid, tag, text, pmid=None, derivation=None, reason=None, sensitivity=None):
        self.items[sid] = dict(tag=tag, text=text, pmid=pmid, derivation=derivation, reason=reason, sensitivity=sensitivity)
        return sid


def build(results_path, fast=False):
    P = load_params(os.path.join(ROOT, "params.json"), verbose=False)
    R = json.load(open(results_path, encoding="utf-8"))
    cal = R["calibration"]["default"]
    agg = R["aggregate"]["default"]
    S = Sources()

    # ---------------- 出處：文獻與設定 ----------------
    S.add("ohare", TAG_LIT, "O'Hare AM 等 2012，Am J Kidney Dis 59(4):513-22", pmid="22305760")
    S.add("kdigo", TAG_LIT, "Stevens PE, Levin A 2013（KDIGO 2012 CKD 指引摘要）", pmid="23732715")
    S.add("tangri", TAG_LIT, "Tangri N 等 2011，JAMA 305(15):1553-9（KFRE）", pmid="21482743")
    S.add("scheffer", TAG_LIT, "Scheffer M 等 2009，Nature 461:53-9", pmid="19727193")
    S.add("xu", TAG_LIT, "Xu Z 等 2022，Int J Epidemiol 51(6):1813-23", pmid="35776101")
    S.add("share_renorm", TAG_DER, "O'Hare 2012 前三類比例（62.8／24.6／9.5%）以三類重新正規化；第四類（災難型）由翻轉型代表",
          pmid="22305760", derivation="p_k / (0.628+0.246+0.095)")
    S.add("kdigo_anchor", TAG_DER, "以 KDIGO 分級界值錨定翻轉型參考幾何：健康態 ↔ 90、摺疊點 ↔ 60，門檻 ↔ 15",
          pmid="23732715", derivation="eGFR = eGFR0 − b·(y − y_L)，b = 30/(y_fold − y_L)")
    S.add("kfre_sign", TAG_SET, "年齡／性別對易感度的方向取 KFRE 係數符號（年齡負、男性正），大小相等為本研究設定",
          pmid="21482743", reason="無可抄錄的易感度模型", sensitivity="kappa 0–1（見校準備援格點）")
    S.add("beta_hr3", TAG_SET, "每低 10 mL/min/1.73m² 的事件風險比取 KFRE 量級 HR≈3（β=ln 3）",
          pmid="21482743", reason="係數值未逐字核對，屬本研究設定", sensitivity="變體 beta_x0.5（HR 1.7）、beta_x2（HR 9）")
    S.add("pilot_fit", TAG_SET, "模組二風險分數係數：於 pilot 世代（seed+1000）以 logistic 迴歸配適一次後固定",
          reason="模擬世界自己的『已發表風險方程』；不用分析世代配適以免樂觀偏誤",
          sensitivity="各變體重新配適之係數見 results.json → calibration.*.risk_coefs")
    S.add("cohort_stats", TAG_SET, "自模擬世代（預設變體 seed 0，n=3000）計算的標準化統計與分層切點",
          reason="標準化只是為了把不同單位的輸入放到同一尺度；切點依模擬世代分數的五分位",
          sensitivity="5 個 seed 的層內統計以平均［最小, 最大］呈現")
    S.add("agg_stats", TAG_SET, "模擬世代五個 seed 的聚合統計（平均［最小, 最大］）", reason="全部 seed 都報，不挑")
    S.add("kappa", TAG_SET, f"kappa={_val(P['baseline']['kappa'])}（年齡／性別對易感度的總相關）", reason="無文獻可依", sensitivity="0–1 格點")
    S.add("type_link", TAG_SET, f"型別歸屬 logit 斜率 {_val(P['baseline']['type_link_beta'])}；邊際翻轉型比例 {_val(P['sim']['flip_share'])}（規格預設各半）",
          reason="規格決定書 v2 捌：型別由易感度決定、不得由 x0 決定")
    S.add("tau", TAG_SET, "系統鬆弛時間 τ：主分析 60 天（規格決定書 v2 壹）；14／30 天為掃描示範，六個月翻轉時程在此 τ 下不可達",
          reason="τ 決定回復速度與基線自相關", sensitivity="14／30／60 天")
    S.add("dmu", TAG_SET, f"翻轉型漂移量中位 Δμ={cal['delta_mu']['delta_mu_median']:.3f}：反推使 median(t_threshold − t_crit)=180 天（O'Hare 災難型 ≤6 個月）",
          pmid="22305760", reason="校準值", sensitivity="變體 fliptime90；12 個月目標不可達（留白）")
    S.add("lam0", TAG_SET, f"每日基礎風險 λ0={cal['hazard']['lambda0_per_day']:.2e}：校準使五年事件率 22.5%（規格 15–30% 中點）",
          reason="校準值", sensitivity="事件率目標區間 15–30%")
    S.add("noise", TAG_SET, f"個人內 eGFR 定態變異 SD {_val(P['noise']['stationary_sd_egfr'])} mL/min/1.73m²",
          reason="O'Hare 2012 僅述『substantial intraindividual variability』無數值", sensitivity="未掃描")
    S.add("onset", TAG_SET, f"漂移／下降起始日均勻分布 {_val(P['flip']['drift_onset_range_days'])} 天；漂移期 {_val(P['flip']['drift_duration_range_days'])} 天",
          reason="規格決定書 v2 玖：數個月至兩年", sensitivity="變體 linear_from_day0")
    S.add("age_dist", TAG_SET, f"年齡分布 N({_val(P['baseline']['age_mean'])}, {_val(P['baseline']['age_sd'])})，範圍 {P['baseline']['age_range']}；男性比例 {_val(P['baseline']['male_share'])}",
          reason="O'Hare 世代為 VA 高齡男性為主，摘要無平均年齡；此為一般族群假設")
    S.add("egfr_range", TAG_LIT, "基線 eGFR 範圍 15–90 mL/min/1.73m²（KDIGO G2–G4）", pmid="23732715")
    S.add("meas", TAG_SET, "量測誤差選項（0＝檢驗室、2.5、5、10＝自填量級）與量測間隔選項（1、7、30、90、180 天）",
          reason="僅用於疊圖上的觀測點示意，不進入任何計算", sensitivity="不適用")
    # 計算值（不是係數）：追溯鏈指向所用的係數與公式；不計入三類標籤的計數
    S.add("calc_score", "計算值", "基線分數：logit = 截距 + Σ 係數 × 輸入值；分數 = 1/(1+e^(−logit))。截距與係數＝pilot 配適（本研究設定），輸入值＝此組輸入。",
          derivation="見第二層表格逐列貢獻量")
    S.add("calc_z", "計算值", "標準化值 = （輸入值 − 模擬世代平均）/ 模擬世代標準差；平均與標準差為本研究設定（模擬世代統計）。")
    S.add("calc_contrib", "計算值", "貢獻量 = 係數 ×（輸入值 − 平均）；權重（標準化）= 係數 × 標準差。係數＝pilot 配適（本研究設定）。")
    S.add("calc_post", "計算值", "型別歸屬機率：生成器在給定年齡、性別下把個案指派為翻轉型的機率（對未觀測易感度積分）；由 κ、權重、θ、β_type（皆本研究設定）算出，不是配適的分類器，也不是任何人的診斷。")
    S.add("calc_stratum", "計算值", "所在五分位：以模擬世代分數的四個切點判定（切點＝本研究設定，模擬世代統計）。")
    S.add("alarm_rule", TAG_SET, "首次警報日：主要警報規則（滾動 AR(1) 與 SD 的 Kendall τ 同為正且超過共同虛無 95 分位；窗 42 天、直線去趨勢）",
          reason="規格決定書 v2 肆；虛無分布取自同一模擬世代 200 人 × 100 次區塊置換")

    # ---------------- 世代（預設變體 seed 0，與 results 相同的生成）----------------
    v = cal["variant"]
    hz = cal["hazard"]
    kw = dict(delta_mu_median=cal["delta_mu"]["delta_mu_median"], lam0=hz["lambda0_per_day"], beta=hz["beta_per_10_egfr"], kappa=hz["kappa"])
    seed0 = R["meta"]["seeds"][0]
    C = make_cohort(P, seed0, tau=v["tau"], **kw)
    coefs = cal["risk_coefs"]
    score = risk_score(C, coefs)
    q5 = stratify(score, 5)
    cut = [float(np.max(score[q5 == s])) for s in range(4)]                        # 前四層的上界 → 五分位切點
    B = P["baseline"]
    stats = dict(age=dict(mean=float(C["age"].mean()), sd=float(C["age"].std())),
                 male=dict(mean=float(C["male"].mean()), sd=float(C["male"].std())),
                 x0=dict(mean=float(C["x0"].mean()), sd=float(C["x0"].std())))

    # 各層統計（5 seed 聚合）
    def m3(x):
        return dict(mean=x["mean"], min=x["min"], max=x["max"])
    strata = []
    for s in range(5):
        row = dict(index=s,
                   event_rate=m3(agg["risk"]["event_rate_by_q5"][s]),
                   flip_share=m3(agg["risk"]["stratum_flip_share"]["q5"][s]),
                   K_A=m3(agg["clustering"]["q5_A"][s]["K"]), K_B=m3(agg["clustering"]["q5_B"][s]["K"]),
                   ari_A=m3(agg["clustering"]["q5_A"][s]["ari_vs_generator"]),
                   ceiling_A=agg["clustering"]["q5_A"][s]["k_ceiling_hit"]["mean"],
                   n=int(agg["clustering"]["q5_A"][s]["n"]["mean"]))
        strata.append(row)

    # 型別後驗查表 P(翻轉型 | 年齡, 性別)：s = kappa·u + sqrt(1−kappa²)·ε 對 ε 積分（Gauss–Hermite）
    from cohort import _thresholds_for_shares
    kappa = hz["kappa"]; bt = _val(B["type_link_beta"]); share = _val(P["sim"]["flip_share"])
    theta = float(_thresholds_for_shares(np.array([1 - share, share]), bt)[0])
    w = B["susceptibility_weights"]; p = _val(B["male_share"])
    z, wt = np.polynomial.hermite_e.hermegauss(40); wt = wt / wt.sum()
    ages = list(range(B["age_range"][0], B["age_range"][1] + 1))
    post = {}
    for male in (0, 1):
        z_male = (male - p) / np.sqrt(p * (1 - p))
        col = []
        for a in ages:
            z_age = (a - _val(B["age_mean"])) / _val(B["age_sd"])
            u = (w["age"] * z_age + w["male"] * z_male) / np.hypot(w["age"], w["male"])
            s_grid = kappa * u + np.sqrt(max(0.0, 1 - kappa ** 2)) * z
            col.append(float(np.sum(wt * (1 - expit(theta - bt * s_grid)))))
        post[str(male)] = col

    # 代表性序列：每層 × {線性型, 翻轉型-已翻轉, 翻轉型-未翻轉} × τ ∈ tau_options
    tau_opts = [60, 30, 14]
    reps = {}
    for tau in tau_opts:
        Ct = C if tau == v["tau"] else make_cohort(P, seed0, tau=tau, **kw)
        first = None
        if not fast:
            r = run_warning(Ct, P, 42, "linear", np.random.default_rng(seed0 + 99))
            first = r["_first_alarm"]
        sc_t = risk_score(Ct, coefs); qt = stratify(sc_t, 5)
        for s in range(5):
            for tname, mask in (("linear", ~Ct["is_flip"]), ("flip_tipped", Ct["is_flip"] & (Ct["t_crit"] >= 0)),
                                ("flip_stable", Ct["is_flip"] & (Ct["t_crit"] < 0))):
                idx = np.where(mask & (qt == s))[0]
                if len(idx) == 0:
                    continue
                # medoid（codex 建議）：在 風險層×型別×τ 這一格內，取「與逐日中位曲線的平方距離最小」的個案；
                # 不以五個時點是否齊全作條件（缺的時點照實顯示為無），輸入只決定查到哪一格
                med_curve = np.median(Ct["X"][idx], axis=0)
                d = ((Ct["X"][idx] - med_curve[None, :]) ** 2).mean(axis=1)
                i = int(idx[np.argmin(d)])
                reps[f"tau{tau}_q{s}_{tname}"] = dict(
                    tau=tau, stratum=s, type=tname, subject=i, egfr0=round(float(Ct["egfr0"][i]), 1),
                    series=[round(float(x), 1) for x in Ct["X"][i]],
                    t_onset=int(Ct["t_onset"][i]) if np.isfinite(Ct["t_onset"][i]) else None,
                    t_crit=int(Ct["t_crit"][i]) if Ct["t_crit"][i] >= 0 else None,
                    t_threshold=int(Ct["t_threshold"][i]) if Ct["t_threshold"][i] >= 0 else None,
                    t_event=int(Ct["t_event"][i]) if Ct["t_event"][i] >= 0 else None,
                    first_alarm=(int(first[i]) if (first is not None and first[i] >= 0) else None) if first is not None else "not_computed")

    # τ 掃描（fig6 的數字）
    scans = {}
    for tau, rows in R["tau_scans"].items():
        vals = [r["median_threshold_minus_crit_days"] for r in rows if r["median_threshold_minus_crit_days"] is not None and np.isfinite(r["median_threshold_minus_crit_days"])]
        scans[tau] = dict(max_median_days=(max(vals) if vals else None), reachable_180=bool(any(x >= 180 for x in vals)))

    coef_rows = [dict(key="age", label="年齡", unit="歲", coef=coefs["coef"][0], src="pilot_fit"),
                 dict(key="male", label="性別（男=1）", unit="", coef=coefs["coef"][1], src="pilot_fit"),
                 dict(key="x0", label="基線 eGFR", unit="mL/min/1.73m²", coef=coefs["coef"][2], src="pilot_fit")]
    presets = [dict(label="範例案例甲：60 歲女性，基線 eGFR 55", age=60, male=0, x0=55),
               dict(label="範例案例乙：72 歲男性，基線 eGFR 32", age=72, male=1, x0=32),
               dict(label="範例案例丙：45 歲女性，基線 eGFR 80", age=45, male=0, x0=80)]
    pack = dict(
        meta=dict(built_from=os.path.relpath(results_path, ROOT), results_date=R["meta"]["date"], n=R["meta"]["n"], seeds=R["meta"]["seeds"],
                  variant="default", gates=R.get("gates")),
        presets=presets,
        sources=S.items,
        risk_score=dict(intercept=coefs["intercept"], intercept_src="pilot_fit", coefs=coef_rows,
                        standardization=stats, standardization_src="cohort_stats",
                        cutpoints=cut, cutpoints_src="cohort_stats",
                        static_c=m3(agg["prediction"]["static"]["auc"]), static_c_src="agg_stats",
                        event_rate=m3(agg["cohort"]["event_rate"])),
        strata=strata, strata_src="agg_stats",
        type_posterior=dict(ages=ages, by_male=post, kappa=kappa, kappa_src="kappa", theta=theta, beta_type=bt, link_src="type_link",
                            weights=dict(age=w["age"], male=w["male"]), weights_src="kfre_sign",
                            age_mean=_val(B["age_mean"]), age_sd=_val(B["age_sd"]), male_share=p, dist_src="age_dist",
                            pre_onset_auroc=m3(agg["cohort"]["type_separability_auc_pre_onset"]) if "type_separability_auc_pre_onset" in agg["cohort"] else None),
        generator=dict(tau_options=tau_opts, tau_main=v["tau"], tau_src="tau", delta_mu=cal["delta_mu"]["delta_mu_median"], delta_mu_src="dmu",
                       lambda0=hz["lambda0_per_day"], lambda0_src="lam0", beta=hz["beta_per_10_egfr"], beta_src="beta_hr3",
                       noise_sd=_val(P["noise"]["stationary_sd_egfr"]), noise_src="noise",
                       onset_range=_val(P["flip"]["drift_onset_range_days"]), duration_range=_val(P["flip"]["drift_duration_range_days"]), onset_src="onset",
                       linear_classes=[dict(name=c["name"], share_raw=c["share_raw"], slope_mean=c["slope_mean"], slope_sd=c["slope_sd"], src="ohare")
                                       for c in P["linear_classes"]["classes"]], share_src="share_renorm",
                       geometry_src="kdigo_anchor", threshold_egfr=_val(P["scale"]["event_threshold_egfr"]), threshold_src="kdigo",
                       alarm_src="alarm_rule"),
        representatives=reps, tau_scan=scans,
        sliders=dict(meas_error_sd=[0, 2.5, 5, 10], meas_interval_days=[1, 7, 30, 90, 180], src="meas"),
        warning_summary=dict(setting="w42_linear",
                             flip_lead_to_event=m3(agg["warning"]["w42_linear"]["by_type"]["flip"]["lead_to_event_days"]["median"]),
                             linear_lead_to_event=m3(agg["warning"]["w42_linear"]["by_type"]["linear"]["lead_to_event_days"]["median"]),
                             perm_p=m3(agg["warning"]["w42_linear"]["perm_test_lead_flip_vs_linear"]["p"]),
                             flip_lead_to_crit=m3(agg["warning"]["w42_linear"]["by_type"]["flip"]["lead_to_crit_days"]["median"])),
    )
    fields = dict(fields=[
        dict(key="age", label="年齡", unit="歲", min=B["age_range"][0], max=B["age_range"][1], step=1, default=int(_val(B["age_mean"])), src="age_dist"),
        dict(key="male", label="性別", unit="", type="select", options=[dict(value=0, label="女"), dict(value=1, label="男")], default=0, src="age_dist"),
        dict(key="x0", label="基線 eGFR", unit="mL/min/1.73m²", min=P["linear_classes"]["uniform_start_range_egfr"][0],
             max=P["linear_classes"]["uniform_start_range_egfr"][1], step=1, default=45, src="egfr_range")])
    factors = dict(factors=[], note="參數檔目前沒有附出處的疾病因子；依規格（無出處者不得出現）此組為空。")
    return pack, fields, factors


def _clean(o):
    """NaN/inf 不是合法 JSON（瀏覽器 JSON.parse 會炸）：一律轉 None。"""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "results.json"))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--render-only", action="store_true", help="不重算，只用既有 ui/params/*.json 重新嵌入 index.html（改模板時用）")
    a = ap.parse_args()
    pdir = os.path.join(HERE, "params"); os.makedirs(pdir, exist_ok=True)
    if a.render_only:
        pack = json.load(open(os.path.join(pdir, "ui_pack.json"), encoding="utf-8"))
        fields = json.load(open(os.path.join(pdir, "ui_fields.json"), encoding="utf-8"))
        factors = json.load(open(os.path.join(pdir, "factors.json"), encoding="utf-8"))
    else:
        pack, fields, factors = build(a.results, fast=a.fast)
        pack, fields, factors = _clean(pack), _clean(fields), _clean(factors)
        for name, obj in (("ui_pack.json", pack), ("ui_fields.json", fields), ("factors.json", factors)):
            with open(os.path.join(pdir, name), "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    tpl = open(os.path.join(HERE, "index.template.html"), encoding="utf-8").read()
    html = (tpl.replace("/*__UI_PACK__*/", json.dumps(pack, ensure_ascii=False, separators=(",", ":")))
               .replace("/*__UI_FIELDS__*/", json.dumps(fields, ensure_ascii=False))
               .replace("/*__FACTORS__*/", json.dumps(factors, ensure_ascii=False)))
    with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"ui_pack: {len(pack['sources'])} 個出處、{len(pack['representatives'])} 條代表性序列；index.html {len(html)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
