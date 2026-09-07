# 非直接抄錄自文獻的參數（assumption / calibrated_*）

| 參數 | 性質 | 現值 | 說明／出處 |
|---|---|---|---|
| scale.egfr_floor | assumption | 0 | 物理下限（eGFR 不為負）：風險在此飽和；序列本身不截斷（截斷會讓翻轉後 SD 歸零、製造假的下降型偏離） |
| linear_classes.linear_start_mode | assumption_alignment_corrected | kdigo_g2_g4_uniform | v2 貳：O'Hare 世代以開始透析為終點回顧兩年，人人最後透析；其起始水準搬到前瞻模擬必得 ~99% 事件率、靜態 C 0.97——設計不相容，非參數設錯。主分析兩型 x0 均勻取自 KDIGO G2–G4（15–90）；ohare_ranges 保留並實際跑一次作示範。 |
| flip.flip_time_target_days | calibrated_to_catastrophic_class | 180 | O'Hare 2012 災難型 <=6 個月（PMID 22305760）；v1 §1 |
| flip.delta_mu_median | calibrated_to_catastrophic_class | None | 程式啟動時以參考幾何 pilot 二分法反推 |
| flip.delta_mu_log_spread | assumption | 0.5 | assumption：個案間漂移量的對數常態離散度，delta_mu_i = median*exp(spread*s_i) |
| flip.drift_onset_range_days | assumption | [0, 1095] | v2 玖：漂移起始日均勻抽於前三年（上界另受 T − 漂移期 限制） |
| flip.drift_duration_range_days | assumption | [90, 730] | v2 玖：漂移期長度數個月至兩年 |
| noise.stationary_sd_egfr | assumption | 5.0 | assumption：個人內 eGFR 變異約 5 mL/min/1.73m2；O'Hare 2012 僅述『substantial intraindividual variability』無數值 |
| hazard.lambda0_per_day | calibrated_event_rate | None | pilot 校準（事件率目標） |
| hazard.beta_per_10_egfr | assumption | 1.0986 | assumption：每低 10 mL/min/1.73m2 的腎衰竭風險比取 KFRE 量級 HR≈3（Tangri 2011 JAMA PMID 21482743 之 eGFR 係數約 −0.56/5 單位；係數值未在本次 PubMed 查證中逐字核對，請確認） |
| hazard.h_max_per_day | assumption | 0.2 | assumption：每日風險封頂（避免 eGFR 封底處數值爆炸） |
| baseline.age_mean | assumption | 65 | assumption（O'Hare 世代為 VA 高齡男性為主，摘要無平均年齡） |
| baseline.age_sd | assumption | 12 | assumption |
| baseline.male_share | assumption | 0.5 | assumption（一般族群） |
| baseline.kappa | assumption | 0.5 | assumption：年齡／性別對易感度的總相關強度；靜態 C 若不在目標區間，改在 kappa_grid 上取最接近中點者 |
| baseline.type_link_beta | assumption | 1.0 | assumption：易感度進入型別歸屬的 logit 斜率（v2 捌：型別由易感度決定、不得由 x0 決定） |
| baseline.class_link_beta | assumption | 1.0 | assumption：易感度 s 進入子類別的累積 logit 斜率 |
| risk_score.coefficients | calibrated_pilot_fit | None | pilot 配適 |
| warning.gaussian_bw_frac | assumption | 0.25 | assumption：窗內高斯核帶寬 = 0.25*窗長 |
| warning.alarm_event_horizon_days | assumption | 365 | assumption：警報後 365 天內發生事件才算對應真實事件 |
| warning.trend_alarm_horizon_days | assumption | 365 | assumption：H3 後半『單純追蹤趨勢』比較基準——線性外推預測 365 天內跨門檻即警報 |
| warning.trend_min_history_days | assumption | 90 | v2 陸：趨勢外推至少要 90 天歷史；期間不警報且自偽警報分母扣除 |

## 放行閘門（v2 附錄參；模組四事後重算時以新規則重跑）

| 閘門 | 值 | 條件 | 結果 |
|---|---|---|---|
| 閘門一 風險層內單一型別佔比（全部層之最大值） | 0.64 | <= 0.9 | 通過 |
| 閘門二 β=0 時動態相對靜態的 C 增益：max_L |ΔC_L|（n=12000×3 seed 平均；各地標／各 seed 見 detail） | 0.0084 {'mean': {'180': 0.0084, '365': 0.0004, '730': 0.0014}, 'per_seed': [{'180': 0.0198, '365': 0.0016, '730': 0.0003}, {'180': 0.0032, '365': 0.0016, '730': -0.0007}, {'180': 0.0021, '365': -0.0022, '730': 0.0047}]} | < 0.01 | 通過 |
| 閘門三 僅取定態期資料（起始日 >= 180 天者之前 180 天）之型別分類 AUROC | 0.502 | 0.45–0.55 | 通過 |
