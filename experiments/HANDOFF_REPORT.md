# 模型訓練階段交接報告

更新時間：2026-08-30（Asia/Taipei）

## 階段結論

本階段已完成目前 CPU 環境可合理執行的腎炎代理模型重複評估、乳房跨資料集驗證、
甲狀腺固定特徵評估，以及 AUL 肝臟 v4 開發。所有低分與失敗嘗試均保留。外部驗證
AUROC 0.90 尚未達成，因此研究目標仍未完成，模型不得宣稱可臨床診斷。

## 已完成與已驗證

### 腎炎／免疫相關腎損傷核心模型

- 修正 ANA 未檢者被當成陰性的標籤污染，只納入 488 位「已做 ANA 且有腎損傷」成人；91 位陽性。
- 完成 5 次 × 5 折外層評估，門檻只在內層選擇。
- 選定 HGB：AUROC 0.584（95% CI 0.517–0.649）、AUPRC 0.272、balanced accuracy 0.552。
- 已輸出研究用途完整模型 `kidney_cause/models/immune_kidney_proxy.joblib`；SHA256
  `8b57fb76b91e869213886405cb6f2dd78e31080453f797b546bde5c94668952f`。
- 這是 ANA／特定自體抗體陽性的代理模型，不是活檢確診腎炎模型。

### 多器官真實超音波

| 資料／評估 | 病人數 | AUROC（95% CI） | 狀態 |
|---|---:|---:|---|
| BUS-BRA expert-mask OOF | 1,064 | 0.931（0.914–0.948） | 內部開發 |
| TCIA 一次性外部驗證 | 252 | 0.807（0.753–0.859） | 已鎖定完成，未達 0.90 |
| BUS-UCLM grouped OOF | 35 | 0.925（0.840–0.974） | 小樣本開發 |
| 甲狀腺 ImageNet／gated MIL | 601 | 0.484／0.489 | Batch 2 未開啟 |
| 甲狀腺 USF-MAE pilot | 601 | 0.454／0.447 | 未通過擴大抽取 gate |
| AUL 肝臟 v4 開發 | 508 | 0.874（0.837–0.907） | 未通過 0.90 gate |

- USF-MAE 官方 100-epoch checkpoint 已做 SHA256 鎖定，1,796 張 label-blind 抽樣影像完成 nested patient-level 評估。
- AUL 三個官方壓縮檔 MD5 均與 Zenodo 一致；v4 的 64 組設定全數寫入結果檔。
- AUL 最佳為 whole + ROI + expanded radiomics、RBF C=10；HOG 沒有改善結果。
- AUL 127 人保留集（40 良性、87 惡性）在任何特徵抽取前已鎖定；目前無結果檔、無特徵快取，狀態為 `locked_not_evaluated`。

## 尚未完成

1. 尚無以腎活檢或專家裁決病因為標準、且可做機構外驗證的腎炎隊列；目前腎模型 AUROC 只有 0.584。
2. 影像任務尚未有新的「未碰過機構」達到 AUROC 0.90；TCIA 外部結果為 0.807。
3. AUL 開發 AUROC 0.874，未達 gate，故不得執行一次性 `--final`。
4. 甲狀腺 Batch 2 仍鎖定；固定 ImageNet 與 USF-MAE 特徵皆近隨機，尚未有可凍結的新方法。
5. 本機無 CUDA GPU，尚未完成端到端超音波微調。
6. GBCU 需由合資格 faculty 簽署機構授權；URI-CADS、Open Kidney Dataset 需人工申請／註冊；ViTaL 完整資料仍在倫理審查。
7. AUL 來源要求聯絡作者、引用與致謝；在權利確認前不可上傳原始資料或衍生模型。
8. Hugging Face 發布尚未執行；模型卡／資料卡僅為本地草稿，且目前證據與權利條件都不適合公開模型。

## 下一階段執行順序

1. 先取得 GPU，在 AUL 的既定 508 人開發集與甲狀腺 Batch 1 做端到端微調或病灶定位式監督；維持病人級切分，不使用任何鎖定集調參。
2. 同步完成 GBCU、URI-CADS、Open Kidney 的資料申請；優先取得有腎病理／專家裁決標籤且可做機構外切分的資料。
3. 只有當 AUL 開發 OOF AUROC 達 0.90、方法與門檻寫入 `aul_model_lock.json` 且狀態凍結後，才執行一次：

   ```powershell
   python experiments/multiorgan_ultrasound/aul_liver_development.py --final
   ```

   目前程式會因 gate 未達而拒絕此命令；不要繞過保護。

4. 甲狀腺同理：先在 Batch 1 凍結方法，再只開 Batch 2 一次，無論結果高低都完整報告。
5. 新腎炎資料到位後，重建「病理／裁決病因」標籤與機構外測試，不以目前 ANA proxy 調整新外部測試集。
6. 外部結果完成且資料／衍生權利書面確認後，再依 Hugging Face paper-publisher 流程發布模型卡、資料清單與論文；不重新散布受限原始資料。

## 關鍵檔案

- 腎模型與 metadata：`experiments/kidney_cause/models/`
- 腎炎重複評估：`experiments/kidney_cause/results/nephritis_proxy_repeated_eval.json`
- AUL 開發結果：`experiments/multiorgan_ultrasound/results/aul_development.json`
- AUL 模型鎖：`experiments/multiorgan_ultrasound/params/aul_model_lock.json`
- AUL 保留集鎖：`experiments/multiorgan_ultrasound/params/aul_holdout_lock.json`
- 甲狀腺 USF-MAE 結果：`experiments/multiorgan_ultrasound/results/thyroid_usf_mae_max3_development.json`
- 全部資料來源與校驗：`experiments/multiorgan_ultrasound/params/data_sources.json`
- 整合實驗摘要：`experiments/EXPERIMENT_RESULTS.md`
- 論文與 Hugging Face 草稿：`experiments/multiorgan_ultrasound/paper/`、`experiments/multiorgan_ultrasound/huggingface/`

## 重現與驗證

```powershell
python -m pytest experiments/kidney_cause/test_nephritis_proxy.py experiments/multiorgan_ultrasound/tests -q
python experiments/multiorgan_ultrasound/aul_liver_development.py
python experiments/multiorgan_ultrasound/thyroid_usf_mae_development.py --max-images-per-patient 3
```

大型原始影像、下載的 checkpoint 與特徵快取位於 Git 忽略的 `data/`，不隨 commit 發布；
移機時需依 `data_sources.json` 的固定來源與 checksum 重新取得。
