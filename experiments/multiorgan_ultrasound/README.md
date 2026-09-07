# 多器官真實超音波驗證

本實驗把超音波擴成獨立的「影像佐證層」，不取代
`experiments/kidney_cause` 的核心血液模型：真實資料、標籤來源不得作特徵、
病人層級切分、所有試驗完整留痕、AUC 0.90 是設計目標而不是保證。

不同資料集沒有可連結的同一批病人，因此目前只做各器官的獨立外部任務；
不得把乳房、心臟或胎兒超音波的 AUC 冒充腎病病因 AUC。

## 第一階段：BUS-BRA

- 來源：Zenodo DOI [`10.5281/zenodo.8231412`](https://doi.org/10.5281/zenodo.8231412)
- 真實資料：1,875 張乳房超音波、1,064 位病人，含活檢病理、BI-RADS、病灶框與遮罩
- 授權：CC BY 4.0；下載檔 MD5 `1f8b2be6476d58fc97bfb5e5a1ea9bab`
- 任務：良性 vs 惡性；以官方 `kFold` 做病人層級五折 out-of-fold 評估
- 主分析：整張影像；专家病灶框、遮罩形态／纹理及其组合仅作事前登记的敏感度分析
- 指標：病人層級 AUROC（bootstrap 95% CI）、平衡正確率、Brier score

```powershell
python -m pytest tests -q
python run.py
```

資料與特徵快取在 `data/`，不進 Git。每次模型結果都追加至
`results/runs_log.jsonl`，不得刪除失敗或低分設定。

## 锁定外部验证：TCIA BREAST-LESIONS-USG

- 来源：TCIA DOI [`10.7937/9WKK-Q141`](https://doi.org/10.7937/9WKK-Q141)，CC BY 4.0
- 队列：256 位患者；预先排除 4 个无病灶遮罩的正常病例，锁定 154 良性与 98 恶性病例
- 标签确认：197 位经活检、55 位经追踪确认
- 规程：只把 BUS-BRA 训练完成的固定模型套用一次，不以 TCIA 调参或校准
- 限制：形态模型需要专家遮罩，因此属于方法上限／敏感性分析，不是无需人工标注的部署模型

锁定规则与档案校验和见 `params/external_test_lock.json`；结果写入后程式会拒绝重复开启测试集。

```powershell
python external_validate.py
```

实际一次性结果为 AUROC `0.807`（95% CI `0.753–0.859`），未达到 0.90。
这项结果保留为域偏移证据，不再以 TCIA 调参。

## 跨设备开发：BUS-UCLM

- 原始来源 DOI `10.17632/7fvgj4jsp7.3`；因 Mendeley 直链回传 403，改用固定的公开 HF 镜像 revision `5874ae42...`
- 683 张影像、38 位患者；良恶性病灶队列经品质审计后为 260 张、35 位患者
- `HESN_002` 至 `HESN_005` 的影像与遮罩尺寸不一致且无配准资讯，因此明确排除
- 五位患者同时有良性与恶性病灶，故以患者分组切折，并用患者群集 bootstrap 计算 CI
- 完整 18 特征与域不变 13 特征的 AUROC 均为 `0.925`，但 CI 很宽（约 `0.84–0.97`）

```powershell
python domain_development.py
```

OASBUD 原始 RF 档已通过官方 MD5，但 MAT 结构没有提供文献所述 78 位女性的映射，
因此不符合严格患者级外部测试，只保留作 RF 方法开发。

## 甲状腺病理超音波

- 来源：Figshare DOI `10.6084/m9.figshare.27021604.v1`，CC BY 4.0
- Batch 1：6,005 张、601 位病例（218 良性／383 恶性），仅用于开发
- Batch 2：排除 8 张缺病理标签影像后为 2,495 张、241 位病例，保持锁定
- 两个官方 RAR 的 MD5 均与 Figshare 完全一致
- 冻结 ImageNet ResNet-18 全影像病例级 OOF AUROC `0.484`；mean/max/mean+max pooling 与 gated attention 仍约 `0.44–0.49`

这些负结果显示自然影像固定特征无法直接迁移到甲状腺病理。Batch 2 尚未开启；
下一项方法改良必须使用超音波专用预训练或端到端微调，而不是继续试 pooling 或翻转标签。

USF-MAE 的 checksum 鎖定 ViT-B/16 pilot 同樣未通過門檻：patch-token mean AUROC `0.454`、
CLS-token AUROC `0.447`。因此停止完整 6,005 張特徵抽取，Batch 2 仍未開啟。

```powershell
python thyroid_usf_mae_development.py --max-images-per-patient 3
```

## AUL 肝臟病灶開發

- 來源：Zenodo DOI `10.5281/zenodo.7272660`；三個官方壓縮檔 MD5 均核對一致
- 主任務：良性 vs 惡性，635 位病人各一張影像；先鎖定 127 人保留集，再做任何特徵抽取
- 開發集：508 人；v4 共保留 64 組線性、RBF、HGB、ExtraTrees 與 HOG/radiomics 組合
- 最佳：全圖＋ROI＋擴充 radiomics 的 RBF C=10，OOF AUROC `0.874`（95% CI `0.837–0.907`）
- 結論：未達預設 `0.90` gate；鎖定保留集保持未評估，禁止執行 `--final`
- 權利：Zenodo 頁面要求聯絡作者、引用與適當致謝；原始資料不得重新散布

```powershell
python aul_liver_development.py
# 僅在模型鎖狀態已達 gate 並凍結後，才允許一次性執行：
python aul_liver_development.py --final
```

## 擴充路線

詳見 `params/data_sources.json`。優先順序是：

1. BUS-BRA 建立可重現的疾病分類基線。
2. 取得 GPU 後，在 AUL/甲狀腺開發分割做端到端或病灶定位式微調；不可使用鎖定集選模型。
3. URI-CADS 真實腎臟病理影像、Open Kidney Dataset 與 EchoNet-Dynamic 需先完成各自的人工註冊／使用條款。
4. 只有拿到「同一病人的檢驗＋超音波」資料後，才允許血液與影像融合。

本機沒有 CUDA GPU；完整多器官深度訓練需遠端 GPU。Hugging Face 目前帳號不是 Pro，
所以先用已安裝的 ImageNet ResNet-18 固定特徵＋簡單分類器建立 CPU 基線。
