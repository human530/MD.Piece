# -*- coding: utf-8 -*-
"""超音波佐證 harness——沒有真實影像與帳本登錄就拒跑（鐵則一）。
    python us_validation/harness.py   # 資料未到位時：印出缺什麼，exit 2
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    man = os.path.join(HERE, "data", "manifest.csv")
    if not os.path.exists(man):
        print("[超音波佐證] 尚無真實影像資料：找不到 us_validation/data/manifest.csv。")
        print("             需要：manifest.csv（格式見 README）＋ images/ 影像檔＋出處帳本登錄。")
        print("             本 harness 不會生成任何假影像或假一致性數字。")
        return 2
    from provenance import require_real
    require_real(man)                                    # 未登錄即 raise
    import csv
    rows = list(csv.DictReader(open(man, encoding="utf-8")))
    need = {"image_file", "patient_key", "us_cause_label"}
    if not rows or not need <= set(rows[0]):
        print(f"[超音波佐證] manifest 欄位不足，需含 {sorted(need)}"); return 2
    missing = [r["image_file"] for r in rows if not os.path.exists(os.path.join(HERE, "data", "images", r["image_file"]))]
    if missing:
        print(f"[超音波佐證] {len(missing)} 張影像檔缺失（例 {missing[:3]}）"); return 2
    print(f"[超音波佐證] 資料到位：{len(rows)} 列。下一步：與 results/predictions.csv 以 patient_key 併接，"
          "算 Cohen's κ／逐類混淆／分歧解剖（見 README 第 1–3 點）。")
    # 一致性分析在資料真正到位後實作於此——刻意不先寫死，避免對想像中的欄位編碼做假設。
    return 0


if __name__ == "__main__":
    sys.exit(main())
