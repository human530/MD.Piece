# -*- coding: utf-8 -*-
"""亂數分流（工作規則 6：禁止 np.random 全域介面；一律經 module_rng）。

為什麼用 SeedSequence([master, module_index, sub])：每個模組的串流由 (master, 固定索引, 子索引) 唯一決定，
新增模組只是多一個索引，既有模組的串流不會位移（tests/test_seeding.py）。"""
import numpy as np

MODULE_INDEX = {"generator": 0, "cluster": 1, "predict": 2, "ews": 3, "gates": 4, "arms": 5, "grid": 6,
                "calibration": 7, "risk": 8, "obs": 9}


def module_rng(master_seed: int, module: str, sub: int = 0) -> np.random.Generator:
    if module not in MODULE_INDEX:
        raise KeyError(f"未知模組 {module}；請在 MODULE_INDEX 末端新增，不要改動既有索引")
    return np.random.default_rng(np.random.SeedSequence([int(master_seed), MODULE_INDEX[module], int(sub)]))
