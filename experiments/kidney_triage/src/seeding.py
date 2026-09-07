# -*- coding: utf-8 -*-
"""固定隨機種子：每個模組一條獨立串流，新增模組不位移既有串流（指示 十一節：重跑結果一致）。
禁用 np.random 全域介面。"""
from numpy.random import SeedSequence, default_rng

MODULE_INDEX = {"archetype": 0, "channels": 1, "panels": 2, "ordering": 3, "morphology": 4,
                "split": 5, "cv": 6, "bootstrap": 7, "ui": 8}


def module_rng(master_seed, module, sub=0):
    if module not in MODULE_INDEX:
        raise KeyError(f"未登記的模組：{module}（請先加入 MODULE_INDEX，勿重排既有索引）")
    return default_rng(SeedSequence([int(master_seed), MODULE_INDEX[module], int(sub)]))
