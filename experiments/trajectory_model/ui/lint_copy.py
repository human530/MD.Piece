# -*- coding: utf-8 -*-
"""文案檢查腳本（平台呈現層_建置提示詞 v1 第一節、第七節、第八節第 6／8 項）。

檢查對象：index.html（或 index.template.html）裡 <script id="copy"> 的全部字串，加上 HTML 靜態文字。
規則（任一違反即不合格，exit 1）：
  1. 禁用語彙：驚嘆號、「讓我們一起」「接下來我們要」「首先」「總而言之」「很重要的是」「值得注意的是」
     「需要特別強調」「太棒了」「別擔心」「放心」「請安心」、行銷詞（智慧、賦能、一站式、全方位、精準守護）
  2. 表情符號
  3. 問句開場（任何以問句開頭的段落）
  4. 指涉使用者本人：「你」「您」一律不得出現（比規格的「你的風險／你會」更嚴：交付後檢查是搜尋「你的」）
  5. 區塊四第二段：每句 ≤ 20 字、不用語助詞（吧喔呢啦嘛呀哦耶囉）、不用擬人／比喻用語（小清單）、不超過三句
  6. 三段說明各不超過三句
"""
import json
import re
import sys

BANNED = ["讓我們一起", "接下來我們要", "首先", "總而言之", "很重要的是", "值得注意的是", "需要特別強調",
          "太棒了", "別擔心", "放心", "請安心", "智慧", "賦能", "一站式", "全方位", "精準守護"]
PARTICLES = list("吧喔呢啦嘛呀哦耶囉")
ANTHRO = ["累累", "回家", "休息一下", "開心", "難過", "想要", "喜歡", "生氣", "小球", "寶寶", "乖乖"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")


def sentences(s):
    return [x for x in re.split(r"[。！？!?]", s) if x.strip()]


def collect(html):
    m = re.search(r'<script id="copy" type="application/json">(.*?)</script>', html, re.S)
    copy = json.loads(m.group(1))
    texts = []

    def walk(o, path):
        if isinstance(o, str):
            texts.append((path, o))
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
    walk(copy, "")
    # 靜態 HTML 文字（去掉 script/style 與標籤）
    body = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S)
    body = re.sub(r"<[^>]+>", "\n", body)
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith(("/*", "{")):
            texts.append(("html", line))
    return copy, texts


def lint(path):
    html = open(path, encoding="utf-8").read()
    copy, texts = collect(html)
    fails = []
    for where, t in texts:
        for b in BANNED:
            if b in t:
                fails.append((where, f"禁用語彙「{b}」", t))
        if "！" in t or "!" in t:
            fails.append((where, "驚嘆號", t))
        if EMOJI.search(t):
            fails.append((where, "表情符號", t))
        if re.match(r"^\s*[^。]{0,12}[？?]", t):
            fails.append((where, "問句開場", t))
        if "你" in t or "您" in t:
            fails.append((where, "指涉使用者本人（你／您）", t))
    for key in ("p1", "p2", "p3"):
        ss = sentences(copy[key])
        if len(ss) > 3:
            fails.append((key, f"超過三句（{len(ss)} 句）", copy[key]))
    for s in sentences(copy["p2"]):
        core = re.sub(r"[，、；：「」（）\s]", "", s)
        if len(core) > 20:
            fails.append(("p2", f"句長 {len(core)} > 20 字", s))
        for p in PARTICLES:
            if p in s:
                fails.append(("p2", f"語助詞「{p}」", s))
        for a in ANTHRO:
            if a in s:
                fails.append(("p2", f"擬人／比喻用語「{a}」", s))
    print(f"檢查 {path}：{len(texts)} 段文字")
    if fails:
        for w, why, t in fails:
            print(f"  [不合格] {w}：{why} ← {t[:60]}")
        print(f"共 {len(fails)} 項不合格")
        return 1
    print("  全部通過（禁用語彙、驚嘆號、表情符號、問句開場、你／您、第二段句長／語助詞／擬人、段落句數）")
    return 0


if __name__ == "__main__":
    sys.exit(lint(sys.argv[1] if len(sys.argv) > 1 else "index.html"))
