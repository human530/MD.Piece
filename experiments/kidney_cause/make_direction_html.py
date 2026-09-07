# -*- coding: utf-8 -*-
"""產生 ui/direction.html——單一離線網頁，內嵌 direction_model.json，瀏覽器內直接計算。
    python make_direction_html.py

JS 端的數學與 direction.py 逐行對應：
  raw  = mean_folds( sigmoid( ((x∨median − mean)/scale) · coef + intercept ) )
  cal  = 線性內插 raw 於 (isotonic_x, isotonic_y)，兩端夾住（同 np.interp）
  band = cal ≥ t_high → 傾向；cal ≤ t_low → 不傾向；否則不確定
  drivers = mean_folds(z·coef) 取 |值| 前四且非缺項
產生後以 verify_direction_html.py 對同一病人比對 Python 與 JS 輸出。

用詞規則（專案 UI 鐵則）：無第二人稱、無紅黃綠燈、無等級、無建議、無單一綜合分數。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from nhanes_cohort import DERIVED, FEATURE_LABELS   # noqa: E402

LABEL = {**FEATURE_LABELS, **DERIVED, "age": "年齡", "sex": "性別"}
UNIT = {
    "LBXSAL": "g/dL", "LBXSATSI": "U/L", "LBXSASSI": "U/L", "LBXSAPSI": "U/L", "LBXSBU": "mg/dL",
    "LBXSCA": "mg/dL", "LBXSCH": "mg/dL", "LBXSC3SI": "mmol/L", "LBXSGTSI": "U/L", "LBXSGL": "mg/dL",
    "LBXSIR": "ug/dL", "LBXSLDSI": "U/L", "LBXSPH": "mg/dL", "LBXSTB": "mg/dL", "LBXSTP": "g/dL",
    "LBXSTR": "mg/dL", "LBXSUA": "mg/dL", "LBXSCR": "mg/dL", "LBXSNASI": "mmol/L", "LBXSKSI": "mmol/L",
    "LBXSCLSI": "mmol/L", "LBXSOSSI": "mmol/kg", "LBXSGB": "g/dL", "LBXWBCSI": "10³/uL",
    "LBXRBCSI": "10⁶/uL", "LBXHGB": "g/dL", "LBXMCVSI": "fL", "LBXMCHSI": "pg", "LBXMC": "g/dL",
    "LBXRDW": "%", "LBXPLTSI": "10³/uL", "LBXMPSI": "fL", "LBXCRP": "mg/dL", "LBXBAP": "ug/L",
    "URXUMA": "ug/mL", "URXUCR": "mg/dL", "LBXTC": "mg/dL", "LBXTR": "mg/dL", "LBDLDL": "mg/dL",
    "LBXBPB": "ug/dL", "LBXBCD": "ug/L", "LBXFER": "ng/mL", "LBXFOL": "ng/mL", "LBXB12": "pg/mL",
    "LBXMMA": "umol/L", "LBXTHG": "ug/L", "LBXRBF": "ng/mL", "LBXCOT": "ng/mL", "LBDVIDMS": "nmol/L",
    "LBXPT21": "pg/mL", "ACR": "mg/g", "eGFR": "mL/min/1.73m²", "NLR": "比值", "age": "歲",
    "sex": "1=男 2=女",
}
GROUPS = [
    ("生化", "LBXSAL LBXSGB LBXSTP LBXSBU LBXSCR LBXSUA LBXSNASI LBXSKSI LBXSCLSI LBXSCA LBXSPH "
             "LBXSC3SI LBXSOSSI LBXSATSI LBXSASSI LBXSAPSI LBXSGTSI LBXSTB LBXSLDSI LBXSGL LBXSIR "
             "LBXSCH LBXSTR".split()),
    ("血球", "LBXWBCSI LBXLYPCT LBXMOPCT LBXNEPCT LBXEOPCT LBXBAPCT LBXRBCSI LBXHGB LBXMCVSI "
             "LBXMCHSI LBXMC LBXRDW LBXPLTSI LBXMPSI".split()),
    ("尿液與腎功能", "URXUMA URXUCR ACR eGFR".split()),
    ("脂質與發炎", "LBXTC LBXTR LBDLDL LBXCRP LBXBAP NLR".split()),
    ("營養／微量元素", "LBXFER LBXFOL LBXB12 LBXMMA LBXRBF LBDVIDMS LBXPT21 LBXCOT LBXBPB LBXBCD LBXTHG".split()),
    ("人口學", "age sex".split()),
]

HTML = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>腎損傷病因方向判定</title>
<style>
:root{--ink:#1a1a1a;--sub:#5a5a5a;--line:#d9d9d9;--bg:#fafafa;--card:#fff;--acc:#2f5f8f;--accs:#e8eef5}
*{box-sizing:border-box}body{margin:0;font:15px/1.6 "Microsoft JhengHei","PingFang TC",system-ui,sans-serif;color:var(--ink);background:var(--bg)}
header{background:var(--card);border-bottom:1px solid var(--line);padding:18px 24px}
h1{margin:0;font-size:20px}header p{margin:4px 0 0;color:var(--sub);font-size:13px}
main{display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:20px;padding:20px 24px;max-width:1280px;margin:auto}
@media(max-width:960px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px 18px}
details{border-top:1px solid var(--line);padding:8px 0}details:first-of-type{border-top:0}
summary{cursor:pointer;font-weight:600;padding:6px 0;color:var(--acc)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:8px 14px;padding:6px 0 4px}
label{display:flex;flex-direction:column;font-size:12.5px;color:var(--sub)}
label b{color:var(--ink);font-weight:500}label small{font-size:11px}
input{margin-top:3px;padding:6px 8px;border:1px solid var(--line);border-radius:5px;font:inherit;font-size:14px}
input:focus{outline:2px solid var(--accs);border-color:var(--acc)}
.bar{display:flex;gap:10px;margin:12px 0}.bar button{flex:1;padding:9px;border:1px solid var(--line);border-radius:6px;background:var(--card);font:inherit;cursor:pointer}
.bar button.p{background:var(--acc);color:#fff;border-color:var(--acc)}
#out{position:sticky;top:16px}#out h2{font-size:16px;margin:0 0 4px}#out .note{color:var(--sub);font-size:12.5px;margin:0 0 12px}
.axis{border:1px solid var(--line);border-radius:6px;padding:12px 14px;margin-bottom:12px}
.axis h3{margin:0 0 6px;font-size:15px;display:flex;justify-content:space-between;align-items:baseline}
.axis h3 span{font-size:13px;color:var(--sub);font-weight:400}
.meter{height:10px;background:var(--accs);border-radius:5px;overflow:hidden;margin:6px 0}
.meter i{display:block;height:100%;background:var(--acc)}
.band{font-weight:700;font-size:15px}.prob{color:var(--sub);font-size:13px}
.drv{margin:8px 0 0;padding:0;list-style:none;font-size:13px}.drv li{display:flex;justify-content:space-between;padding:2px 0;border-top:1px dashed var(--line)}
.drv li span:last-child{color:var(--sub)}.miss{font-size:12px;color:var(--sub);margin-top:6px}
.empty{color:var(--sub);text-align:center;padding:40px 10px}
footer{color:var(--sub);font-size:12px;padding:12px 24px 30px;max-width:1280px;margin:auto;border-top:1px solid var(--line)}
</style></head><body>
<header><h1>腎損傷病因方向判定</h1>
<p>輸入常規檢驗數值，輸出感染／代謝兩軸的「大概方向」。留空的項目以訓練集中位數補入並標明。非診斷工具。</p></header>
<main>
<section class="card">
  <div class="bar"><button class="p" id="run">計算方向</button><button id="demo">載入示範數值</button><button id="clr">清除</button></div>
  <div id="form"></div>
</section>
<aside class="card" id="out"><div class="empty">尚未輸入數值</div></aside>
</main>
<footer>模型：NHANES 1999–2018，n=8,983。感染軸 OOF AUROC __AUC_INF__、代謝軸 __AUC_MET__。門檻依盛行率倍數事前定義（傾向 ≧2×、不傾向 ≦0.5×）。免疫軸因常規檢驗無訊號而未納入。輸出為排序線索，不構成診斷或建議。</footer>
<script>
const M=__MODEL__, L=__LABELS__, U=__UNITS__, G=__GROUPS__, DEMO=__DEMO__;
const all=[...new Set(Object.values(M.axes).flatMap(a=>a.features))];
const med={};for(const a of Object.values(M.axes))a.features.forEach((f,i)=>{med[f]=a.ensemble.reduce((s,fp)=>s+fp.medians[i],0)/a.ensemble.length});
const form=document.getElementById('form');
for(const [g,fs] of G){const d=document.createElement('details');d.open=(g==='生化'||g==='血球');
 d.innerHTML=`<summary>${g}（${fs.length}）</summary><div class="grid">`+fs.map(f=>`<label><b>${L[f]||f}</b><small>${f}${U[f]?'　'+U[f]:''}　中位 ${fmt(med[f])}</small><input type="number" step="any" data-f="${f}" placeholder="留空＝中位數"></label>`).join('')+'</div>';form.appendChild(d)}
function fmt(v){return v==null||isNaN(v)?'—':(Math.abs(v)>=100?v.toFixed(0):Math.abs(v)>=10?v.toFixed(1):v.toFixed(2))}
function read(){const o={};form.querySelectorAll('input').forEach(i=>{if(i.value!=='')o[i.dataset.f]=+i.value});return o}
function sig(t){return 1/(1+Math.exp(-t))}
function interp(x,xs,ys){if(x<=xs[0])return ys[0];if(x>=xs[xs.length-1])return ys[ys.length-1];let i=1;while(xs[i]<x)i++;const t=(x-xs[i-1])/(xs[i]-xs[i-1]);return ys[i-1]+t*(ys[i]-ys[i-1])}
function predict(v){const res={};for(const [name,a] of Object.entries(M.axes)){const ff=a.features,n=ff.length;const x=ff.map(f=>f in v?v[f]:NaN);const miss=ff.filter((f,i)=>isNaN(x[i]));
 let raw=0;const contrib=new Array(n).fill(0);
 for(const fp of a.ensemble){let t=fp.intercept;for(let i=0;i<n;i++){const xi=isNaN(x[i])?fp.medians[i]:x[i];const z=(xi-fp.mean[i])/fp.scale[i];t+=z*fp.coef[i];contrib[i]+=z*fp.coef[i]}raw+=sig(t)}
 raw/=a.ensemble.length;for(let i=0;i<n;i++)contrib[i]/=a.ensemble.length;
 const cal=interp(raw,a.isotonic_x,a.isotonic_y);const band=cal>=a.t_high?'傾向':cal<=a.t_low?'不傾向':'不確定';
 const order=[...contrib.keys()].filter(i=>!isNaN(x[i])).sort((p,q)=>Math.abs(contrib[q])-Math.abs(contrib[p])).slice(0,4);
 res[name]={cal,prev:a.prevalence,times:cal/a.prevalence,band,miss:miss.length,n,drivers:order.map(i=>({f:ff[i],name:L[ff[i]]||ff[i],val:x[i],push:contrib[i]>0?'→推向':'←推離'}))}}return res}
function render(res){const o=document.getElementById('out');o.innerHTML='<h2>方向判定</h2><p class="note">大概方向，非診斷。每軸獨立判定。</p>'+Object.entries(res).map(([k,a])=>{const w=Math.min(a.cal/(4*a.prev),1)*100;
 return `<div class="axis"><h3>${k}方向<span>盛行率 ${(a.prev*100).toFixed(1)}%</span></h3><div class="meter"><i style="width:${w.toFixed(0)}%"></i></div><div class="band">${a.band}</div><div class="prob">校準機率 ${a.cal.toFixed(4)}　盛行率 ×${a.times.toFixed(1)}</div><ul class="drv">${a.drivers.map(d=>`<li><span>${d.push} ${d.name}</span><span>${fmt(d.val)}</span></li>`).join('')}</ul>${a.miss?`<div class="miss">缺 ${a.miss}/${a.n} 項，以中位數補入</div>`:''}</div>`}).join('')}
document.getElementById('run').onclick=()=>{const v=read();if(!Object.keys(v).length){document.getElementById('out').innerHTML='<div class="empty">尚未輸入數值</div>';return}render(predict(v))};
document.getElementById('demo').onclick=()=>{form.querySelectorAll('input').forEach(i=>{i.value=i.dataset.f in DEMO?DEMO[i.dataset.f]:''});render(predict(DEMO))};
document.getElementById('clr').onclick=()=>{form.querySelectorAll('input').forEach(i=>i.value='');document.getElementById('out').innerHTML='<div class="empty">尚未輸入數值</div>'};
</script></body></html>
"""


def main():
    M = json.load(open(os.path.join(ROOT, "params", "direction_model.json"), encoding="utf-8"))
    feats = sorted({f for a in M["axes"].values() for f in a["features"]})
    groups = [(g, [f for f in fs if f in feats]) for g, fs in GROUPS]
    listed = {f for _, fs in groups for f in fs}
    rest = [f for f in feats if f not in listed]
    if rest:
        groups.append(("其他", rest))
    # 示範數值：保留集中一位 B/C 肝陽性者（與 direction.py demo 同一位，可對照）
    demo = json.load(open(os.path.join(ROOT, "params", "direction_demo_patient.json"),
                          encoding="utf-8")) if os.path.exists(
        os.path.join(ROOT, "params", "direction_demo_patient.json")) else {}
    html = (HTML.replace("__MODEL__", json.dumps(M, ensure_ascii=False))
                .replace("__LABELS__", json.dumps(LABEL, ensure_ascii=False))
                .replace("__UNITS__", json.dumps(UNIT, ensure_ascii=False))
                .replace("__GROUPS__", json.dumps(groups, ensure_ascii=False))
                .replace("__DEMO__", json.dumps(demo, ensure_ascii=False))
                .replace("__AUC_INF__", f"{M['axes']['感染']['oof_auroc']:.3f}")
                .replace("__AUC_MET__", f"{M['axes']['代謝']['oof_auroc']:.3f}"))
    os.makedirs(os.path.join(ROOT, "ui"), exist_ok=True)
    out = os.path.join(ROOT, "ui", "direction.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"[完成] {out}  ({len(html)//1024} KB，{len(feats)} 個輸入欄位，{len(groups)} 組)")


if __name__ == "__main__":
    main()
