// 驗收第 4 項：同一組輸入連按十次計算，輸出完全一致；並輸出三組測試輸入的結果供 Python 對照。
// 用法：node verify_ui.js index.html
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[2] || "index.html", "utf8");
const grab = (id) => JSON.parse(html.match(new RegExp(`<script id="${id}" type="application/json">([\\s\\S]*?)</script>`))[1]);
const PACK = grab("ui_pack"), FIELDS = grab("ui_fields");
const pure = html.split("/*__PURE_BEGIN__*/")[1].split("/*__PURE_END__*/")[0];
const ctx = {}; vm.createContext(ctx); vm.runInContext(pure + "\nthis.computeAll = computeAll;", ctx);
const cases = [
  { age: 65, male: 0, x0: 45, tau: 60, measErr: 0, measInt: 7, repType: "flip_tipped" },
  { age: 40, male: 1, x0: 80, tau: 60, measErr: 5, measInt: 30, repType: "linear" },
  { age: 82, male: 1, x0: 20, tau: 30, measErr: 10, measInt: 180, repType: "flip_stable" },
];
let ok = true; const outs = [];
for (const c of cases) {
  const first = JSON.stringify(ctx.computeAll(c, PACK, FIELDS));
  for (let i = 0; i < 10; i++) { if (JSON.stringify(ctx.computeAll(c, PACK, FIELDS)) !== first) { ok = false; console.log("不一致：", JSON.stringify(c)); } }
  const o = JSON.parse(first);
  outs.push({ input: c, score: o.score, logit: o.logit, q: o.q, pflip: o.pflip, counts: o.counts, n_used: o.used.length, repKey: o.repKey, has_rep: !!o.rep, n_obs: o.obs.length });
}
console.log(JSON.stringify({ deterministic_10x: ok, cases: outs }));
process.exit(ok ? 0 : 1);
