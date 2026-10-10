// Fails when the website makes a performance claim that no measurement backs.
//
//   node website/scripts/check-performance-claims.mjs [--tensors-readme <path>]
//
// The site once advertised "up to 500x faster than PyTorch" and four speedup cards (500x GEMM, 200x Conv2D, 100x GEMV,
// 40x MaxPool2D) with no benchmark behind any of them; AiDotNet.Tensors' own notes record 3-6x losses to cuDNN on dense
// convolution. This check makes every published speed number traceable:
//
// 1. src/data/tensors-cpu-benchmarks.json, the only performance numbers the site shows, must equal the summary table of
//    the AiDotNet.Tensors README (wins and losses per library, the cited Results/<run>/ folder, the versions named).
//    The README is fetched from GitHub main, so a new Tensors run fails this check until the site is updated.
// 2. Everywhere else under src/ (the generated API reference excepted), a multiplier ("2x", "500x"), "faster than",
//    "slower than", "fastest", "outperform" or "speedup" fails unless performance-claims-allowlist.json lists that
//    line with the reason it is not a performance claim. Allowlist entries that no longer match also fail.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const site = fileURLToPath(new URL('..', import.meta.url));
const src = join(site, 'src');
const SKIP = [join(src, 'content', 'docs', 'reference')];
const EXTENSIONS = /\.(astro|md|mdx|ts|tsx|js|mjs|json|html)$/;
const TENSORS_README = 'https://raw.githubusercontent.com/ooples/AiDotNet.Tensors/main/README.md';
const MULTIPLIER = /(?<![\w.×])\d+(?:[.,]\d+)?\+?\s?[x×](?![\w\d])/;
const CLAIM = /\b(?:faster|slower) than\b|\bfastest\b|\boutperform\w*|\bspeed-?ups?\b/i;

const errors = [];
const fail = (msg) => errors.push(msg);

function* files(dir) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (SKIP.includes(path)) continue;
    if (statSync(path).isDirectory()) yield* files(path);
    else if (EXTENSIONS.test(name)) yield path;
  }
}

async function tensorsReadme() {
  const i = process.argv.indexOf('--tensors-readme');
  if (i > 0 && process.argv[i + 1]) return readFileSync(process.argv[i + 1], 'utf8');
  const response = await fetch(TENSORS_README);
  if (!response.ok) throw new Error(`Could not fetch ${TENSORS_README}: HTTP ${response.status}`);
  return response.text();
}

function checkBenchmarkData(readme) {
  const dataPath = join(src, 'data', 'tensors-cpu-benchmarks.json');
  const data = JSON.parse(readFileSync(dataPath, 'utf8'));
  const runs = new Set([...readme.matchAll(/tests\/AiDotNet\.Tensors\.Benchmarks\/Results\/([^/`)\s]+)\//g)].map((m) => m[1]));
  if (!runs.has(data.run)) fail(`${relative(site, dataPath)} cites run ${data.run}; the Tensors README cites ${[...runs].join(', ') || 'none'}.`);
  const summary = new Map();
  for (const m of readme.matchAll(/^\| ([^|]+?) (?:\([^|]*\) )?\| (\d+) \| (\d+) \|$/gm)) summary.set(m[1].trim(), [Number(m[2]), Number(m[3])]);
  for (const r of data.results) {
    const published = summary.get(r.library);
    if (!published) { fail(`The Tensors README summary has no row for ${r.library}.`); continue; }
    if (published[0] !== r.wins || published[1] !== r.losses)
      fail(`${r.library}: the site shows ${r.wins} wins / ${r.losses} losses; the Tensors README shows ${published[0]} / ${published[1]}.`);
    const version = r.label.match(/^(.+?) (\d+(?:\.\d+)*)/);
    if (version && !new RegExp(`${version[1].replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} ${version[2].replace(/\./g, '\\.')}(?![\\d])`).test(readme))
      fail(`${r.library}: the site names "${version[1]} ${version[2]}", which the Tensors README does not.`);
  }
  for (const library of summary.keys())
    if (!data.results.some((r) => r.library === library || library.startsWith(r.library)))
      fail(`The Tensors README reports ${library}, which the site's benchmark data leaves out.`);
}

function checkClaims() {
  const allowPath = join(site, 'performance-claims-allowlist.json');
  const allow = JSON.parse(readFileSync(allowPath, 'utf8')).entries;
  const used = new Set();
  for (const path of files(src)) {
    const rel = relative(site, path).split(sep).join('/');
    readFileSync(path, 'utf8').split(/\r?\n/).forEach((line, i) => {
      if (rel === 'src/data/tensors-cpu-benchmarks.json') return;   // verified against the Tensors README above
      if (!MULTIPLIER.test(line) && !CLAIM.test(line)) return;
      const entry = allow.findIndex((a) => a.file === rel && line.includes(a.text));
      if (entry >= 0) { used.add(entry); return; }
      fail(`${rel}:${i + 1} makes a performance claim with no benchmark behind it: ${line.trim().slice(0, 160)}`);
    });
  }
  allow.forEach((a, i) => {
    if (!used.has(i)) fail(`performance-claims-allowlist.json entry for ${a.file} ("${a.text}") matches nothing; remove it.`);
  });
}

checkBenchmarkData(await tensorsReadme());
checkClaims();
if (errors.length) {
  console.log(`${errors.length} unbacked or stale performance claim(s):`);
  for (const e of errors) console.log(` - ${e}`);
  process.exit(1);
}
console.log('Every performance claim on the site is backed by published AiDotNet.Tensors benchmarks.');
