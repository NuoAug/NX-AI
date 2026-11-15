#!/usr/bin/env node
// NX-AI 模型资产校验清单生成器。
// 用法（仓库根目录）：node NX-AI/Tools/make_manifest.js
// 输出：NX-AI/manifest.json（相对路径、字节数、sha256），排序后便于 diff 复核。
"use strict";

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "manifest.json");
const SCAN = ["Models", "Runtime"];

function walk(dir, acc) {
  for (const name of fs.readdirSync(dir).sort()) {
    const full = path.join(dir, name);
    const st = fs.statSync(full);
    if (st.isDirectory()) walk(full, acc);
    else if (st.isFile()) acc.push(full);
  }
  return acc;
}

function sha256(file) {
  const hash = crypto.createHash("sha256");
  const fd = fs.openSync(file, "r");
  const buf = Buffer.alloc(4 * 1024 * 1024);
  try {
    let n;
    while ((n = fs.readSync(fd, buf, 0, buf.length, null)) > 0) {
      hash.update(buf.subarray(0, n));
    }
  } finally {
    fs.closeSync(fd);
  }
  return hash.digest("hex");
}

const entries = [];
for (const sub of SCAN) {
  const base = path.join(ROOT, sub);
  if (!fs.existsSync(base)) continue;
  for (const file of walk(base, [])) {
    const rel = path.relative(ROOT, file).split(path.sep).join("/");
    entries.push({ path: rel, bytes: fs.statSync(file).size, sha256: sha256(file) });
  }
}

const manifest = {
  generator: "NX-AI/Tools/make_manifest.js",
  generated_at: new Date().toISOString(),
  note: "sha256 用于目标板部署后核对；模型资产独立保存。",
  files: entries,
};
fs.writeFileSync(OUT, JSON.stringify(manifest, null, 2) + "\n", "utf8");
console.log(`manifest: ${entries.length} files -> ${path.relative(process.cwd(), OUT)}`);
