#!/usr/bin/env node

const minimumMajor = 20;
const [major] = process.versions.node.split(".").map(Number);

if (!Number.isFinite(major) || major < minimumMajor) {
  console.error(`@papaya/ai requires Node.js ${minimumMajor} or newer. Current version: ${process.version}`);
  process.exit(1);
}
