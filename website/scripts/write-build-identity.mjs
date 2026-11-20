import { writeFileSync } from 'node:fs';
writeFileSync('dist/verification-build.json', JSON.stringify({ commit: process.env.GITHUB_SHA ?? null, run_id: process.env.GITHUB_RUN_ID ?? null })+'\n');
