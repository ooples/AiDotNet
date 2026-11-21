import { test as setup } from '@playwright/test';
import { writeFileSync } from 'node:fs';
import { authorizePreview } from '../../../scripts/authorize-preview.mjs';
import { DEPLOYMENT_STORAGE_STATE } from '../helpers/auth-state';

setup('authorize the protected preview deployment', async ({ baseURL }) => {
  const state = await authorizePreview({ baseURL: baseURL!, secret: process.env.PREVIEW_BYPASS_SECRET,
    expectedCommit: process.env.EXPECTED_PREVIEW_COMMIT, expectedRun: process.env.EXPECTED_PREVIEW_RUN });
  writeFileSync(DEPLOYMENT_STORAGE_STATE, JSON.stringify(state), { mode: 0o600 });
});
