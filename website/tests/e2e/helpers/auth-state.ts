import { mkdtempSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// Controller config sets the run directory before spawning worker processes.
const directory = process.env.AIDOTNET_AUTH_RUN_DIR ?? mkdtempSync(join(tmpdir(), 'aidotnet-e2e-auth-'));
process.env.AIDOTNET_AUTH_RUN_DIR = directory;
mkdirSync(directory, { recursive: true, mode: 0o700 });
export const USER_STORAGE_STATE = join(directory, 'user.json');
export const ADMIN_STORAGE_STATE = join(directory, 'admin.json');
export const DEPLOYMENT_STORAGE_STATE = join(directory, 'deployment.json');
