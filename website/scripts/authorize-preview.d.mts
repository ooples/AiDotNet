import type { BrowserContext } from '@playwright/test';
export function approvedPreviewOrigin(value: string): string;
export function authorizePreview(options: {baseURL:string; secret?:string; expectedCommit?:string; expectedRun?:string; fetcher?:typeof fetch}): Promise<Awaited<ReturnType<BrowserContext['storageState']>>>;
