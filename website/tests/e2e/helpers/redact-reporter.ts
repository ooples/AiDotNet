import type { Reporter, TestCase, TestResult, TestStep, TestError } from '@playwright/test/reporter';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

export function redactText(text: string, values: string[]): string {
  for (const value of values) if (value.length > 3) text = text.split(value).join('[redacted]');
  return text.replace(/(authorization|apikey|cookie|x-vercel-protection-bypass)\s*:[^\r\n]*/gi, '$1: [redacted]')
    .replace(/([?&#](?:access_token|refresh_token|token_hash|token|code)=)[^&\s"'<>]+/gi,'$1[redacted]')
    .replace(/eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/g,'[redacted token]')
    .replace(/\badn_[0-9a-f]{64}\b/gi,'[redacted API key]')
    .replace(/\bAIDN-[A-Z0-9-]+-[A-F0-9]{8,}\b/gi,'[redacted license key]')
    .replace(/E2e![A-Za-z0-9_-]{12,}/g,'[redacted fixture password]');
}

export default class RedactReporter implements Reporter {
  private values() {
    const keys=['SUPABASE_SERVICE_ROLE_KEY','STRIPE_SECRET_KEY','STRIPE_WEBHOOK_SECRET','VERCEL_AUTOMATION_BYPASS_SECRET','PLAYWRIGHT_USER_PASSWORD','PLAYWRIGHT_ADMIN_PASSWORD'];
    const values=keys.map(key=>process.env[key]??'').filter(Boolean);
    if (process.env.AIDOTNET_AUTH_RUN_DIR) for (const name of ['deployment.json','user.json','admin.json']) {
      try {
        const state=JSON.parse(readFileSync(join(process.env.AIDOTNET_AUTH_RUN_DIR,name),'utf8'));
        values.push(...(state.cookies??[]).map((cookie:{value:string})=>cookie.value));
      } catch { /* state is written by its setup project */ }
    }
    return values;
  }
  private error(error?:TestError) {
    if(!error) return;
    const values=this.values();
    for(const key of ['message','stack','value','snippet'] as const) if(typeof error[key]==='string') error[key]=redactText(error[key]!,values);
    if(error.cause) this.error(error.cause);
  }
  onStepBegin(_test:TestCase,_result:TestResult,step:TestStep) { step.title=redactText(step.title,this.values()); }
  onStepEnd(_test:TestCase,_result:TestResult,step:TestStep) { step.title=redactText(step.title,this.values());this.error(step.error); }
  onTestEnd(_test:TestCase,result:TestResult) { result.errors.forEach(error=>this.error(error)); }
  onError(error:TestError) { this.error(error); }
}
