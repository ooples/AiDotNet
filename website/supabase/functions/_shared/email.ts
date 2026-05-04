// Shared email helper for Supabase edge functions.
//
// Sends transactional license-key emails via the Resend HTTPS API. The
// alternative (SMTP) requires a long-lived TCP socket which Deno Deploy
// edge runtimes don't reliably support; Resend's REST endpoint works
// inside any HTTPS-capable runtime and has a free tier of 3000 emails/mo
// at time of writing — adequate for the current scale of license issuance.
//
// Configuration is via Supabase project secrets (set with `supabase secrets
// set` or the dashboard):
//   RESEND_API_KEY    — required to send. If unset, email is skipped (logged).
//   EMAIL_FROM        — optional override of the From: header. Default
//                       "AiDotNet <licenses@aidotnet.dev>". Must be a verified
//                       sender domain in the Resend dashboard or sends will
//                       be rejected by Resend.
//   ACCOUNT_URL       — optional override of the link in the email body.
//                       Default "https://www.aidotnet.dev/account/licenses/".
//
// The send is BEST-EFFORT: failures are logged but never thrown. The license
// row in Postgres is the source of truth — if email fails (Resend down, key
// rotated, recipient bounces) the user can always retrieve the key by
// signing in to /account/licenses. Failing the issuance request because the
// email transport hiccupped would be a worse outcome than a missing email.

const RESEND_ENDPOINT = "https://api.resend.com/emails";
const DEFAULT_FROM = "AiDotNet <licenses@aidotnet.dev>";
const DEFAULT_ACCOUNT_URL = "https://www.aidotnet.dev/account/licenses/";
// Cap the Resend round-trip so a degraded provider doesn't pile latency
// onto webhook callers (Stripe retries the webhook if we don't 200 within
// ~10s). 5s is generous for a transactional REST call to Resend's NA
// region; failures past that are treated as send_failed and the caller
// proceeds without blocking issuance.
const RESEND_TIMEOUT_MS = 5000;
// Loose RFC-5322 sanity check matching the
// `license_keys_customer_email_format` DB constraint pattern. Catches
// obvious garbage ("@", "@@", "a@") without trying full RFC compliance —
// Resend rejects malformed addresses anyway, this is just an early-out.
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/**
 * PII-redacting log helper. Edge-function logs are centralized and may
 * be retained beyond what's necessary for transactional debugging, so
 * raw recipient addresses must not appear in log lines. Keeps the
 * domain (useful for transport debugging — bounces, MX issues) and
 * masks the local-part beyond the first two characters.
 *
 * Examples:
 *   "alice@example.com" → "al***@example.com"
 *   "ab@example.com"    → "***@example.com"
 *   "@example.com"      → "***@example.com"
 *   "garbage"           → "***"
 *   ""                  → "***"
 */
function redactEmail(email: string): string {
  if (!email) return "***";
  const at = email.indexOf("@");
  if (at < 0) return "***";
  const local = email.slice(0, at);
  const domain = email.slice(at + 1);
  const masked = local.length <= 2 ? "***" : `${local.slice(0, 2)}***`;
  return domain ? `${masked}@${domain}` : "***";
}

export interface LicenseEmailInput {
  to: string;
  licenseKey: string;
  tier: string;
  product: string;
  isExisting?: boolean;
}

export interface LicenseEmailResult {
  ok: boolean;
  reason?: "no_api_key" | "no_recipient" | "send_failed";
  status?: number;
  body?: string;
}

export async function sendLicenseKeyEmail(input: LicenseEmailInput): Promise<LicenseEmailResult> {
  const apiKey = Deno.env.get("RESEND_API_KEY");
  if (!apiKey) {
    console.warn(
      "sendLicenseKeyEmail: RESEND_API_KEY not configured — skipping email "
      + "(license was still issued in DB and is retrievable from /account/licenses).",
    );
    return { ok: false, reason: "no_api_key" };
  }

  if (!input.to || !EMAIL_RE.test(input.to)) {
    console.warn(
      `sendLicenseKeyEmail: invalid or missing recipient '${redactEmail(input.to ?? "")}' — skipping email.`,
    );
    return { ok: false, reason: "no_recipient" };
  }

  const from = Deno.env.get("EMAIL_FROM") ?? DEFAULT_FROM;
  const accountUrl = Deno.env.get("ACCOUNT_URL") ?? DEFAULT_ACCOUNT_URL;

  // Use the actual product name, not the brand name, so the subject is
  // accurate when AiDotNet ships multiple products (the wire format
  // already passes a `product` field; the original copy was treating it
  // as informational and hardcoding "AiDotNet").
  const productName = input.product?.trim() || "AiDotNet";
  const subject = input.isExisting
    ? `Your ${productName} ${input.tier} license key`
    : `Welcome to ${productName} — your ${input.tier} license key`;

  const text = renderText(input, accountUrl);
  const html = renderHtml(input, accountUrl);

  // AbortController-based timeout. AbortSignal.timeout() exists in modern
  // Deno but a manual controller is portable to older runtimes and lets us
  // distinguish a timeout from other fetch failures via the AbortError
  // name in the catch.
  const ac = new AbortController();
  const timeoutHandle = setTimeout(() => ac.abort(), RESEND_TIMEOUT_MS);

  try {
    const resp = await fetch(RESEND_ENDPOINT, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ from, to: input.to, subject, text, html }),
      signal: ac.signal,
    });

    if (!resp.ok) {
      const body = await resp.text();
      // Body kept on the returned object (caller may persist it for
      // ops triage) but NOT echoed into log output — Resend's error
      // body sometimes includes the raw `to` address in error messages,
      // and we don't want that in centralized logs.
      console.error(
        `sendLicenseKeyEmail: Resend returned ${resp.status} for ${redactEmail(input.to)}.`,
      );
      return { ok: false, reason: "send_failed", status: resp.status, body };
    }

    console.log(
      `sendLicenseKeyEmail: dispatched ${input.tier} key to ${redactEmail(input.to)} (status ${resp.status}).`,
    );
    return { ok: true, status: resp.status };
  } catch (err) {
    const isTimeout = (err as { name?: string })?.name === "AbortError";
    console.error(
      `sendLicenseKeyEmail: ${isTimeout ? "timed out after " + RESEND_TIMEOUT_MS + "ms" : "fetch failed"} for ${redactEmail(input.to)}.`,
    );
    return { ok: false, reason: "send_failed" };
  } finally {
    clearTimeout(timeoutHandle);
  }
}

function renderText(i: LicenseEmailInput, accountUrl: string): string {
  const verb = i.isExisting ? "Here is your existing" : "Here is your new";
  return [
    `Hi,`,
    ``,
    `${verb} AiDotNet ${i.tier} license key:`,
    ``,
    `    AIDOTNET_LICENSE_KEY=${i.licenseKey}`,
    ``,
    `Quick start:`,
    `  • Set the environment variable above on your dev machine, or`,
    `  • Save the key to ~/.aidotnet/license.key (one line, no quotes).`,
    ``,
    `You can always view, copy, or revoke this key at:`,
    `  ${accountUrl}`,
    ``,
    `Product: ${i.product}`,
    `Tier: ${i.tier}`,
    ``,
    `If you didn't request this license, please reply to this email so we can revoke it.`,
    ``,
    `— The AiDotNet team`,
  ].join("\n");
}

function renderHtml(i: LicenseEmailInput, accountUrl: string): string {
  const verb = i.isExisting ? "Here is your existing" : "Here is your new";
  // Inline styles only — most email clients strip <style> blocks.
  return `<!DOCTYPE html>
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #0f172a; max-width: 560px; margin: 0 auto; padding: 24px;">
  <h2 style="margin-top: 0;">Your AiDotNet ${escapeHtml(i.tier)} license key</h2>
  <p>${verb} AiDotNet <strong>${escapeHtml(i.tier)}</strong> license key for the <strong>${escapeHtml(i.product)}</strong> product:</p>
  <pre style="background: #0f172a; color: #f1f5f9; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px;"><code>AIDOTNET_LICENSE_KEY=${escapeHtml(i.licenseKey)}</code></pre>
  <p><strong>Quick start:</strong></p>
  <ul>
    <li>Set the environment variable above on your dev machine, or</li>
    <li>Save the key to <code>~/.aidotnet/license.key</code> (one line, no quotes).</li>
  </ul>
  <p>You can always view, copy, or revoke this key at:<br>
    <a href="${escapeHtml(accountUrl)}" style="color: #2563eb;">${escapeHtml(accountUrl)}</a>
  </p>
  <p style="color: #64748b; font-size: 13px; margin-top: 32px;">If you didn't request this license, please reply to this email so we can revoke it.</p>
  <p style="color: #64748b; font-size: 13px;">— The AiDotNet team</p>
</body>
</html>`;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
