# Deployment checklist — switch live payment flow from test → production

This branch (`fix/website-critical-issues`) ships the code-level fixes for
the payment/license flow, but **two non-code changes must be made in the
Vercel + Supabase dashboards** before customers can pay successfully. Until
both are done, the symptoms the user reported (payment freezes, "critical
error", no license issued, just-created key returns "License key not
found") will continue to occur.

## 1. Vercel environment variables — switch Stripe Payment Links from TEST to LIVE mode

The production bundle at `https://www.aidotnet.dev/_astro/pricing.astro_...js`
currently hard-codes:

```
pro.month:        https://buy.stripe.com/test_7sY6oG3XwbaccTl6tY7bW00   ← TEST mode
pro.year:         https://buy.stripe.com/test_7sYdR8alU3HKbPh2dI7bW02   ← TEST mode
enterprise.month: ""                                                     ← missing
enterprise.year:  ""                                                     ← missing
```

These come from build-time `import.meta.env` reads of:

| Env var                                  | What it should be                          |
|------------------------------------------|--------------------------------------------|
| `PUBLIC_STRIPE_PRO_MONTHLY_LINK`         | `https://buy.stripe.com/<LIVE-mode link>`  |
| `PUBLIC_STRIPE_PRO_YEARLY_LINK`          | `https://buy.stripe.com/<LIVE-mode link>`  |
| `PUBLIC_STRIPE_ENTERPRISE_MONTHLY_LINK`  | `https://buy.stripe.com/<LIVE-mode link>`  |
| `PUBLIC_STRIPE_ENTERPRISE_YEARLY_LINK`   | `https://buy.stripe.com/<LIVE-mode link>`  |

**Action:** in Vercel → Project Settings → Environment Variables, set all
four to the **LIVE-mode** Payment Link URLs from
[Stripe Dashboard → Payment Links](https://dashboard.stripe.com/payment-links).
Make sure the dashboard is in LIVE mode (toggle top-right) when you copy
the URL. LIVE-mode links do NOT start with `buy.stripe.com/test_`.

When you save each one, choose **Production**, **Preview**, and **Development**
scopes (or at minimum **Production**). The values are inlined into the
client JS at build time, so a new deploy is required after the change.

Then trigger a redeploy: Vercel → Deployments → latest → "..." → "Redeploy".

**Verify:** after the redeploy completes, fetch the same bundle and
confirm the `buy.stripe.com/` URLs no longer have the `test_` prefix:

```bash
curl -s "https://www.aidotnet.dev/_astro/pricing.astro_*.js" \
  | grep -oE 'buy\.stripe\.com/[a-zA-Z0-9_-]+' | sort -u
```

## 2. Supabase Edge Function secrets — match Stripe LIVE mode

The webhook at `supabase/functions/stripe-webhook` reads two secrets at
runtime:

| Secret                       | What it should be                                     |
|------------------------------|-------------------------------------------------------|
| `STRIPE_SECRET_KEY`          | LIVE-mode secret key from Stripe Dashboard (`sk_live_…`) |
| `STRIPE_WEBHOOK_SECRET`      | LIVE-mode webhook signing secret (`whsec_…`)          |

**Action:**
1. In [Stripe Dashboard (LIVE mode)](https://dashboard.stripe.com/) →
   Developers → API keys, copy the live secret key.
2. Developers → Webhooks → Add endpoint pointing at
   `https://<project>.supabase.co/functions/v1/stripe-webhook`
   (or update the existing endpoint to LIVE if you have one). Subscribe
   to at least:
   - `checkout.session.completed`
   - `invoice.paid`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
3. Copy the signing secret for that endpoint.
4. In Supabase Dashboard → Edge Functions → Secrets, set both
   `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET` to the LIVE values.

## 3. Stripe Payment Link metadata — set `tier`

Each LIVE Payment Link needs `metadata.tier` set so the webhook knows
which license tier to issue. The webhook now accepts both `pro` and
`professional` as synonyms (see `stripe-webhook/index.ts:TIER_CANONICAL`)
so either works, but the recommended canonical values are:

| Plan         | `metadata.tier` value |
|--------------|-----------------------|
| Pro Monthly  | `pro`                 |
| Pro Yearly   | `pro`                 |
| Enterprise   | `enterprise`          |

**Action:** in Stripe Dashboard → Payment Links → for each LIVE link →
Advanced → Metadata, add a key/value pair `tier=pro` (or `tier=enterprise`).

## 4. Stripe Payment Link success URL

Each Payment Link's "After payment → Show confirmation page → Custom URL"
should redirect customers to:

```
https://www.aidotnet.dev/account/licenses/?new=true
```

The `?new=true` query param is what triggers the post-checkout polling
spinner on the licenses page (see `src/pages/account/licenses/index.astro`).

## 5. Apply the new Supabase migration

The new migration `supabase/migrations/20260610000000_log_validations_to_api_usage.sql`
plumbs every `validate_license_key` RPC call into `api_usage` so the
`/account/usage/` dashboard stops showing "No usage data yet" for active
license holders.

**Action:**
```bash
cd website
supabase db push
```

(Or apply via the Supabase Dashboard → SQL Editor if you don't have CLI
access configured locally.)

## 6. Smoke-test the live flow

1. Browse to https://www.aidotnet.dev/pricing/ in an Incognito window.
2. Confirm Subscribe buttons resolve to `buy.stripe.com/` URLs **without**
   the `test_` prefix.
3. Sign in as a test user, click Subscribe on Pro Monthly, complete a
   real card payment.
4. Verify Stripe Dashboard → Payments shows the charge under the LIVE
   account (not test mode).
5. Within ~5s of Stripe's redirect to `/account/licenses/?new=true` the
   green "License issued!" banner should appear, and a row should be
   visible with format `AIDN-PROD-PRO-{32hex}`.
6. Wait one minute, navigate to `/account/usage/`, confirm a row appeared
   under "Calls by Endpoint" → `validate_license_key`.
7. Configure the freshly-issued key in the SDK (`AIDOTNET_LICENSE_KEY`
   env var) and run any `IModel.Save()` call to trigger a validation.
   Refresh `/account/licenses/` and confirm "Last seen" timestamps on
   the activation row updated.
