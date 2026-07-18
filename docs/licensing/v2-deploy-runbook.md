# v2 licensing — server deploy runbook

This is the ordered, safe rollout for the offline `aidn2` licensing server side. The client SDK changes
(AiDotNet #1891, AiDotNet.Tensors #808) and the additive DB objects are already done; the steps below are the
remaining pieces that require the **Ed25519 signing key**, which is intentionally not automatable from the
agent (the private key must be generated on a trusted machine and set as a secret — it never enters a chat or
the repo).

## What is already deployed / merged-ready

- **DB (applied to `yfkqwpgjahoamlgckjib`):** `public.revocations` table + `public.revoke_license(uuid,text)`
  (migrations `20260718000000_revocations.sql`, `20260718000100_lock_revoke_license_execute.sql`). Additive
  and locked down (service_role-only; verified via the security advisor). No existing object changed, so the
  live paid customers are unaffected. `validate_license_key` already returns `license_id` (the natural `jti`)
  and per-tier `capabilities` — **no change needed** to that hot-path function.
- **Edge function code (committed, NOT yet deployed):** `issue-license`, `get-revocations`, and the shared
  signer `_shared/aidn2.ts`. They 503 until the signing secret exists — deploy them in step 3.
- **Client SDK:** capability-authoritative persistence gate, aidn2 scope/machine/CRL verification, embedded +
  online CRL provider. The online→offline *auto-fetch glue* (SDK calling `issue-license` / `get-revocations`)
  is deliberately deferred to step 5 so it's wired against live, tested endpoints.

## Step 1 — generate the signing keypair (on your machine)

```
pip install cryptography
python tools/license-issuer/keygen.py --kid prod-2026a
```

This prints (a) the `supabase secrets set …` line with the base64 PKCS#8 private key, (b) a
`LicensePublicKey.json` JWK + the `gh variable set` commands for the public key. The private key is printed to
**your** stdout only — do not paste it anywhere shared.

Run it a **second** time with `--kid ci-2026a` for the CI-only keypair (separate private key, stored as a CI
secret, used by a short-exp `scope:ci` token — see step 6).

## Step 2 — set the server secret

Using the `supabase secrets set` line keygen printed (against project `yfkqwpgjahoamlgckjib`):

```
supabase secrets set \
  AIDOTNET_LICENSE_SIGNING_KEY_PKCS8=<base64 pkcs8 from keygen> \
  AIDOTNET_LICENSE_KID=prod-2026a
```

(Optional overrides: `OFFLINE_TOKEN_EXP_DAYS` for issue-license, `CRL_EXP_DAYS` for get-revocations.)

## Step 3 — deploy the edge functions

```
supabase functions deploy issue-license   --project-ref yfkqwpgjahoamlgckjib
supabase functions deploy get-revocations --project-ref yfkqwpgjahoamlgckjib
```

Smoke test:
```
# CRL should return a signed { kid, payload, sig } envelope (empty rjti/rkids initially):
curl https://yfkqwpgjahoamlgckjib.supabase.co/functions/v1/get-revocations
# issue-license needs a real active license_key + machine hash; expect { valid:true, offline_token:"aidn2..." }
```

## Step 4 — embed the public key in the SDKs

Copy the `LicensePublicKey.json` from step 1 into `src/BuildKey/LicensePublicKey.json` in **both** repos
(AiDotNet and AiDotNet.Tensors) and set the CI variable so release/CI builds inject it (the `gh variable set`
lines keygen printed). Cut SDK releases embedding it. Until a released SDK embeds this exact public key, no
client can verify an aidn2 token — so this must land before step 5/6 go wide.

## Step 5 — wire the client auto-fetch glue (follow-up PR)

With the endpoints live and the public key embedded, add to the SDK: after a successful online validation,
call `issue-license` and cache the returned `offline_token`; on startup fetch `get-revocations`, verify it
against the embedded key, and hand it to `LicenseRevocationProvider.TryInstallFetched`. Keep both strictly
**fail-open** and off the latency-critical path (background refresh + cache). This is deferred out of #1891 on
purpose: it's hot-path network code that should be built and tested against the now-live endpoints, not
shipped blind.

## Step 6 — CI key (replaces the ModuleInitializer stopgap, optional)

CI is already green via the `ModuleInitializer` synthetic offline license (no external dependency — the right
default for a test suite). If you'd rather CI exercise a real aidn2 key:
```
python tools/license-issuer/aidn2_issuer.py --tier enterprise --kid ci-2026a --days 90 --sub aidotnet-ci
# then: gh secret set AIDOTNET_LICENSE_KEY (the token) ; and set AIDOTNET_LICENSE_SCOPE=ci in the test job
```
Use the **ci-2026a** keypair from step 1, not the prod key, and give the token `scope:"ci"` so it only works
in CI.

## Step 7 — migrate issuance + retire AIDN-* (rollout)

Once clients in the wild embed the public key: update `register-community-license` and `stripe-webhook` to
ALSO mint an aidn2 token (dual-issue) via `_shared/aidn2.ts`, re-issue existing paid keys with caps, then
deprecate the online-only `AIDN-*` path after the 90-day migration window in the design doc.

## Rollback

Every step is independently reversible: unset the secret (functions revert to 503, clients fall back to
online AIDN-* validation), `supabase functions delete issue-license get-revocations`, or
`drop function public.revoke_license(uuid,text); drop table public.revocations;`. Nothing here alters the
existing online validation path, so paid customers keep validating throughout.
