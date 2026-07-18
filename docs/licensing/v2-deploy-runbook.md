# v2 licensing — server deploy runbook

This is the ordered, safe rollout for the offline `aidn2` licensing server side. The client SDK changes
(AiDotNet #1891, AiDotNet.Tensors #808) and the additive DB objects are already done; the steps below are the
remaining pieces that require the **Ed25519 signing key**, which is intentionally not automatable from the
agent (the private key must be generated on a trusted machine and set as a secret — it never enters a chat or
the repo).

## Status — steps 1–4 DONE (prod-2026a key live, 2026-07-18)

- **DB (applied to `yfkqwpgjahoamlgckjib`):** `public.revocations` table + `public.revoke_license(uuid,text)`
  (migrations `20260718000000_revocations.sql`, `20260718000100_lock_revoke_license_execute.sql`). Additive
  and locked down (service_role-only; verified via the security advisor). No existing object changed, so the
  live paid customers are unaffected. `validate_license_key` already returns `license_id` (the natural `jti`)
  and per-tier `capabilities` — **no change needed** to that hot-path function.
- **DONE — signing key:** keypair `kid=prod-2026a` generated; private key set as the
  `AIDOTNET_LICENSE_SIGNING_KEY_PKCS8` + `AIDOTNET_LICENSE_KID` function secrets.
- **DONE — edge functions deployed + ACTIVE:** `issue-license`, `get-revocations` (both `verify_jwt:false`,
  matching the sibling license functions). Smoke-tested: `get-revocations` returns a signed CRL and its
  signature **verifies against the embedded public key** — the private key in the secret and the embedded
  public key are a proven matched pair.
- **DONE — public key embedded:** `src/BuildKey/LicensePublicKey.json` (kid prod-2026a) committed to
  ooples/AiDotNet (#1891) + set as the `AIDOTNET_LICENSE_PUBLIC_KEY_JSON` CI variable. Because it's committed,
  every build (not just CI) embeds it, so a released SDK verifies aidn2 tokens offline with no extra wiring.
- **Client SDK:** capability-authoritative persistence gate, aidn2 scope/machine/CRL verification, embedded +
  online CRL provider. The online→offline *auto-fetch glue* (SDK auto-calling `issue-license` / auto-fetching
  the CRL) is step 5 below — now unblocked and testable against the live endpoints.
- **NOTE — AiDotNet.Tensors is a separate keypair.** Tensors' offline path is RSA `SignedEntitlement`, NOT
  aidn2/Ed25519, so it does **not** consume the prod-2026a JWK. Its RSA signing key is still the placeholder
  and needs its own keygen + issuer rollout (tracked separately; #808 added its scope/CRL verification only).

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

## Step 5 — client auto-fetch glue — DONE (#1891, `OnlineLicenseServices`)

Implemented + tested against the live endpoints: after a successful online validation a throttled BACKGROUND
task (off the hot path) fetches + installs the signed CRL and mints + caches a short-lived, machine-bound
`aidn2` token via `issue-license`; when the server is later unreachable, `Validate()`/`ValidateAsync()` fall
back to that cached token (verified locally — signature + exp + machine-lock + CRL — so it only grants when
genuinely valid). `LicenseRevocationProvider` lazily loads the last-fetched CRL from disk so revocation is
enforced even on a fully-offline start. Strictly fail-open/best-effort. 8 `OnlineLicenseServicesTests` green
(18/18 with the aidn2 binding tests); project builds clean.

## Step 6 — CI key — DONE (dedicated enterprise CI license key)

CI uses ONE dedicated **enterprise** license key stored as the global `AIDOTNET_LICENSE_KEY` secret:
- Real server key `AIDN-PROD-ENTERPRISE-…` under a persistent "AiDotNet CI" user, **capability = save-capable**
  (`tensors:save`/`model:save`, so it lifts the persistence op cap under capability gating) with effectively
  **unlimited activations** (`max_activations` 2e9) so ephemeral CI runners never trip the seat limit.
- `sonarcloud.yml` + `heavy-timeout-nightly.yml` test jobs read `secrets.AIDOTNET_LICENSE_KEY`. On the main
  suite, `ModuleInitializer` swaps this online `AIDN-*` key for a deterministic synthetic OFFLINE license, so
  the persistence-heavy suite is licensed with **no network round-trip**; jobs/consumers without that
  hardening validate the key online. One key, all branches — simplest + save-capable.
- **Self-expiring:** the key has `expires_at` = **1 year out**, so a leaked key stops working on its own
  (the containment property the retired scope-fenced ci token had) while staying save-capable and simple.

**Rotation (annual, ~5 min, plain DB op — NO signing key or `secrets:write` PAT):**
1. Issue a fresh enterprise CI key (reuses the persistent CI user `…00000000c1c1`):
   ```sql
   insert into public.license_keys (user_id, license_key, tier, status, max_activations, product, organization_name, expires_at, notes)
   values ('00000000-0000-0000-0000-00000000c1c1',
           'AIDN-PROD-ENTERPRISE-' || replace(gen_random_uuid()::text,'-',''),
           'enterprise','active', 2000000000, 'aidotnet', 'AiDotNet CI',
           now() + interval '1 year', 'Dedicated CI/CD key (rotated).')
   returning id, license_key;
   ```
2. `gh secret set AIDOTNET_LICENSE_KEY --repo ooples/AiDotNet --body '<new key>'` (and update any external
   pipelines using the old key).
3. Revoke the old key once CI is green on the new one: `select public.revoke_license('<old license_id>', 'rotated');`
   (Immediate: it's an online key, so `status=revoked` takes effect on the next validation — no CRL wait.)

Current CI key id (revoke target): `e5c265c7-23f0-4e66-a1f6-1d9c93da4cd4`.

_(An earlier design used a scope-fenced `ci-2026a` offline aidn2 token + a rotation workflow that needed a
`secrets:write` PAT; that was retired in favour of this single, self-expiring enterprise key. The prod-2026a
Ed25519 signing key remains — it's what `issue-license` uses to mint real users' offline tokens.)_

**Release embed — automated:** `release-please.yml` injects `LicensePublicKey.json` (+ `LicenseRevocation.json`)
from the repo var before the release build, so a key rotation ships in the released NuGet without a code
change (a set var overrides the committed file). The repo also commits the file, so a build embeds it even
with the var unset.

**Online fallback key (design §9) — superseded:** the ModuleInitializer hardening (validate the env key
offline; fall back to the synthetic license when it doesn't verify Active) already covers "the offline embed
is missing" for CI, and real users keep their durable `AIDN-*` online key — so a dedicated high-activation
fallback key isn't provisioned.

## Step 7 — rollout status + retire AIDN-*

- **Community bootstrap issuance — DONE:** `register-community-license` now also mints a short-exp,
  load-only (`tensors:load`) `aidn2` bootstrap token so a new community user can load offline immediately.
  It's non-machine-bound but safe *because* it's load-only; **paid tiers get no non-bound token** — their
  SDK derives a machine-bound one via `issue-license`, so `stripe-webhook` needs no aidn2 change.
  (Redeploy the function — edge functions are NOT auto-deployed by `deploy-website.yml`, which only runs
  `supabase db push`.)
- **Re-issue paid keys with caps — no-op:** caps are tier-derived in `validate_license_key`, so all existing
  paid keys already receive them on validation (verified: 9 pro + 3 enterprise + 42 community).
- **Publish first CRL — DONE:** `get-revocations` serves the signed (initially empty) CRL.
- **Deprecate `AIDN-*`:** keep the online `AIDN-*` path through the 90-day migration window, then retire it.

## Rollback

Every step is independently reversible: unset the secret (functions revert to 503, clients fall back to
online AIDN-* validation), `supabase functions delete issue-license get-revocations`, or
`drop function public.revoke_license(uuid,text); drop table public.revocations;`. Nothing here alters the
existing online validation path, so paid customers keep validating throughout.
