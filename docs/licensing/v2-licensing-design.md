# AiDotNet v2 Licensing — Design (for review)

Status: **DRAFT for review** · Scope: AiDotNet SDK + AiDotNet.Tensors + aidotnet.dev (Supabase) + CI
Owner: licensing · Related: PR #1802 (attestation), #1807/#1808/#1832 (aidn2), #1883 (CI secret)

> Decision inputs already agreed:
> - Revocation model: **short expiry + periodic online refresh + CRL deny-list**.
> - Paid-tier gating: **model/tensor save-load persistence**, **encrypted-model IP protection**, **air-gapped + higher seats**. GPU packs stay **free**.
> - Private signing key: **server-side only** (never ships).
> - CI key: **short-lived scoped offline `aidn2` primary + revocable online key fallback**.
> - Sequence: **design first (this doc), then implement.**

---

## 1. Problem statement

1. **CI is red (urgent).** After #1883 started passing `AIDOTNET_LICENSE_KEY` into the test jobs, license-gated persistence tests (`LicensingIntegrationTests.*`, `ExpressionTree_Serialize*`, `InferenceSession…Serialize*`) fail. Root cause: the secret is a **community `AIDN-PROD-COMMUNITY-*` server key** that (a) only validates **online** and (b) does **not** reliably lift the free-trial operation cap in CI, so the shared trial budget drains to 0 and every `SaveEncrypted`/`Load` that expects success throws `LicenseRequiredException`. Before #1883, `ModuleInitializer` injected a synthetic offline key and CI was green.
2. **v1 is reverse-engineerable.** v1 `aidn.` keys use a **symmetric HMAC build key** that ships in every DLL (`BuildKey.bin`). Anyone who extracts it can mint unlimited valid keys. This is why v2 exists.
3. **v2 is code-complete but not operational.** `aidn2` (Ed25519/EdDSA) verification exists in both `AiDotNet` and `AiDotNet.Tensors`, but: the release pipeline injects **`BuildKey.bin` only, not `LicensePublicKey.json`**, there is **no issuer** (the website still mints `AIDN-*` server keys), and there is **no revocation** story for offline tokens.

## 2. Current state (as-is)

| Concern | AiDotNet SDK | AiDotNet.Tensors | Website / Supabase |
|---|---|---|---|
| v1 `aidn.` HMAC (symmetric) | `LicenseValidator.ValidateOffline` + `BuildKeyProvider` (embedded `AiDotNet.BuildKey`) | mirror | not issued by site |
| v2 `aidn2.` Ed25519 (asymmetric) | `AsymmetricLicenseVerifier` + `LicensePublicKeyProvider` (embedded `AiDotNet.LicensePublicKey`, **not injected at release**) | mirror (`LicenseValidator`, `SignedEntitlement`, `LicenseResponseVerifier`) | **no issuer** |
| Online server key `AIDN-*` | `LicenseValidator.ValidateOnline` → Supabase `validate-license` → `validate_license_key` RPC (exact-match) | mirror | `register-community-license` (community, `max_activations:3`), `stripe-webhook` (paid) |
| Offline grace | `OfflineGracePeriod` (default **7d**), `OnlineValidationAttestation` (**30d**), require prior online validation before honoring "pending" (#1802) | `TrialState`, `PersistenceGuard` | — |
| Trial | community free trial: 30 days / N operations (`TrialStateManager`, `ModelPersistenceGuard`) | `TrialState` | — |
| Capabilities | plumbed (`LicenseValidator` parses `capabilities[]`, `TensorLicenseFlow`) but **paid gating is partial** | `TensorLicenseFlow` capability enforcement | RPC returns per-tier `capabilities[]` |
| Revocation | none for offline; online = status flip in DB | none for offline | DB `status` column |

Key takeaway: **the hybrid machinery already exists** (asymmetric verify, exp, grace, attestation, signed responses, capabilities). What's missing is (a) making v2 issuance+embed operational, (b) a revocation policy, (c) real paid-tier gating, (d) a CI-appropriate key.

## 3. Goals & non-goals

**Goals**
- Keys are **unforgeable** (asymmetric; private key never ships) — done in principle by aidn2; make it operational.
- **Offline-capable but revocable** within one token lifetime (hybrid: short exp + refresh + CRL).
- **Paid conversion** driven by capability tiering, not friction.
- **CI green** with a locked-down, low-blast-radius key.

**Non-goals**
- Preventing a determined attacker from **patching the client binary** to skip verification. No client scheme can; v2 stops *key forgery*, and the BSL license governs the rest. State this explicitly to avoid over-promising "completely locked down."

## 4. Token design (aidn2 v2)

Grammar (unchanged): `aidn2.<base64url(claims_json)>.<base64url(ed25519_sig)>`; signature over the raw claim bytes; verified against the embedded public key selected by `kid`.

**Claims** (`LicenseClaims`, extend):

| Claim | Meaning | New? |
|---|---|---|
| `sub` | license holder / customer id | existing |
| `tier` | `community` \| `pro` \| `enterprise` | existing |
| `seats` | seat/activation count (advisory offline) | existing |
| `iat` / `exp` | issued-at / expiry (unix s). **exp short** (see §5). | existing |
| `kid` | signing key id (rotation) | existing |
| `alg` | `EdDSA` | existing |
| **`jti`** | unique token id → **CRL revocation target** | **new** |
| **`caps`** | explicit capability list (authoritative offline; see §7) | **new** |
| **`mach`** | machine-binding hash (node-lock; optional) | **new** |
| **`scope`** | audience/scope, e.g. `"ci"`, `"prod"` (optional binding) | **new** |

Verifier changes (`AsymmetricLicenseVerifier`, mirrored in Tensors):
- Enforce `jti` ∉ embedded/refreshed **CRL** (§6).
- If `mach` present → require it to equal the local machine hash (`LicenseValidator.GetMachineIdHash` / `MachineFingerprint`), else `Invalid`. (Node-lock for customer offline keys.)
- If `scope` present → require it to equal the host's configured expected scope (`AIDOTNET_LICENSE_SCOPE` env / builder config), else `Invalid`. (Used to fence the CI token off from product use.)
- `caps` becomes the source of truth for offline capability gating (§7).

## 5. Hybrid online/offline flow (revocation part 1: short expiry + refresh)

Formalize the cadence the code already half-implements:

- **Token `exp`: short.** Customer tokens **7–30 days**; CI token **≤ 30 days** (rotated by a scheduled job, §9).
- **Refresh:** the SDK re-validates **online** opportunistically; a successful online validation records an **attestation** (30d) and refreshes the cached result (grace 7d). If the app runs offline, it keeps working until **min(exp, attestation+30d)**. Past that → `LicenseRequiredException` (deactivates).
- **Revoke by refusing re-issue:** to revoke, flip the DB `status`/delete the key; the next refresh fails and the server won't mint a new token. Net revocation latency ≤ one token lifetime + grace.
- Keep `OnlineValidationAttestation` gating (#1802): "pending" is only honored with a prior successful online validation — prevents "block the server → free forever."

Tunables to expose: `exp` per tier, `AttestationValidity` (30d), `OfflineGracePeriod` (7d), refresh interval.

## 6. CRL deny-list (revocation part 2: catch leaks before expiry)

- Server publishes a **signed** revocation list: `{ "revoked_kids": [...], "revoked_jti": [...], "iat", "exp", "sig" }`, signed with the same Ed25519 private key (or a dedicated CRL key), verifiable with the embedded public key.
- Distribution: (a) **embedded/updated with releases** (`AiDotNet.LicenseRevocation` resource, refreshed each release) for pure-offline builds, and (b) **fetched on online refresh** and cached, for connected clients.
- Verifier rejects a token whose `kid` or `jti` is on a valid (unexpired, signature-verified) CRL.
- Bounds a leaked token to `min(exp, next release / next online refresh)`.

## 7. Paid-tier capability gating (conversion driver)

Authoritative capability set = `caps` claim (offline) ∪ server `capabilities[]` (online), enforced by **both** `ModelPersistenceGuard` (AiDotNet) and `PersistenceGuard`/`TensorLicenseFlow` (Tensors).

| Capability | community (free) | pro | enterprise |
|---|---|---|---|
| `tensors:load`, model **load** | ✅ | ✅ | ✅ |
| **persistence**: `tensors:save`, `model:save`/`load` (unlimited) | ❌ (trial op cap) | ✅ | ✅ |
| **encrypted model files / IP protection** (`model:encrypt`) | ❌ | ✅ | ✅ |
| **air-gapped/offline operation + higher seats** (`offline`, seats) | ❌ (online-only, low grace) | limited | ✅ |
| GPU accel packs (CUDA/OpenCL/OneDNN) | ✅ **free** | ✅ | ✅ |

Enforcement changes: replace the current "any `Active` lifts the trial cap" with **capability-checked** gating — `Active` + `model:save`/`tensors:save` lifts persistence; `model:encrypt` gates `SaveEncrypted`; `offline` gates air-gapped grace. (This is the real product change; today an Active community key would wrongly unlock everything.)

## 8. Issuance & key custody

- **Private key: server-side only.** Generate an Ed25519 keypair; store the private half in a Supabase secret (or KMS). Public half → `LicensePublicKey.json` (JWK OKP), embedded at build.
- **Issuer = Supabase edge function** `issue-license` (new): authenticates the user, looks up their tier/entitlement, signs a **short-exp** `aidn2` token (with `jti`, `caps`, optional `mach`/`scope`), stores `jti`→license mapping for CRL, returns the token. `register-community-license` and `stripe-webhook` call it instead of emitting `AIDN-*` strings.
- **kid rotation:** embed multiple public keys (`keys[]`); rotate by issuing under a new `kid` while old tokens still verify.
- **Reference issuer** (this repo, `tools/license-issuer/aidn2_issuer.py`) is dev/CI-bootstrap only; production signing is the edge function.
- **Migration:** existing `AIDN-*` community keys keep working online during a deprecation window; the account page re-issues an `aidn2` token on next visit.

## 9. CI key (unblock)

- **Primary: short-lived scoped offline `aidn2`.** `tier` = lowest that lifts persistence (**pro** with `model:save`/`tensors:save`/`model:encrypt`, or a dedicated `caps`), `scope:"ci"`, `exp` ≤ 30d, its own `kid` (`ci-<year><rev>`), signed by a **CI-only keypair** distinct from the customer key (so CI compromise ≠ customer forgery). Set as `AIDOTNET_LICENSE_KEY`.
- **Public-key embed in CI build:** add a step to `sonarcloud.yml` (and keep in `release-please.yml`) that writes `src/BuildKey/LicensePublicKey.json` from a repo secret **before** build, so the CI-built `AiDotNet.dll` (and restored `AiDotNet.Tensors`) can verify the token offline. Do the same in the Tensors repo build.
- **Fallback: revocable online key.** Provision one high-/unlimited-activation server key so validation still returns `Active` if the offline embed is ever missing.
- **Rotation job:** a scheduled workflow re-signs the CI token (calls the edge function or the reference issuer with the CI keypair) and updates the `AIDOTNET_LICENSE_KEY` secret before expiry.
- Also make `ModuleInitializer` inject an offline test key **even when the env key is set-but-unverifiable**, so a key hiccup degrades to trial-lifted test mode instead of red.

## 10. Component change list

- **AiDotNet SDK**: `LicenseClaims` (+`jti`/`caps`/`mach`/`scope`); `AsymmetricLicenseVerifier` (CRL + mach + scope + caps); new `LicenseRevocationProvider` (embedded + fetched CRL); `ModelPersistenceGuard` (capability-checked gating); `LicensePublicKeyProvider` (multi-kid already ok); `ModuleInitializer` hardening.
- **AiDotNet.Tensors**: mirror verifier/CRL/caps in `LicenseValidator` + `PersistenceGuard` + `TensorLicenseFlow`; embed public key + CRL at release.
- **Website/Supabase**: new `issue-license` edge fn (server signs aidn2); `register-community-license`/`stripe-webhook` call it; `validate_license_key` returns `jti`/`caps`; add `revocations` table + signed CRL endpoint; keep `AIDN-*` online path for migration.
- **CI**: inject `LicensePublicKey.json`; set short-lived scoped `aidn2` secret; rotation workflow; online fallback key.
- **Tooling**: keep `tools/license-issuer` as reference/bootstrap.

## 11. Threat model / explicit limits

- **Forgery:** prevented (asymmetric; private key server-only). ✅
- **Leak of an offline token:** bounded by `min(exp, CRL update, next online refresh)`; `mach`/`scope` binding further narrows use. ✅ (bounded, not zero)
- **Patched binary / stripped targets:** not preventable client-side; governed by BSL + `AiDotNet.Tensors.Enterprise.targets` gate for the disable flags. ⚠️ (accepted)
- **CI secret leak:** blast radius limited to a CI-only keypair, `scope:"ci"`, short exp, revocable via CRL. ✅

## 12. Decisions (RESOLVED — approved for implementation)

1. **CI token caps** → **dedicated minimal caps set**: `caps:["model:save","tensors:save","model:encrypt"]`, `scope:"ci"`, short `exp`. Lowest blast radius. (Depends on caps-enforcement landing — see #3.)
2. **CRL distribution** → **both**: signed CRL embedded at release (`AiDotNet.LicenseRevocation`) **and** fetched/cached on each online refresh.
3. **`caps` authority** → **switch now**: `caps` is authoritative immediately; persistence/encryption gating is capability-checked (retire "any `Active` unlocks all"). **Action:** re-issue existing paid keys with `caps` during rollout so they aren't disrupted.
4. **Keypair topology** → **two keypairs**: a **customer** keypair (private key server-only) and a separate **CI-only** keypair (private key = CI secret). Embed both public keys (two `kid`s). A CI compromise cannot forge customer licenses.
5. **Migration window** for `AIDN-*` community keys: default **90 days** — keep the online `AIDN-*` path working and auto-re-issue an `aidn2` token on next account visit.

## 13. Implementation phases

- **P1 (client SDK, unblocked now):** `LicenseClaims` (+`jti`/`caps`/`mach`/`scope`); `AsymmetricLicenseVerifier` (CRL + `mach` + `scope`); `LicenseRevocationProvider` (embedded + fetched CRL); `ModelPersistenceGuard` capability-checked gating; `ModuleInitializer` hardening; CI `LicensePublicKey.json` inject step. Mirror in **AiDotNet.Tensors**.
- **P2 (server, needs Supabase MCP):** `issue-license` edge fn (server signs `aidn2`), `revocations` table + signed CRL endpoint, `validate_license_key` returns `jti`/`caps`, wire `register-community-license`/`stripe-webhook` to the issuer.
- **P3 (CI):** generate CI-only keypair; set short-exp scoped `aidn2` secret + public-key embed secret; online fallback key; rotation workflow.
- **P4 (rollout):** re-issue paid keys with `caps`; publish first CRL; deprecate `AIDN-*` community after the 90-day window.
