# Model Encryption & Licensing

AiDotNet is distributed under the **Business Source License 1.1** (see [`LICENSE`](../../LICENSE)).
Model **save/load** is a licensed capability: it is free for non-commercial use and
available under a free community license or a paid commercial tier (see
<https://aidotnet.dev>). Everything else in the library — training, inference,
the full model zoo, the tensor engine — works without a license.

This document explains, openly and completely, the mechanism that gates model
save/load, why it exists, and how to obtain and use a key. Nothing here is secret:
under BSL 1.1 the source is fully readable, so we document the protection rather
than rely on it being hidden.

## What is gated

| Capability | License required? |
|---|---|
| Training, fitting, prediction, evaluation | No |
| Using any built-in model / layer / optimizer | No |
| Saving a trained model to the `.aimf` format | **After a 10-operation free trial** |
| Loading an encrypted `.aimf` model | **Yes** (the key that encrypted it, or a compatible tier key) |

The first **10** save/load operations work with no key at all, so you can evaluate
the round-trip before deciding on a license. After the trial, save/load calls
require a valid license key.

## How it works (the three layers)

Model payloads are encrypted with **AES-256-GCM**. The AES key is derived from your
license key via **PBKDF2-SHA256** with a random per-payload salt
([`ModelPayloadEncryption.cs`](../../src/Helpers/ModelPayloadEncryption.cs)). GCM provides
both confidentiality and tamper detection (the 16-byte authentication tag).

Three layers protect the integrity of that scheme:

1. **Build key** ([`BuildKeyProvider.cs`](../../src/Helpers/BuildKeyProvider.cs)) — official
   CI/CD builds embed a signing key as an assembly resource. It contributes entropy to
   key derivation so that payloads produced by official builds and forks/dev builds use
   different key material. A build with no embedded key (any fork or local build) simply
   reports "not an official build" and derives its own keys — it is not blocked from
   developing against the library.
2. **License validation** ([`LicenseValidator.cs`](../../src/Helpers/LicenseValidator.cs)) —
   when a key carries a server URL (or uses the default endpoint), the validator contacts
   the AiDotNet license server (a Supabase Edge Function) to confirm the key is valid, what
   tier it belongs to, and whether the machine-activation limit is reached. Results are
   cached for an **offline grace period** so transient network failures don't break loading.
   With no server URL configured, the validator runs in **offline-only** mode and performs
   local format/signature validation.
3. **Assembly integrity** ([`AssemblyIntegrityChecker.cs`](../../src/Helpers/AssemblyIntegrityChecker.cs)) —
   before any crypto operation, official builds verify the embedded build key and that the
   critical encryption types are still present, so a tampered assembly can't silently weaken
   the scheme. Dev/fork builds (no embedded integrity hash) always pass, to allow development.

## Why it exists

AiDotNet is BSL 1.1: free for non-commercial use, paid for commercial use. The save/load
gate is the enforcement point for that commercial tier. It is deliberately narrow — it does
**not** restrict training or inference, only persistence of trained artifacts — so the
library remains fully usable for evaluation, research, and non-commercial work without a key.

## How to obtain and use a key

1. Get a key (free community tier or a paid tier) at <https://aidotnet.dev>.
2. Provide it when constructing models that save/load, e.g. via the license-key option on
   the relevant builder/serializer (an `AiDotNetLicenseKey` with the key string; set
   `ServerUrl` to override the default validation endpoint, or leave the server URL empty
   for offline-only validation).
3. Save/load then works beyond the 10-operation trial.

### Fork / build-from-source behavior

A build from source has no embedded build key. Such a build can train, infer, and develop
freely. It derives its **own** encryption keys, so models it encrypts are loadable by that
same build with the same license key, but are not interchangeable with official-build
artifacts. This is by design and is compatible with BSL 1.1 — the license governs commercial
*use*, and the source is fully available for inspection and contribution.

## What the license server sees

When server-side validation is used, the validator sends the license key (for verification)
and a machine activation identifier (to enforce per-tier activation limits). It does **not**
send model data, weights, training data, or PII. If you require fully air-gapped operation,
use offline-only mode (no server URL) with an appropriate tier.

## Related

- Telemetry (separate, opt-in, off by default) is documented in the telemetry docs and is
  unrelated to licensing — it never carries model or license data.
- License text: [`LICENSE`](../../LICENSE) (BSL 1.1). Attribution and related-project notes:
  [`NOTICE`](../../NOTICE).
