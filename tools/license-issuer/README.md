# aidn2 license signing keys — issuing, CI wiring, and rotation

This directory holds the tooling for the **aidn2** (v2) offline license system: an Ed25519
(EdDSA) token that the SDK verifies *offline* against a public key embedded in the build.

- `keygen.py` — generates one Ed25519 keypair, optionally writes its private PEM, prints the
  private PKCS#8 server-secret value, and writes the public JWK for embedding.
- `aidn2_issuer.py` — mints a signed `aidn2.<claims>.<sig>` token from a private key and can
  write only that token to a protected output file for automation.

> **Run these on a trusted machine.** Private keys are printed to stdout. Never commit a
> private key. The only private key that should exist long-term lives in a secret store.

## The token grammar

```text
aidn2.<base64url(claims_json)>.<base64url(ed25519_signature)>
```

The signature is over the **raw UTF-8 bytes** of `claims_json` — canonical bytes matter, so
always mint with `aidn2_issuer.py` (never hand-assemble). Claim fields mirror
`src/Helpers/LicenseClaims.cs` and the Supabase signer `website/supabase/functions/_shared/aidn2.ts`:
`sub, tier, seats, iat, exp, kid, jti, caps[]`, and optional `scope` / `mach`.

## Two keys, two very different trust levels

| Key | `kid` (example) | Private half lives in | Public half embedded in | Purpose |
|-----|-----------------|-----------------------|-------------------------|---------|
| **Production** | `prod-2026a` | Supabase function secret `AIDOTNET_LICENSE_SIGNING_KEY_PKCS8` (+ `AIDOTNET_LICENSE_KID`) | committed `src/BuildKey/LicensePublicKey.json`, shipped in every NuGet | signs **customer** licenses (via `issue-license`) |
| **CI** | `ci-2026a` | GitHub Actions secret (never elsewhere) | see the trust-scope decision below | signs the **CI test** license so tests run offline-licensed |

The two keys are independent. Compromise of the CI key must never let someone mint a
**customer**-trusted license, and vice-versa. That is the whole point of separate `kid`s.

## What is already wired (do not rebuild)

- `src/AiDotNet.csproj` embeds `BuildKey/LicensePublicKey.json` → resource `AiDotNet.LicensePublicKey`
  (and `LicenseRevocation.json` → `AiDotNet.LicenseRevocation`). `LicensePublicKeyProvider`
  reads that JWK set; multiple `kid`s can coexist, which is what makes rotation seamless.
- CI (`release-please.yml`, `sonarcloud.yml`) injects `LicensePublicKey.json` from the repo
  var/secret `AIDOTNET_LICENSE_PUBLIC_KEY_JSON` when set, else uses the committed file.
- Test workflows (`sonarcloud.yml`, `heavy-timeout-nightly.yml`) pass the minted CI token to
  tests as the secret `AIDOTNET_LICENSE_KEY`.

So enabling durable offline-licensed CI is only: **generate the CI key, mint a scoped CI
token, and install two values.** No code changes are required.

## Generate the CI keypair (once)

```bash
cd tools/license-issuer
umask 077
work_dir="$(mktemp -d)"
cp ../../src/BuildKey/LicensePublicKey.json "$work_dir/ci_merged_jwks.json"
python keygen.py \
  --kid ci-2026a \
  --jwk-out "$work_dir/ci_merged_jwks.json" \
  --private-key-out "$work_dir/ci_private.pem"
```

This starts the test-only bundle from the committed production JWK and appends `ci-2026a`.
Keep `ci_private.pem` only long enough to mint the token (next step). It does **not** go to
Supabase — the CI key is not a customer-signing key. Keep the same shell open so `$work_dir`
remains available to the mint and install commands below.

## Mint the scoped, short-lived CI token

```bash
python aidn2_issuer.py \
  --private-key-pem "$work_dir/ci_private.pem" \
  --kid ci-2026a \
  --sub aidotnet-ci \
  --tier enterprise \
  --caps model:save,tensors:save,model:load,tensors:load,model:encrypt \
  --days 30 \
  --token-out "$work_dir/ci_token.txt" \
  --out-dir "$work_dir/issuer-output"
```

Scope it to exactly the capabilities the test suite exercises. `--days 30` is the cap; CI
re-mints on the cadence below. The install step consumes `ci_token.txt` and the merged JWK
created above; `issuer-output/LicensePublicKey.json` is the issuer's single-key verification
artifact and is not the merged workflow variable.

## Install the two values (the only sensitive step)

`AIDOTNET_LICENSE_KEY` is a **secret** (it is a bearer token). The merged public JWK is a
repo **variable** used only by test workflows. Install both with an authenticated GitHub CLI:

```bash
gh auth status
gh secret set AIDOTNET_LICENSE_KEY --repo ooples/AiDotNet < "$work_dir/ci_token.txt"
gh variable set AIDOTNET_LICENSE_PUBLIC_KEY_JSON_TEST \
  --repo ooples/AiDotNet \
  --body "$(cat "$work_dir/ci_merged_jwks.json")"
```

For the **public** key, pick one of the two trust scopes:

### Trust-scope decision (make this deliberately)

The var `AIDOTNET_LICENSE_PUBLIC_KEY_JSON` is consumed by **both** the release build and the
test builds. If you put the CI key there, **published NuGets will also trust the CI key** —
i.e. a leaked CI private key could mint licenses accepted by shipped packages until you rotate.

- **Recommended — CI key never ships.** Keep `AIDOTNET_LICENSE_PUBLIC_KEY_JSON` = production
  key(s) only. Inject a *merged* JWK (prod + ci) into **test-only** workflows via a separate
  var (e.g. `AIDOTNET_LICENSE_PUBLIC_KEY_JSON_TEST`) referenced only by `sonarcloud.yml` /
  `heavy-timeout-nightly.yml`, not by `release-please.yml`. This requires a one-line workflow
  edit in those two files and keeps the shipped trust boundary = production only.
- **Simpler, weaker — CI key ships.** Merge `ci-2026a` into the committed
  `src/BuildKey/LicensePublicKey.json` (keygen appends rather than overwrites when pointed at
  an existing JWK). Every package then grants full signing trust to that key: anyone holding its
  private half can mint arbitrary claims, capabilities, and expiration dates until the public key
  is removed in an emergency rotation. Scope and expiration are token claims enforced by issuer
  policy; they are not intrinsic restrictions on the signing key. Accept this option only if that
  expanded trust surface is intentional.

`keygen.py` already merges kids into one JWK set, so producing a `prod+ci` bundle is:
copy the committed production bundle, then pass that concrete copy to `--jwk-out`, as shown in
the generation commands above.

## Rotation

Rotate the **CI key** whenever a token is about to expire or a key may be exposed. These commands
preserve the currently trusted kids during the overlap and create concrete rotation artifacts:

```bash
rotation_dir="$(mktemp -d)"
gh variable get AIDOTNET_LICENSE_PUBLIC_KEY_JSON_TEST \
  --repo ooples/AiDotNet > "$rotation_dir/ci_merged_jwks.json"
python keygen.py \
  --kid ci-2026b \
  --jwk-out "$rotation_dir/ci_merged_jwks.json" \
  --private-key-out "$rotation_dir/ci_private.pem"
python aidn2_issuer.py \
  --private-key-pem "$rotation_dir/ci_private.pem" \
  --kid ci-2026b \
  --sub aidotnet-ci \
  --tier enterprise \
  --caps model:save,tensors:save,model:load,tensors:load,model:encrypt \
  --days 30 \
  --token-out "$rotation_dir/ci_token.txt" \
  --out-dir "$rotation_dir/issuer-output"
gh secret set AIDOTNET_LICENSE_KEY --repo ooples/AiDotNet < "$rotation_dir/ci_token.txt"
gh variable set AIDOTNET_LICENSE_PUBLIC_KEY_JSON_TEST \
  --repo ooples/AiDotNet \
  --body "$(cat "$rotation_dir/ci_merged_jwks.json")"
```

Never reuse a `kid` for a new key. Keep the old kid in the JWK set until every build that could
still present an old token has cycled, then remove it from the variable. Securely delete the
temporary directory after installation or transfer the private key to its intended secret store.

Rotate the **production key** the same way but under `prod-YYYYx`, updating the Supabase
`AIDOTNET_LICENSE_SIGNING_KEY_PKCS8` + `AIDOTNET_LICENSE_KID` secrets and shipping the new
public key in `src/BuildKey/LicensePublicKey.json` (keep the retired prod kid trusted until
all outstanding customer tokens signed by it have expired).

Because a token names its `kid` and the JWK set can hold many kids, rotation never invalidates
still-valid tokens signed by an older, still-trusted key.
