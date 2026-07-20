# aidn2 license signing keys — issuing, CI wiring, and rotation

This directory holds the tooling for the **aidn2** (v2) offline license system: an Ed25519
(EdDSA) token that the SDK verifies *offline* against a public key embedded in the build.

- `keygen.py` — generates one Ed25519 keypair and prints everything each consumer needs
  (private PEM, private PKCS#8 for the server secret, and the public JWK for embedding).
- `aidn2_issuer.py` — mints a signed `aidn2.<claims>.<sig>` token from a private key.

> **Run these on a trusted machine.** Private keys are printed to stdout. Never commit a
> private key. The only private key that should exist long-term lives in a secret store.

## The token grammar

```
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
python keygen.py --kid ci-2026a --jwk-out ci_pubkey.json
# prints: private PEM, private PKCS#8 (base64), and the public JWK (also written to ci_pubkey.json)
```

Keep the private PEM only long enough to mint the token (next step). It does **not** go to
Supabase — the CI key is not a customer-signing key.

## Mint the scoped, short-lived CI token

```bash
python aidn2_issuer.py \
  --private-key-pem ci_private.pem \
  --kid ci-2026a \
  --sub aidotnet-ci \
  --tier enterprise \
  --caps model:save,tensors:save,model:load,tensors:load,model:encrypt \
  --days 30            # short by design: a leaked CI token self-expires
# prints the aidn2.… token
```

Scope it to exactly the capabilities the test suite exercises. `--days 30` is the cap; CI
re-mints on the cadence below.

## Install the two values (the only sensitive step)

`AIDOTNET_LICENSE_KEY` is a **secret** (it is a bearer token). The public JWK is either a
committed file or a repo **variable**. Using the repo API (PyNaCl is required to seal the
secret; `pip install pynacl`):

```bash
# 1) the token → encrypted repo secret
python - <<'PY'
import base64, json, os, urllib.request
from nacl import encoding, public
TOK=os.environ["GH_TOKEN"]; REPO="ooples/AiDotNet"
val=open("ci_token.txt").read().strip()
def api(p, data=None, method="GET"):
    r=urllib.request.Request(f"https://api.github.com/repos/{REPO}{p}",
        data=(json.dumps(data).encode() if data else None), method=method)
    r.add_header("Authorization","token "+TOK); r.add_header("User-Agent","cc")
    return json.load(urllib.request.urlopen(r))
pk=api("/actions/secrets/public-key")
sealed=public.SealedBox(public.PublicKey(pk["key"].encode(), encoding.Base64Encoder)).encrypt(val.encode())
api(f"/actions/secrets/AIDOTNET_LICENSE_KEY", {"encrypted_value":base64.b64encode(sealed).decode(),"key_id":pk["key_id"]}, "PUT")
print("AIDOTNET_LICENSE_KEY set")
PY
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
  an existing JWK). Every package then trusts a scoped, short-exp-token-only CI key. Acceptable
  only if you accept that expanded trust surface.

`keygen.py` already merges kids into one JWK set, so producing a `prod+ci` bundle is:
`python keygen.py --kid ci-2026a --jwk-out <copy-of-committed-LicensePublicKey.json>`.

## Rotation

Rotate the **CI key** whenever a token is about to expire or a key may be exposed:

1. `python keygen.py --kid ci-2026b …` (new kid — never reuse a kid for a new key).
2. Re-mint the CI token under `ci-2026b`, update the `AIDOTNET_LICENSE_KEY` secret.
3. Publish the new public key by whichever scope you chose above. Keep the old kid in the JWK
   set until every build that could still present an old token has cycled, then drop it.

Rotate the **production key** the same way but under `prod-YYYYx`, updating the Supabase
`AIDOTNET_LICENSE_SIGNING_KEY_PKCS8` + `AIDOTNET_LICENSE_KID` secrets and shipping the new
public key in `src/BuildKey/LicensePublicKey.json` (keep the retired prod kid trusted until
all outstanding customer tokens signed by it have expired).

Because a token names its `kid` and the JWK set can hold many kids, rotation never invalidates
still-valid tokens signed by an older, still-trusted key.
