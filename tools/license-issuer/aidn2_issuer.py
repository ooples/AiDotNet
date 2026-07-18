#!/usr/bin/env python3
"""
aidn2 license issuer — mints offline-verifiable Ed25519 (EdDSA) license tokens.

Token grammar (must match src/Helpers/AsymmetricLicenseVerifier.cs):
    aidn2.<base64url(claims_json)>.<base64url(ed25519_signature)>
The Ed25519 signature is computed over the RAW UTF-8 bytes of claims_json (the exact
bytes recovered by base64url-decoding the middle segment). The client re-verifies those
exact bytes, so JSON field order/whitespace is irrelevant to verification — only the
signed bytes matter. Claims fields mirror src/Helpers/LicenseClaims.cs.

Outputs:
  - the aidn2 token           -> set as the AIDOTNET_LICENSE_KEY secret
  - the public-key JWK (OKP)  -> src/BuildKey/LicensePublicKey.json (embedded at build)
  - the private key (PEM)     -> keep in a secret store; NEVER commit. Signs future tokens.

Design note: in production the PRIVATE key must live on the license server and never
ship. For a CI-only enterprise key you may generate once and store the private half as a
CI secret. Pass --private-key-pem to reuse an existing server-held key instead of
generating a fresh one (so the embedded public key / kid stay stable).
"""
import argparse
import base64
import datetime as dt
import json
import sys

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def build_claims(sub, tier, seats, kid, days) -> bytes:
    now = dt.datetime.now(dt.timezone.utc)
    exp = now + dt.timedelta(days=days)
    # Field order mirrors LicenseClaims declaration; compact (no spaces) like Formatting.None.
    claims = {
        "sub": sub,
        "tier": tier,
        "seats": seats,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "kid": kid,
        "alg": "EdDSA",
    }
    return json.dumps(claims, separators=(",", ":")).encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Mint an aidn2 offline license token.")
    ap.add_argument("--sub", default="aidotnet-ci", help="subject / license holder")
    ap.add_argument("--tier", default="enterprise",
                    choices=["community", "pro", "professional", "enterprise"],
                    help="license tier (enterprise/pro lift the persistence op cap)")
    ap.add_argument("--seats", type=int, default=1000, help="seat count (advisory for offline tokens)")
    ap.add_argument("--kid", default="ci-2026a", help="key id; must match the embedded public key's kid")
    ap.add_argument("--days", type=int, default=1825, help="validity in days (default 5 years)")
    ap.add_argument("--private-key-pem", default=None,
                    help="path to an existing Ed25519 private key PEM to reuse (else generate a fresh one)")
    ap.add_argument("--out-dir", default=".", help="directory to write private_key.pem + LicensePublicKey.json")
    args = ap.parse_args()

    if args.private_key_pem:
        with open(args.private_key_pem, "rb") as f:
            priv = serialization.load_pem_private_key(f.read(), password=None)
        if not isinstance(priv, Ed25519PrivateKey):
            print("ERROR: provided PEM is not an Ed25519 private key", file=sys.stderr)
            return 2
    else:
        priv = Ed25519PrivateKey.generate()

    pub: Ed25519PublicKey = priv.public_key()
    pub_raw = pub.public_bytes_raw()  # 32 bytes

    claims_bytes = build_claims(args.sub, args.tier, args.seats, args.kid, args.days)
    signature = priv.sign(claims_bytes)  # Ed25519 -> 64 bytes
    token = "aidn2." + b64url(claims_bytes) + "." + b64url(signature)

    # Self-verify (fail closed) before emitting anything.
    pub.verify(signature, claims_bytes)

    jwk = {"keys": [{"kty": "OKP", "crv": "Ed25519", "kid": args.kid, "x": b64url(pub_raw)}]}

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    jwk_path = os.path.join(args.out_dir, "LicensePublicKey.json")
    with open(jwk_path, "w", encoding="utf-8") as f:
        json.dump(jwk, f, separators=(",", ":"))

    priv_path = os.path.join(args.out_dir, "private_key.pem")
    if not args.private_key_pem:
        with open(priv_path, "wb") as f:
            f.write(priv.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ))

    print("=== aidn2 license issued ===")
    print(f"kid:            {args.kid}")
    print(f"tier/seats:     {args.tier} / {args.seats}")
    print(f"claims:         {claims_bytes.decode()}")
    print()
    print("AIDOTNET_LICENSE_KEY (secret) =")
    print(token)
    print()
    print(f"public JWK -> {jwk_path}  (embed as src/BuildKey/LicensePublicKey.json)")
    print(f"public x (base64url 32B):   {b64url(pub_raw)}")
    if not args.private_key_pem:
        print(f"private key -> {priv_path}  (KEEP SECRET — never commit; store in the signing secret store)")
    print()
    print("self-verify: OK (signature valid over the exact claim bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
