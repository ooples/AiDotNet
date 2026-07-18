#!/usr/bin/env python3
"""
v2 license signing keypair bootstrap.

Generates ONE Ed25519 keypair and prints everything needed to wire up offline `aidn2` licensing, in the
exact formats each consumer expects. RUN THIS ON YOUR OWN MACHINE — the private key is printed to stdout and
never leaves your terminal; do not paste it into chat, commit it, or store it unencrypted.

Emits:
  1. AIDOTNET_LICENSE_SIGNING_KEY_PKCS8  — base64(DER PKCS#8) private key. This is what the Supabase edge
     functions (issue-license, get-revocations) import via crypto.subtle.importKey("pkcs8", ...). Set it as a
     function secret (see the printed `supabase secrets set` line). SERVER-SIDE ONLY — never ships to clients.
  2. LicensePublicKey.json               — JWK OKP public key. Embed in BOTH SDKs as the AiDotNet.LicensePublicKey
     resource (src/BuildKey/LicensePublicKey.json) AND set as the CI variable AIDOTNET_LICENSE_PUBLIC_KEY_JSON
     so sonarcloud.yml's inject step writes it before build. Public — safe to commit / expose.
  3. kid                                  — key id; must match between the private key (AIDOTNET_LICENSE_KID
     secret) and every token/JWK signed with it.

Two keypairs per the design: run once for the PRODUCTION customer-signing key (kid e.g. prod-2026a, private
key = Supabase secret) and once for the CI-only key (kid e.g. ci-2026a, private key = a CI secret used by a
short-exp scope:ci token). Keep the two private keys in different places so a CI compromise can't mint
customer licenses.

Requires: pip install cryptography
"""
import argparse
import base64
import json
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64std(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a v2 license signing keypair.")
    ap.add_argument("--kid", required=True, help="key id (e.g. prod-2026a or ci-2026a)")
    ap.add_argument("--jwk-out", default="LicensePublicKey.json",
                    help="path to write the public JWK (embed + CI var). Default: ./LicensePublicKey.json")
    args = ap.parse_args()

    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    # DER PKCS#8 (unencrypted) — the exact bytes crypto.subtle.importKey("pkcs8", ...) wants, base64'd.
    der_pkcs8 = priv.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = pub.public_bytes_raw()  # 32 bytes

    jwk = {"keys": [{"kty": "OKP", "crv": "Ed25519", "kid": args.kid, "x": b64url(pub_raw)}]}
    with open(args.jwk_out, "w", encoding="utf-8") as f:
        json.dump(jwk, f, separators=(",", ":"))
    jwk_compact = json.dumps(jwk, separators=(",", ":"))

    print("=== v2 license signing keypair ===")
    print(f"kid: {args.kid}\n")

    print("--- 1. SERVER (Supabase function secret) — PRIVATE, do not share/commit ---")
    print("Run against the license project (yfkqwpgjahoamlgckjib):")
    print(f"  supabase secrets set \\")
    print(f"    AIDOTNET_LICENSE_SIGNING_KEY_PKCS8={b64std(der_pkcs8)} \\")
    print(f"    AIDOTNET_LICENSE_KID={args.kid}\n")

    print("--- 2. SDK embed + CI variable (PUBLIC — safe to commit/expose) ---")
    print(f"Wrote JWK -> {args.jwk_out}")
    print("Copy it to BOTH:  src/BuildKey/LicensePublicKey.json  (AiDotNet)  and the AiDotNet.Tensors equivalent.")
    print("Set the CI variable so the build injects it:")
    print(f"  gh variable set AIDOTNET_LICENSE_PUBLIC_KEY_JSON --repo ooples/AiDotNet --body '{jwk_compact}'")
    print(f"  gh variable set AIDOTNET_LICENSE_PUBLIC_KEY_JSON --repo ooples/AiDotNet.Tensors --body '{jwk_compact}'\n")

    print(f"public x (base64url, 32B): {b64url(pub_raw)}")
    print("\nNEXT: deploy issue-license + get-revocations (they read the secret above), then re-issue")
    print("customer keys / mint the CI token with aidn2_issuer.py using this same kid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
