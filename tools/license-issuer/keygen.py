#!/usr/bin/env python3
"""
v2 license signing keypair bootstrap.

Generates ONE Ed25519 keypair and prints everything needed to wire up offline `aidn2` licensing, in the
exact formats each consumer expects. RUN THIS ON YOUR OWN MACHINE. Use --private-key-out when the PEM is
needed by aidn2_issuer.py; do not paste it into chat, commit it, or store it unencrypted.

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
import os
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
    ap.add_argument("--private-key-out", default=None,
                    help="optional path to write the private Ed25519 key as owner-only PKCS#8 PEM")
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

    new_key = {"kty": "OKP", "crv": "Ed25519", "kid": args.kid, "x": b64url(pub_raw)}
    keys = [new_key]
    # MERGE mode: if the target bundle already exists, PRESERVE its keys and append this one, so running
    # keygen for prod-2026a and then ci-2026a yields ONE bundle carrying BOTH kids (the release/CI embed the
    # design requires) instead of the second run overwriting the first. A duplicate kid is refused.
    if os.path.exists(args.jwk_out):
        try:
            with open(args.jwk_out, encoding="utf-8") as f:
                existing = json.load(f).get("keys", [])
        except (json.JSONDecodeError, OSError) as e:
            print(f"ERROR: {args.jwk_out} exists but is not a readable JWK set: {e}", file=sys.stderr)
            return 2
        if any(k.get("kid") == args.kid for k in existing):
            print(f"ERROR: kid '{args.kid}' is already in {args.jwk_out}; refusing to duplicate. "
                  f"Use a different --kid, or remove the existing entry first.", file=sys.stderr)
            return 2
        keys = existing + [new_key]

    jwk = {"keys": keys}
    with open(args.jwk_out, "w", encoding="utf-8") as f:
        json.dump(jwk, f, separators=(",", ":"))
    jwk_compact = json.dumps(jwk, separators=(",", ":"))

    if args.private_key_out:
        parent = os.path.dirname(os.path.abspath(args.private_key_out))
        os.makedirs(parent, exist_ok=True)
        fd = os.open(args.private_key_out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, OSError):
            pass
        with os.fdopen(fd, "wb") as f:
            f.write(priv.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ))

    print("=== v2 license signing keypair ===")
    print(f"kid: {args.kid}\n")

    print("--- 1. SERVER (Supabase function secret) — PRIVATE, do not share/commit ---")
    print("Run against the license project (yfkqwpgjahoamlgckjib):")
    print(f"  supabase secrets set \\")
    print(f"    AIDOTNET_LICENSE_SIGNING_KEY_PKCS8={b64std(der_pkcs8)} \\")
    print(f"    AIDOTNET_LICENSE_KID={args.kid}\n")

    print("--- 2. SDK embed + CI variable (PUBLIC — safe to commit/expose) ---")
    print(f"Wrote JWK -> {args.jwk_out}")
    if args.private_key_out:
        print(f"Wrote private PEM -> {args.private_key_out}  (KEEP SECRET — never commit)")
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
