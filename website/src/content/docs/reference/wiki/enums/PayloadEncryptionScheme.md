---
title: "PayloadEncryptionScheme"
description: "Specifies the encryption scheme applied to the model payload within an AIMF envelope."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the encryption scheme applied to the model payload within an AIMF envelope.

## How It Works

**For Beginners:** When a model is saved with encryption enabled, the payload (model weights)
is encrypted using a license key. The header remains plaintext so tools can inspect model metadata
(type, shapes, whether it's encrypted) without needing the key. Only the actual model data
requires a valid license key to decrypt and load.

## Fields

| Field | Summary |
|:-----|:--------|
| `AesGcm256` | Payload is encrypted using AES-256-GCM with a key derived from a license key via PBKDF2-SHA256. |
| `AesGcm256Signed` | Payload is encrypted using AES-256-GCM with enhanced key derivation that incorporates a build-time signing key and optional server-side decryption token. |
| `None` | Payload is stored as plaintext (no encryption). |

