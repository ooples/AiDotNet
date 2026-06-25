---
title: "SecureGradientExchange<T>"
description: "Provides encryption for gradient tensors exchanged between parties in vertical FL."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Vertical`

Provides encryption for gradient tensors exchanged between parties in vertical FL.

## For Beginners

In vertical FL, parties exchange embedding values and gradients.
These intermediate values can leak information about the raw features. Secure gradient
exchange encrypts these values during transit so that an eavesdropper or curious party
cannot analyze them.

## How It Works

This implementation uses symmetric encryption (AES-GCM) with per-session keys
derived via Diffie-Hellman key agreement. In production, each party pair would negotiate
a shared secret; in simulation mode, keys are derived deterministically from a seed.

**Two modes of protection:**

**Reference:** Based on techniques from Google's production FL system and FATE VFL framework.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SecureGradientExchange(Boolean,Nullable<Int32>)` | Initializes a new instance of `SecureGradientExchange`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DecryptGradients(Tensor<>,Tensor<>)` | Decrypts gradients using the nonce to regenerate the same keystream. |
| `EncryptGradients(Tensor<>)` | Encrypts gradients by converting to double values and XOR-encrypting with a keystream derived from the session key. |
| `MaskGradients(Tensor<>)` | Masks gradients with additive random noise. |
| `ProtectGradients(Tensor<>)` | Protects a gradient tensor before sending it to another party. |
| `RecoverGradients(Tensor<>,Tensor<>)` | Recovers the original gradient tensor from a protected representation. |
| `UnmaskGradients(Tensor<>,Tensor<>)` | Unmasks gradients by subtracting the mask. |

