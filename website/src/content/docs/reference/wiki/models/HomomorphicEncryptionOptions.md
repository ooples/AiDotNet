---
title: "HomomorphicEncryptionOptions"
description: "Configuration options for homomorphic encryption (HE) in federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for homomorphic encryption (HE) in federated learning.

## How It Works

**For Beginners:** Homomorphic encryption lets the server combine encrypted client updates without seeing them in plaintext.
This can enable stronger privacy guarantees, at the cost of compute and bandwidth.

## Properties

| Property | Summary |
|:-----|:--------|
| `BfvFixedPointScale` | Gets or sets the fixed-point scaling factor for BFV encoding. |
| `BfvPlainModulusBitSize` | Gets or sets BFV plain modulus bit size (batching-friendly prime). |
| `CkksCoeffModulusBits` | Gets or sets CKKS coefficient modulus bit sizes. |
| `CkksScale` | Gets or sets CKKS scale used for encoding. |
| `Enabled` | Gets or sets whether homomorphic encryption is enabled. |
| `EncryptedRanges` | Gets or sets which parameter ranges are encrypted when `Mode` is `Hybrid`. |
| `Mode` | Gets or sets the HE mode. |
| `PolyModulusDegree` | Gets or sets the polynomial modulus degree. |
| `Scheme` | Gets or sets the HE scheme. |

