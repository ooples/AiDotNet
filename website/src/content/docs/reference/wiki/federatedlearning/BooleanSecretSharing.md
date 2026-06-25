---
title: "BooleanSecretSharing"
description: "Implements XOR-based boolean secret sharing for bitwise operations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.MPC`

Implements XOR-based boolean secret sharing for bitwise operations.

## For Beginners

Boolean secret sharing splits a bit string into random XOR shares.
To share a bit b among n parties: generate n-1 random bits r1..r(n-1), and compute
the last share as b XOR r1 XOR r2 XOR ... XOR r(n-1). To reconstruct, XOR all shares
together.

## How It Works

**Why both arithmetic AND boolean sharing?** Different operations are cheaper in
different representations:

Hybrid MPC uses boolean sharing for comparisons and bit manipulations, and arithmetic
sharing for linear algebra. Converting between the two is called "share conversion".

**Reference:** ABY framework (NDSS 2015) for arithmetic, boolean, and Yao sharing.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BooleanSecretSharing(Int32)` | Initializes a new instance of `BooleanSecretSharing`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateAndTriple(Int32)` | Generates an AND triple (u, v, w) where w = u AND v, split into boolean shares. |
| `Reconstruct(Byte[][])` | Reconstructs the secret by XORing all shares together. |
| `SecureAnd(Byte[][],Byte[][],BooleanTriple)` | Performs AND on two sets of boolean shares using pre-shared correlated randomness. |
| `SecureXor(Byte[][],Byte[][])` | Performs XOR on two sets of boolean shares (local operation — no communication). |
| `Share(Byte[])` | Splits a byte array into XOR shares for the specified number of parties. |

