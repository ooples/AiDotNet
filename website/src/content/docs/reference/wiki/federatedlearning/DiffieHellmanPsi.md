---
title: "DiffieHellmanPsi"
description: "Implements Diffie-Hellman based Private Set Intersection using commutative encryption."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.PSI`

Implements Diffie-Hellman based Private Set Intersection using commutative encryption.

## For Beginners

Think of this like two people mixing paint colors:

## How It Works

This protocol exploits the commutative property of exponentiation in a prime-order group:
H(x)^(a*b) = (H(x)^a)^b = (H(x)^b)^a. Both parties hash their elements, raise to their secret
exponents, exchange and re-exponentiate, then compare the doubly-encrypted values.

**Complexity:** O(n+m) computation, O(n+m) communication where n and m are set sizes.

**Security:** Secure against semi-honest adversaries under the Decisional Diffie-Hellman assumption.

**Reference:** Meadows, "A More Efficient Cryptographic Matchmaking Protocol for Use in the
Absence of a Continuously Available Third Party", IEEE S&P 1986.

## Properties

| Property | Summary |
|:-----|:--------|
| `ProtocolName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeExactIntersection(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` |  |
| `GenerateExponents(Nullable<Int32>)` | Generates private exponents for both parties. |
| `HashToGroup(String,Int32)` | Hashes a string identifier to a group element using hash-to-curve-like approach. |

