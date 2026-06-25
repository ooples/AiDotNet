---
title: "UnbalancedPsi"
description: "Implements PSI optimized for asymmetric (unbalanced) set sizes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.PSI`

Implements PSI optimized for asymmetric (unbalanced) set sizes.

## For Beginners

Imagine looking up 10 phone numbers in a phonebook with
10 million entries. It's much faster to search the phonebook for your 10 numbers
than to compare every pair. Unbalanced PSI works similarly: the small party's elements
are looked up in a compact representation of the large party's set.

## How It Works

When one party's set is much larger than the other's (e.g., 10M records vs 1K),
standard PSI protocols waste computation on the larger set. Unbalanced PSI optimizes
by having the smaller party (client) do minimal work while the larger party (server)
builds a compressed data structure for efficient querying.

**Algorithm:**

**Complexity:** O(n_small * log(n_large)) computation, O(n_small) communication.
Huge savings when set sizes differ by orders of magnitude.

**Reference:** Chen et al., "Labeled PSI from Fully Homomorphic Encryption with
Malicious Security", ACM CCS 2018. Kiss et al., "Private Set Intersection for
Unequal Set Sizes", PETS 2017.

## Properties

| Property | Summary |
|:-----|:--------|
| `ProtocolName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeExactCardinality(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` |  |
| `ComputeExactIntersection(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` |  |
| `ComputeTag(String,Byte[],Int32)` | Computes a PRF tag for an element. |
| `DeriveKey(Nullable<Int32>,Int32)` | Derives a PRF key from the seed or generates a random one. |

