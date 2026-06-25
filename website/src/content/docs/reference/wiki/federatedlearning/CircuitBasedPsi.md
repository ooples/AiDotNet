---
title: "CircuitBasedPsi"
description: "Implements circuit-based Private Set Intersection using garbled circuit evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.PSI`

Implements circuit-based Private Set Intersection using garbled circuit evaluation.

## For Beginners

Imagine a black box that two people feed their lists into.
The box doesn't just find matching items — it can also compute statistics about them
(e.g., "the sum of matching patients' ages") without either person seeing the individual matches.
This is more powerful than basic PSI but also more computationally expensive.

## How It Works

Circuit-based PSI evaluates a comparison circuit over secret-shared inputs. Unlike
DH or OT-based protocols that only reveal the intersection, circuit PSI can compute
arbitrary functions on the intersection (e.g., sum of associated values, count, statistics)
without revealing the actual intersecting elements.

**Complexity:** O(n * m * k) where k is the comparison circuit depth.
More expensive than DH or OT-based PSI but supports richer functionality.

**Security:** Secure against semi-honest adversaries using garbled circuits
(Yao's protocol) or secret-shared circuits (GMW/BGW).

**Reference:** Huang et al., "Private Set Intersection: Are Garbled Circuits
Better than Custom Protocols?", NDSS 2012. Pinkas et al., "Efficient Circuit-Based PSI",
EUROCRYPT 2018.

## Properties

| Property | Summary |
|:-----|:--------|
| `ProtocolName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AreEqual(Byte[],Byte[])` | Constant-time comparison of two byte arrays. |
| `ComputeExactCardinality(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` | Computes the cardinality of the intersection without revealing elements. |
| `ComputeExactIntersection(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` |  |
| `ComputePrf(String,Byte[],Int32)` | Computes a PRF (Pseudorandom Function) value for an element using HMAC-SHA256. |
| `DeriveCircuitKey(Nullable<Int32>,Int32)` | Derives a circuit key from the seed or generates a random one. |

