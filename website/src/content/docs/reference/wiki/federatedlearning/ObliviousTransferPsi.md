---
title: "ObliviousTransferPsi"
description: "Implements Oblivious Transfer based Private Set Intersection using cuckoo hashing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.PSI`

Implements Oblivious Transfer based Private Set Intersection using cuckoo hashing.

## For Beginners

Imagine a library catalog system:

## How It Works

OT-based PSI is the fastest known approach for large-scale set intersection.
The receiver inserts elements into a cuckoo hash table, and the sender uses
Oblivious Transfer extensions to compare against each bin without learning which
bins contain elements.

**Complexity:** O(n) computation and communication with small constants thanks to OT extensions.

**Security:** Secure against semi-honest adversaries in the random oracle model.

**Reference:** Pinkas et al., "Efficient Circuit-Based PSI via Cuckoo Hashing",
EUROCRYPT 2018.

## Properties

| Property | Summary |
|:-----|:--------|
| `ProtocolName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildCuckooHashTable(IReadOnlyList<String>,Int32,Int32[])` | Builds a cuckoo hash table from the local ID set. |
| `ComputeExactIntersection(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` |  |

