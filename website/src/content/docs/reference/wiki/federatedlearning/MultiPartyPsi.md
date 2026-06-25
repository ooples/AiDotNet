---
title: "MultiPartyPsi"
description: "Implements multi-party Private Set Intersection for 3 or more parties."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.PSI`

Implements multi-party Private Set Intersection for 3 or more parties.

## For Beginners

Two-party PSI is like two people comparing guest lists.
Multi-party PSI is like a group of people finding guests who are on everyone's list.
For example, 3 hospitals finding patients who appear in all 3 systems.

## How It Works

Multi-party PSI extends two-party protocols to find the intersection common
to all participating parties. The result contains only elements present in every
party's set.

This implementation uses a star topology: a designated leader runs pairwise
two-party PSI with each other party, then intersects all pairwise results to produce
the global intersection.

**Complexity:** (P-1) two-party PSI executions where P is the number of parties,
plus O(n) intersection of pairwise results.

**Security:** In the star topology, the leader learns all pairwise intersections
(which is more than just the global intersection). For stronger security, tree or
ring topologies can be used at the cost of more rounds.

**Reference:** Kolesnikov et al., "Efficient Batched Oblivious PRF with Applications
to Private Set Intersection", ACM CCS 2016. Li et al., "Lightweight MP-PSI", 2025.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiPartyPsi` | Initializes a new instance of `MultiPartyPsi` using Diffie-Hellman PSI for pairwise intersections. |
| `MultiPartyPsi(IPrivateSetIntersection)` | Initializes a new instance of `MultiPartyPsi` using a specified two-party PSI protocol. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ProtocolName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeExactIntersection(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` |  |
| `ComputeMultiPartyIntersection(IReadOnlyList<IReadOnlyList<String>>,PsiOptions)` | Computes the intersection across multiple parties using the star topology. |

