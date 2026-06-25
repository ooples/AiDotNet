---
title: "FuzzyPsi"
description: "Implements approximate entity matching using multiple similarity strategies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.PSI`

Implements approximate entity matching using multiple similarity strategies.

## For Beginners

In real-world data, the same entity may have slightly different
identifiers across organizations. "John Smith" at one hospital might be "Jon Smith" or
"SMITH, JOHN" at another. Fuzzy matching finds these approximate matches.

## How It Works

This class provides a unified PSI wrapper around fuzzy matching that first applies
approximate matching to find candidate pairs, then runs the underlying PSI protocol
on the matched identifiers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FuzzyPsi(IPrivateSetIntersection)` | Initializes a new instance of `FuzzyPsi` wrapping an inner PSI protocol. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ProtocolName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeExactIntersection(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` |  |

