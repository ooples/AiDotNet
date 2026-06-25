---
title: "IPrivateSetIntersection"
description: "Defines the interface for Private Set Intersection protocols."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.PSI`

Defines the interface for Private Set Intersection protocols.

## For Beginners

Imagine two hospitals that want to find shared patients
without revealing their full patient lists to each other. PSI solves this:
each hospital inputs its patient IDs, and the protocol outputs only the IDs
that appear in both hospitals' records, without either hospital learning about
patients unique to the other.

## How It Works

PSI allows two or more parties to compute the intersection of their sets
without revealing elements not in the intersection. This is the foundational
building block for entity alignment in vertical federated learning.

All implementations are simulation-safe: they operate on in-memory data
structures rather than requiring actual network communication, making them
suitable for testing and single-machine VFL experiments.

## Properties

| Property | Summary |
|:-----|:--------|
| `ProtocolName` | Gets the name of this PSI protocol. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCardinality(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` | Computes only the cardinality (count) of the intersection without revealing the actual elements. |
| `ComputeIntersection(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` | Computes the intersection between the local party's IDs and the remote party's IDs. |

