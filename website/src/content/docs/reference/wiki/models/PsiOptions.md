---
title: "PsiOptions"
description: "Configuration options for Private Set Intersection (PSI) protocols."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Private Set Intersection (PSI) protocols.

## For Beginners

PSI lets two or more parties find which IDs they have in common
without revealing IDs that aren't shared. This is the first step in vertical federated learning:
before parties can jointly train a model, they need to know which entities (patients, customers, etc.)
they both have data for.

## How It Works

Example: Two hospitals want to jointly train a model on shared patients:

## Properties

| Property | Summary |
|:-----|:--------|
| `BloomFilterFalsePositiveRate` | Gets or sets the false-positive rate for Bloom filter based PSI. |
| `BloomFilterHashCount` | Gets or sets the number of hash functions used in Bloom filter PSI. |
| `CardinalityOnly` | Gets or sets whether to only compute the intersection cardinality (count) without revealing the actual intersecting elements. |
| `FuzzyMatch` | Gets or sets options for fuzzy (approximate) entity matching. |
| `HashFunction` | Gets or sets the hash function used for element hashing within the protocol. |
| `MaxSetSize` | Gets or sets the maximum expected set size for memory pre-allocation. |
| `NumberOfParties` | Gets or sets the number of parties participating in the PSI protocol. |
| `Protocol` | Gets or sets the PSI protocol to use for computing the set intersection. |
| `RandomSeed` | Gets or sets the random seed for reproducible protocol execution. |
| `SecurityParameter` | Gets or sets the cryptographic security parameter in bits. |

