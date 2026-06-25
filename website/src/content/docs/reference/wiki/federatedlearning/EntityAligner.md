---
title: "EntityAligner"
description: "High-level orchestrator for entity alignment in vertical federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.PSI`

High-level orchestrator for entity alignment in vertical federated learning.

## For Beginners

Think of entity alignment like matching rows in two spreadsheets
that share a common ID column (e.g., patient ID). If Hospital A has patient data in rows 1-1000
and Hospital B has patient data in rows 1-500, entity alignment figures out which rows in A
correspond to which rows in B, based on shared patient IDs, without either hospital seeing
the other's full patient list.

## How It Works

Entity alignment is the first step in any vertical FL pipeline. Before parties can
jointly train a model, they need to identify which entities (patients, customers, transactions, etc.)
exist in all parties' datasets and align their data rows so that features can be correctly paired
during training.

The `EntityAligner` class provides a convenient facade over the PSI protocols.
It handles protocol selection, fuzzy matching, multi-party coordination, and produces
alignment mappings ready for use in VFL training.

**Usage:**

**Multi-party alignment:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EntityAligner` | Initializes a new instance of `EntityAligner` with the Diffie-Hellman protocol as default. |
| `EntityAligner(IPrivateSetIntersection)` | Initializes a new instance of `EntityAligner` with a specified default protocol. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AlignEntities(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` | Aligns entities between two parties using the configured PSI protocol. |
| `AlignMultipleParties(IReadOnlyList<IReadOnlyList<String>>,PsiOptions)` | Aligns entities across three or more parties using multi-party PSI. |
| `CheckOverlapSufficiency(IReadOnlyList<String>,IReadOnlyList<String>,Double,PsiOptions)` | Checks whether there is sufficient overlap between two parties' datasets for viable VFL training. |
| `ComputeOverlapCount(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` | Computes only the cardinality (count) of the intersection between two parties. |
| `SelectProtocol(PsiOptions)` | Selects the appropriate PSI protocol implementation based on the options. |

