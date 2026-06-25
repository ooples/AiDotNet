---
title: "H2PCAlgorithm<T>"
description: "H2PC — Hybrid HPC (Hybrid Parents and Children) algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Hybrid`

H2PC — Hybrid HPC (Hybrid Parents and Children) algorithm.

## For Beginners

H2PC is a refined version of MMHC with a smarter first phase.
It finds candidate parents/children more accurately before running the scoring step,
leading to better results especially with smaller datasets.

## How It Works

H2PC uses a two-phase approach where the first phase identifies candidate parents
and children for each variable using a combination of marginal and conditional
association tests (the HPC algorithm), and the second phase uses hill climbing with
BIC scoring restricted to the candidates. The HPC phase differs from MMPC by using
a recursive decomposition: first find spouses of target's neighbors to improve accuracy.

**Algorithm:**

- **HPC phase (per variable):**
- Forward: add variable with max min-association (like MMPC)
- Backward: remove false positives via CI tests
- Spouse discovery: for each neighbor n of target, check if any variable

is associated with target given n (spouse relationship)

- **Hill climbing phase:** greedy DAG construction using BIC within candidate sets

Reference: Gasse et al. (2014), "A Hybrid Algorithm for Bayesian Network Structure
Learning with Application to Multi-Label Learning", Expert Systems with Applications.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `H2PCAlgorithm(CausalDiscoveryOptions)` | Initializes H2PC with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |
| `FindHPC(Matrix<>,Int32,Int32,Int32)` | HPC (Hybrid Parents and Children) — finds candidate PC set with spouse discovery. |

