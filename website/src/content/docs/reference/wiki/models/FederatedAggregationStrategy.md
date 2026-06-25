---
title: "FederatedAggregationStrategy"
description: "Specifies which federated aggregation strategy to use."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies which federated aggregation strategy to use.

## How It Works

**For Beginners:** In federated learning, each client trains locally and sends an update.
The server then combines those updates using an aggregation strategy.

## Fields

| Field | Summary |
|:-----|:--------|
| `Boba` | BOBA — Bayesian Optimal Byzantine-robust Aggregation with posterior inference over honest/malicious client labels. |
| `Bucketing` | Bucketing — Randomly partitions clients into buckets before applying a robust aggregation rule, provably improving the breakdown point. |
| `Bulyan` | Bulyan (Multi-Krum selection + trimming). |
| `DivideAndConquer` | DnC — Divide and Conquer. |
| `FLTrust` | FLTrust — Server maintains a root dataset and computes trust scores by comparing client update directions. |
| `FedAa` | FedAA — Adaptive Aggregation. |
| `FedAlign` | FedAlign — Feature alignment across clients using shared anchor representations. |
| `FedAvg` | Federated Averaging (FedAvg). |
| `FedBN` | Federated Batch Normalization (FedBN). |
| `FedDecorr` | FedDecorr — Decorrelation regularizer that encourages diverse feature representations across clients to reduce dimensional collapse. |
| `FedLc` | FedLC — Logit Calibration. |
| `FedMa` | FedMA — Matched Averaging. |
| `FedNtd` | FedNTD — Not-True Distillation. |
| `FedProx` | Federated Proximal (FedProx) for heterogeneity. |
| `FedSam` | FedSAM — Sharpness-Aware Minimization for FL. |
| `Flame` | FLAME — Cosine-similarity filtering with adaptive clipping and DP noise injection for backdoor resistance. |
| `Krum` | Krum (Byzantine-robust selection). |
| `Median` | Coordinate-wise median aggregation. |
| `Moon` | MOON — Model-Contrastive learning. |
| `MultiKrum` | Multi-Krum (select m central updates, then average). |
| `OptiGradTrust` | OptiGradTrust — Optimized gradient trust scoring with EMA-based historical reputation tracking. |
| `Rfa` | Robust Federated Aggregation (geometric median / RFA). |
| `TrimmedMean` | Coordinate-wise trimmed mean aggregation. |
| `WinsorizedMean` | Coordinate-wise winsorized mean aggregation. |

