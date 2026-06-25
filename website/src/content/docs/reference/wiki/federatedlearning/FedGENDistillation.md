---
title: "FedGENDistillation<T>"
description: "FedGEN — Data-free federated distillation using a lightweight generator on the server."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Distillation`

FedGEN — Data-free federated distillation using a lightweight generator on the server.

## For Beginners

Instead of needing a shared test dataset (like FedMD), the server
creates its own fake data that captures what all clients know. Clients only share statistical
summaries (like "class A has these average features"), which is more privacy-preserving
than sharing predictions on real data.

## How It Works

FedGEN (Zhu et al., 2021) eliminates the need for a public dataset by training a
small generator on the server that produces synthetic samples. Clients share class-conditional
statistics (means and variances) rather than logits. The server's generator learns to
produce samples that match the consensus statistics, then distills this knowledge back.

Reference: Zhu et al. (2021), "Data-Free Knowledge Distillation for Heterogeneous Federated Learning".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedGENDistillation(Int32,Int32,Int32,Double,Int32)` | Creates a new FedGEN distillation strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateKnowledge(Dictionary<Int32,Matrix<>>,Dictionary<Int32,Double>)` |  |
| `ApplyKnowledge(Vector<>,Matrix<>,Matrix<>,Double)` |  |
| `ExtractKnowledge(Vector<>,Matrix<>)` |  |

