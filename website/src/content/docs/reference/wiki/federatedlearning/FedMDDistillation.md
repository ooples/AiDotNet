---
title: "FedMDDistillation<T>"
description: "FedMD — Model-agnostic federated learning via mutual distillation on a public dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Distillation`

FedMD — Model-agnostic federated learning via mutual distillation on a public dataset.

## For Beginners

All clients are given the same "test questions" (public dataset).
Each client submits their answers (predicted probabilities). The server averages all
answers to get a "consensus answer," then each client learns from this consensus.
This way, a simple model on a phone can learn from a powerful model on a server.

## How It Works

FedMD (Li & Wang, 2019) enables heterogeneous model architectures across clients.
Each client computes logits on a shared public dataset, the server averages the logits,
and clients distill the consensus logits into their local models.

Reference: Li & Wang (2019), "FedMD: Heterogeneous Federated Learning via Model Distillation".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedMDDistillation(Double,Double,Int32,Double)` | Creates a new FedMD distillation strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateKnowledge(Dictionary<Int32,Matrix<>>,Dictionary<Int32,Double>)` |  |
| `ApplyKnowledge(Vector<>,Matrix<>,Matrix<>,Double)` |  |
| `ExtractKnowledge(Vector<>,Matrix<>)` |  |

