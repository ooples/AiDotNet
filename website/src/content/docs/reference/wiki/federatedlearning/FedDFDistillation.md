---
title: "FedDFDistillation<T>"
description: "FedDF — Federated ensemble distillation using model averaging on unlabeled public data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Distillation`

FedDF — Federated ensemble distillation using model averaging on unlabeled public data.

## For Beginners

Think of this as a "committee vote." Each client model votes on
what the answer should be for some shared unlabeled examples. The server then trains
a fresh model to match the committee's averaged predictions. This means each client
can use a completely different type of model.

## How It Works

FedDF (Lin et al., 2020) performs federated distillation by treating each client model
as an ensemble member. The server distills the ensemble's collective predictions on
unlabeled data into a single global model. Unlike FedMD, FedDF works with completely
different model architectures and does not require labeled public data.

Reference: Lin et al. (2020), "Ensemble Distillation for Robust Model Fusion in Federated Learning".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedDFDistillation(Double,Int32,Double)` | Creates a new FedDF distillation strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateKnowledge(Dictionary<Int32,Matrix<>>,Dictionary<Int32,Double>)` |  |
| `ApplyKnowledge(Vector<>,Matrix<>,Matrix<>,Double)` |  |
| `ExtractKnowledge(Vector<>,Matrix<>)` |  |

