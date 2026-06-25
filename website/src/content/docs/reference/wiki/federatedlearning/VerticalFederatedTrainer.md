---
title: "VerticalFederatedTrainer<T>"
description: "Main orchestrator for vertical federated learning training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Vertical`

Main orchestrator for vertical federated learning training.

## For Beginners

This is the central coordinator that manages the entire VFL
training pipeline. It brings together all the components:

## How It Works

**Training loop per batch:**

**Reference:** Based on the FATE framework architecture and VFLAIR (ICLR 2025).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VerticalFederatedTrainer(VerticalFederatedLearningOptions,ILabelProtector<>)` | Initializes a new instance of `VerticalFederatedTrainer`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AlignEntities(PsiOptions)` |  |
| `Predict(IReadOnlyList<Int32>)` |  |
| `RegisterParty(IVerticalParty<>)` |  |
| `Train` |  |
| `TrainEpoch` |  |
| `UnlearnEntities(IReadOnlyList<String>)` |  |

