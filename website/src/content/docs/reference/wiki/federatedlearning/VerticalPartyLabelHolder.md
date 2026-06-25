---
title: "VerticalPartyLabelHolder<T>"
description: "Represents the label-holding party in vertical federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Vertical`

Represents the label-holding party in vertical federated learning.

## For Beginners

In VFL, one special party holds the labels (prediction targets).
For example, a hospital knows patient outcomes while partner banks only have financial data.
The label holder plays a unique role:

## How It Works

**Security note:** The label holder's gradients can reveal label information.
For example, large gradient magnitudes suggest the model made a large error, which hints
at the true label. Always use label protection in production deployments.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VerticalPartyLabelHolder(String,Tensor<>,Tensor<>,IReadOnlyList<String>,Int32,Nullable<Int32>)` | Initializes a new instance of `VerticalPartyLabelHolder`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` |  |
| `FeatureCount` |  |
| `IsLabelHolder` |  |
| `PartyId` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyBackward(Tensor<>,Double)` |  |
| `ComputeForward(IReadOnlyList<Int32>)` |  |
| `ComputeLoss(Tensor<>,Tensor<>)` | Computes the mean squared error loss and its gradient. |
| `GetEntityIds` |  |
| `GetLabels(IReadOnlyList<Int32>)` | Gets the labels for the specified aligned entity indices. |
| `GetParameters` |  |
| `SetParameters(IReadOnlyList<Tensor<>>)` |  |

