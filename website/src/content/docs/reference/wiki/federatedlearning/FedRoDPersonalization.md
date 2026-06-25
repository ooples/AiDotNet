---
title: "FedRoDPersonalization<T>"
description: "Implements FedRoD (Representation on Demand) personalization with dual classifiers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Personalization`

Implements FedRoD (Representation on Demand) personalization with dual classifiers.

## For Beginners

FedRoD trains two classification heads on top of a shared
feature extractor: a generic head (aggregated globally, works well on average) and a
personalized head (kept locally, works well on this client's data). At inference time,
the client can choose to use either head or combine their predictions depending on
whether the input is more like global data or local data.

## How It Works

Architecture:

Reference: Chen, H.-Y. & Chao, W.-L. (2023). "On Bridging Generic and Personalized
Federated Learning for Image Classification." ICLR 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedRoDPersonalization(Double,Double)` | Creates a new FedRoD personalization strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadFraction` | Gets the head fraction. |
| `MixingAlpha` | Gets the mixing weight for generic head. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CombinePredictions([],[])` | Combines generic and personalized predictions. |
| `ComputeBalancedSoftmaxLogits([],Int32[])` | Computes the balanced softmax loss, which adjusts for class frequency imbalance. |
| `ExtractGenericHead(Dictionary<String,[]>)` | Extracts the generic head parameters (aggregated with the body). |
| `ExtractPersonalizedHead(Dictionary<String,[]>)` | Extracts the personalized head parameters (kept local). |
| `ExtractSharedParameters(Dictionary<String,[]>)` | Extracts the shared body + generic head (to be aggregated). |

