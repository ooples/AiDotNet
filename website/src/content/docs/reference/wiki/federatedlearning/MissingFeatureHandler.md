---
title: "MissingFeatureHandler<T>"
description: "Handles missing features in vertical FL when not all parties have data for all entities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Vertical`

Handles missing features in vertical FL when not all parties have data for all entities.

## For Beginners

In real-world VFL, not all parties have records for all entities.
A bank might have 100,000 customers while the partner hospital only has 30,000 patients,
with only 20,000 in common. For the other 80,000 bank customers, the hospital's features
are "missing".

## How It Works

This class provides strategies for handling those missing features:

**Reference:** Based on "Vertical Federated Learning with Missing Features During
Training and Inference" (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MissingFeatureHandler(MissingFeatureOptions)` | Initializes a new instance of `MissingFeatureHandler`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Strategy` | Gets the configured missing feature strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateMissingnessIndicator(Int32,Int32,IReadOnlyCollection<Int32>)` | Creates a missingness indicator tensor for a batch, marking which features are imputed. |
| `ImputeEmbeddings(String,Int32,Int32)` | Imputes missing embeddings for a party that doesn't have data for the current batch entities. |
| `ShouldIncludeEntity(String,IReadOnlyCollection<String>,Int32)` | Determines whether an entity should be included in training based on its availability across parties. |
| `UpdateStatistics(String,Tensor<>)` | Updates the running mean statistics for a party's embeddings. |

