---
title: "FedPACPersonalization<T>"
description: "Implements FedPAC (Personalization via Aggregation and Calibration) with prototype alignment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Personalization`

Implements FedPAC (Personalization via Aggregation and Calibration) with prototype alignment.

## For Beginners

FedPAC personalizes in two steps. First, it calibrates the
aggregation itself — instead of averaging all clients equally, each client aggregates only
from "similar" clients (measured by prototype similarity). Second, it aligns class prototypes
(average feature vectors per class) across clients so that the shared feature space has
consistent semantics. This is especially effective when clients have different class
distributions (label skew).

## How It Works

Algorithm:

Reference: FedPAC: Personalization via Aggregation and Calibration (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedPACPersonalization(Double,Double)` | Creates a new FedPAC personalization strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CalibrationWeight` | Gets the calibration loss weight. |
| `SimilarityThreshold` | Gets the similarity threshold for client inclusion. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCalibrationLoss(Dictionary<Int32,[]>,Dictionary<Int32,[]>)` | Computes the calibration loss that aligns local features to global prototypes. |
| `ComputeClassPrototypes([][],Int32[])` | Computes class prototypes from a client's local data: p_c = mean(features where label = c). |
| `ComputeGlobalPrototypes(Dictionary<Int32,Dictionary<Int32,[]>>,Dictionary<Int32,Dictionary<Int32,Int32>>)` | Computes global prototypes by averaging client prototypes per class (weighted by sample count). |
| `ComputePersonalizedWeights(Int32)` | Computes personalized aggregation weights based on prototype similarity. |
| `RegisterPrototypes(Int32,Dictionary<Int32,[]>)` | Registers class prototypes for a client. |

