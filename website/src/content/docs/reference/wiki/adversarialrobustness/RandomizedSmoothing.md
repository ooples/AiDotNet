---
title: "RandomizedSmoothing<T, TInput, TOutput>"
description: "Implements Randomized Smoothing for certified robustness."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.CertifiedRobustness`

Implements Randomized Smoothing for certified robustness.

## For Beginners

Randomized Smoothing is like asking multiple slightly
different versions of a question and taking the majority vote. By adding random noise
to the input many times and seeing what the model predicts each time, we can
mathematically prove that the prediction is robust to small changes.

## How It Works

Randomized Smoothing creates a smoothed classifier by averaging predictions over
Gaussian noise, enabling provable robustness guarantees.

Original paper: "Certified Adversarial Robustness via Randomized Smoothing"
by Cohen et al. (2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomizedSmoothing(CertifiedDefenseOptions<>)` | Initializes a new instance of Randomized Smoothing. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for vectorized operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CertifyBatch([],IFullModel<,,>)` |  |
| `CertifyPrediction(,IFullModel<,,>)` |  |
| `ComputeCertifiedRadius(,IFullModel<,,>)` |  |
| `Deserialize(Byte[])` |  |
| `EvaluateCertifiedAccuracy([],[],IFullModel<,,>,)` |  |
| `GetOptions` |  |
| `LoadModel(String)` |  |
| `Reset` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |

