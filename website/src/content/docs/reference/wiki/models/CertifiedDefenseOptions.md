---
title: "CertifiedDefenseOptions<T>"
description: "Configuration options for certified defense mechanisms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for certified defense mechanisms.

## For Beginners

These settings control how the "guaranteed protection" works.
You can adjust how many samples to use, how tight the guarantees should be, and what
certification method to apply.

## How It Works

These options control certified robustness methods that provide provable guarantees
about model predictions under adversarial perturbations.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for certification. |
| `CertificationMethod` | Gets or sets the certification method to use. |
| `ConfidenceLevel` | Gets or sets the confidence level for certification. |
| `NoiseSigma` | Gets or sets the noise standard deviation for randomized smoothing. |
| `NormType` | Gets or sets the norm type for certification. |
| `NumSamples` | Gets or sets the number of samples for randomized smoothing. |
| `RandomSeed` | Gets or sets the random seed for reproducible certification. |
| `UseTightBounds` | Gets or sets whether to use tight bounds computation. |

