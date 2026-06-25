---
title: "MiniRocketOptions<T>"
description: "Configuration options for the MiniRocket time series classifier."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the MiniRocket time series classifier.

## For Beginners

MiniRocket is an optimized version of ROCKET that uses
deterministic kernels instead of random ones. This makes it faster and more reproducible
while maintaining similar accuracy.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumBiasesPerDilation` | Gets or sets the number of bias values to use per dilation. |
| `NumFeatures` | Gets or sets the number of features to extract. |
| `RandomSeed` | Gets or sets the random seed for reproducible results. |
| `UseBias` | Gets or sets whether to use bias terms in the kernel computation. |

