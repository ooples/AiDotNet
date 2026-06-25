---
title: "KernelType<T>"
description: "The type of string kernel to use."
section: "API Reference"
---

`Enums` · `AiDotNet.Kernels`

The type of string kernel to use.

## Fields

| Field | Summary |
|:-----|:--------|
| `BagOfWords` | Bag of words kernel: treats strings as bags of words. |
| `EditDistance` | Normalized edit distance kernel: exp(-d(s,t)/scale). |
| `Spectrum` | Spectrum kernel: counts shared k-mers (substrings of length k). |
| `Subsequence` | Subsequence kernel: counts shared subsequences with gap penalty. |

