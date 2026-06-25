---
title: "PoolingStrategy"
description: "Specifies the global feature pooling strategy for vision encoders."
section: "API Reference"
---

`Enums` · `AiDotNet.VisionLanguage.Encoders`

Specifies the global feature pooling strategy for vision encoders.

## Fields

| Field | Summary |
|:-----|:--------|
| `ClsPlusMean` | Concatenate CLS and mean-pooled features. |
| `ClsToken` | Use the [CLS] token output as the global feature. |
| `LastToken` | Use the last token output. |
| `MeanPool` | Average pool all patch token outputs. |

