---
title: "BinningStrategy"
description: "Specifies the strategy for defining bin widths."
section: "API Reference"
---

`Enums` · `AiDotNet.Preprocessing.Discretizers`

Specifies the strategy for defining bin widths.

## Fields

| Field | Summary |
|:-----|:--------|
| `KMeans` | Uses K-means clustering to determine bin edges. |
| `Quantile` | Each bin contains approximately the same number of samples (quantile-based binning). |
| `Uniform` | All bins have the same width (equal-width binning). |

