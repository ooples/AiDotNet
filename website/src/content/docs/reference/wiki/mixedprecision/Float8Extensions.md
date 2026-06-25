---
title: "Float8Extensions"
description: "Provides utility methods for working with FP8 types."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.MixedPrecision`

Provides utility methods for working with FP8 types.

## For Beginners

This class provides helper methods to convert between FP8 formats
and standard floating-point types, as well as bulk conversion methods for arrays.

## Methods

| Method | Summary |
|:-----|:--------|
| `ToE4M3(Float8E5M2)` | Converts E5M2 to E4M3 (for weights/activations). |
| `ToE4M3(Single[])` | Converts an array of floats to E4M3 format. |
| `ToE5M2(Float8E4M3)` | Converts E4M3 to E5M2 (for gradients). |
| `ToE5M2(Single[])` | Converts an array of floats to E5M2 format. |
| `ToFloatArray(Float8E4M3[])` | Converts an array of E4M3 values to floats. |
| `ToFloatArray(Float8E5M2[])` | Converts an array of E5M2 values to floats. |

