---
title: "Mixed Precision"
description: "All 10 public types in the AiDotNet.mixedprecision namespace, organized by kind."
section: "API Reference"
---

**10** public types in this namespace, organized by kind.

## Models & Types (5)

| Type | Summary |
|:-----|:--------|
| [`LayerPrecisionPolicy`](/docs/reference/wiki/mixedprecision/layerprecisionpolicy/) | Defines precision policies for different layer types during mixed-precision training. |
| [`LossScaler<T>`](/docs/reference/wiki/mixedprecision/lossscaler/) | Implements dynamic loss scaling for mixed-precision training to prevent gradient underflow. |
| [`MixedPrecisionContext`](/docs/reference/wiki/mixedprecision/mixedprecisioncontext/) | Manages master weights (FP32) and working weights (FP16) for mixed-precision training. |
| [`MixedPrecisionScope`](/docs/reference/wiki/mixedprecision/mixedprecisionscope/) | Provides an ambient context for mixed-precision operations during forward and backward passes. |
| [`MixedPrecisionTrainingLoop<T>`](/docs/reference/wiki/mixedprecision/mixedprecisiontrainingloop/) | Implements mixed-precision training loop for neural networks following NVIDIA's approach. |

## Structs (2)

| Type | Summary |
|:-----|:--------|
| [`Float8E4M3`](/docs/reference/wiki/mixedprecision/float8e4m3/) | Represents an 8-bit floating point number in E4M3 format (4 exponent bits, 3 mantissa bits). |
| [`Float8E5M2`](/docs/reference/wiki/mixedprecision/float8e5m2/) | Represents an 8-bit floating point number in E5M2 format (5 exponent bits, 2 mantissa bits). |

## Options & Configuration (1)

| Type | Summary |
|:-----|:--------|
| [`MixedPrecisionConfig`](/docs/reference/wiki/mixedprecision/mixedprecisionconfig/) | Configuration settings for mixed-precision training. |

## Helpers & Utilities (2)

| Type | Summary |
|:-----|:--------|
| [`Float8Extensions`](/docs/reference/wiki/mixedprecision/float8extensions/) | Provides utility methods for working with FP8 types. |
| [`LayerPrecisionPolicyExtensions`](/docs/reference/wiki/mixedprecision/layerprecisionpolicyextensions/) | Extension methods for applying layer precision policies. |

