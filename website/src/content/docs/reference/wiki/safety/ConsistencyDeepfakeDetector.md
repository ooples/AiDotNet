---
title: "ConsistencyDeepfakeDetector<T>"
description: "Detects deepfake/AI-generated images by checking spatial consistency and natural image statistics violations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Image`

Detects deepfake/AI-generated images by checking spatial consistency and natural image
statistics violations.

## For Beginners

Real photos have certain statistical patterns — for example, noise
is usually consistent across the image, and edges follow natural gradients. AI-generated
images often have subtly wrong noise patterns or edges that look slightly "off". This
detector checks for those inconsistencies.

## How It Works

Real images follow natural image statistics (NIS) — predictable relationships between
neighboring pixels, consistent noise patterns, and regular edge profiles. AI-generated
images often violate these statistics: inconsistent noise levels across regions, unnatural
symmetry patterns, and edge artifacts. This detector measures multiple NIS features and
flags images with anomalous combinations.

**References:**

- NACO: Self-supervised natural consistency for face forgery detection (ECCV 2024, arxiv:2407.10550)
- Spatio-temporal consistency exploitation for deepfake detection (2025, arxiv:2502.08216)
- Rich and poor texture analysis for deepfake detection (2023)
- Generalizable deepfake detection (CVPR 2025, arxiv:2508.06248)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConsistencyDeepfakeDetector(Double,Int32)` | Initializes a new consistency-based deepfake detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeColorConsistencyAnomaly(ReadOnlySpan<>,ConsistencyDeepfakeDetector<>.ImageLayout)` | Detects color transition anomalies. |
| `ComputeEdgeAnomaly(ReadOnlySpan<>,ConsistencyDeepfakeDetector<>.ImageLayout)` | Detects unnatural edge profiles. |
| `ComputeNoiseInconsistency(ReadOnlySpan<>,ConsistencyDeepfakeDetector<>.ImageLayout)` | Measures noise level inconsistency across different image regions. |
| `ComputeSymmetryAnomaly(ReadOnlySpan<>,ConsistencyDeepfakeDetector<>.ImageLayout)` | Detects unnatural bilateral symmetry, which is common in AI-generated faces. |
| `EvaluateImage(Tensor<>)` |  |

