---
title: "DeepfakeDetectorConfig"
description: "Configuration for deepfake and AI-generated image detection modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety.Image`

Configuration for deepfake and AI-generated image detection modules.

## For Beginners

Use this to configure how the deepfake detector works.
You can set the detection threshold and choose which analysis methods to enable.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConsistencyAnalysis` | Whether to use consistency analysis. |
| `FrequencyAnalysis` | Whether to use frequency domain analysis. |
| `ProvenanceAnalysis` | Whether to use provenance/metadata analysis. |
| `Threshold` | Detection threshold (0.0-1.0). |

