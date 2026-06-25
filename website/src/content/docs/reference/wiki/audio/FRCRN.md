---
title: "FRCRN<T>"
description: "FRCRN (Frequency Recurrence CRN) speech enhancement model (Zhao et al., ICASSP 2022)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Enhancement`

FRCRN (Frequency Recurrence CRN) speech enhancement model (Zhao et al., ICASSP 2022).

## For Beginners

FRCRN processes audio frequencies in sequence (low to high),
so each frequency "knows" about its neighbors. This helps it tell speech from noise
because speech frequencies appear in related patterns, while noise is more random.

**Usage:**

## How It Works

FRCRN (Alibaba DAMO Academy) uses frequency recurrence to model spectral correlations
and complex spectral mapping. It won 1st place in the ICASSP 2022 DNS Challenge
non-personalized track with superior noise suppression while preserving speech quality.

## Properties

| Property | Summary |
|:-----|:--------|
| `EnhancementStrength` |  |
| `LatencySamples` |  |
| `NumChannels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Enhance(Tensor<>)` |  |
| `EnhanceAsync(Tensor<>,CancellationToken)` |  |
| `EnhanceWithReference(Tensor<>,Tensor<>)` |  |
| `EstimateNoiseProfile(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |

