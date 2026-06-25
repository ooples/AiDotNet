---
title: "ShortTimeObjectiveIntelligibility<T>"
description: "Short-Time Objective Intelligibility (STOI) metric for speech intelligibility assessment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Short-Time Objective Intelligibility (STOI) metric for speech intelligibility assessment.

## How It Works

STOI predicts the intelligibility of degraded speech signals. It correlates well with
human listening tests and is commonly used to evaluate speech enhancement and separation.

Values range from 0 to 1, where higher values indicate better intelligibility.

- >0.9: Excellent intelligibility
- 0.7-0.9: Good intelligibility
- 0.5-0.7: Fair intelligibility
- <0.5: Poor intelligibility

Based on "A short-time objective intelligibility measure for time-frequency weighted noisy speech"
by Taal et al. (2011).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShortTimeObjectiveIntelligibility(Int32,Int32,Int32,Int32)` | Initializes a new instance of STOI calculator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes STOI between degraded and clean speech signals. |
| `ComputeBandEnergy(Tensor<>,Int32,Int32,Double,Double)` | Computes weighted energy approximation for a frequency band. |
| `ComputeNormalizedCorrelation([0:,0:],[0:,0:],Int32,Int32,Int32)` | Computes normalized correlation between two envelope sequences. |
| `ComputeSpectralEnvelopes(Tensor<>)` | Computes spectral envelopes using one-third octave band analysis. |
| `GetOneThirdOctaveCenterFrequencies` | Gets center frequencies for one-third octave bands. |

