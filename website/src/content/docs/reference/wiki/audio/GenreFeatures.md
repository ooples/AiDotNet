---
title: "GenreFeatures"
description: "Features extracted for genre classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

Features extracted for genre classification.

## For Beginners

GenreFeatures provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `MfccMean` | Mean of MFCC coefficients across time. |
| `MfccStd` | Standard deviation of MFCC coefficients across time. |
| `RmsEnergy` | RMS energy. |
| `SpectralCentroidMean` | Mean spectral centroid. |
| `SpectralCentroidStd` | Standard deviation of spectral centroid. |
| `SpectralRolloffMean` | Mean spectral rolloff. |
| `Tempo` | Estimated tempo in BPM. |
| `ZeroCrossingRate` | Zero crossing rate. |

