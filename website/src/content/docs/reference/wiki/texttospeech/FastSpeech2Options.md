---
title: "FastSpeech2Options"
description: "Options for FastSpeech 2 (variance adaptor with pitch, energy, and duration predictors)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.TextToSpeech.Classic`

Options for FastSpeech 2 (variance adaptor with pitch, energy, and duration predictors).

## For Beginners

These options configure the FastSpeech2 model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FastSpeech2Options(FastSpeech2Options)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumEnergyBins` | Gets or sets the number of energy bins for quantization. |
| `NumPitchBins` | Gets or sets the number of pitch bins for quantization. |
| `UseEnergyPredictor` | Gets or sets whether to use energy prediction. |
| `UsePitchPredictor` | Gets or sets whether to use pitch prediction. |
| `VariancePredictorFilterSize` | Gets or sets the variance predictor filter size. |
| `VariancePredictorKernelSize` | Gets or sets the variance predictor kernel size. |

