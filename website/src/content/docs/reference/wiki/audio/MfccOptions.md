---
title: "MfccOptions"
description: "Options for MFCC extraction."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Features`

Options for MFCC extraction.

## For Beginners

These options configure the Mfcc model. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MfccOptions` | Initializes a new instance with default values. |
| `MfccOptions(MfccOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AppendDelta` | Gets or sets whether to append delta (velocity) coefficients. |
| `AppendDeltaDelta` | Gets or sets whether to append delta-delta (acceleration) coefficients. |
| `FMax` | Gets or sets the maximum frequency for the mel filterbank. |
| `FMin` | Gets or sets the minimum frequency for the mel filterbank. |
| `IncludeEnergy` | Gets or sets whether to replace the first coefficient with log energy. |
| `NumCoefficients` | Gets or sets the number of MFCC coefficients to compute. |
| `NumMels` | Gets or sets the number of Mel filterbank channels. |

