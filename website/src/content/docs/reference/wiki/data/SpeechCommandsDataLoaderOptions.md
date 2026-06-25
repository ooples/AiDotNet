---
title: "SpeechCommandsDataLoaderOptions"
description: "Configuration options for the Google Speech Commands v2 data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration options for the Google Speech Commands v2 data loader.

## How It Works

Google Speech Commands v2 contains ~65,000 one-second audio clips of 35 spoken words
recorded by thousands of different speakers at 16kHz. The "core" 12-class subset
(yes, no, up, down, left, right, on, off, stop, go, silence, unknown) is the
standard benchmark, with CNN baselines achieving ~95% accuracy.

Dataset: https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz (2.3GB)
Paper: https://arxiv.org/abs/1804.03209
License: Creative Commons BY 4.0

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download and extract the dataset if it isn't present at `DataPath`. |
| `DataPath` | Root data path. |
| `MaxSamplesPerClass` | Optional maximum number of samples to load per class. |
| `SampleRate` | Target audio sample rate in Hz. |
| `SilenceSampleCount` | Train-equivalent number of `_silence_` clips to synthesize from the background-noise directory when the core 12-class subset is in use. |
| `Split` | Dataset split to load. |
| `TargetLength` | Target audio length in samples at `SampleRate`. |
| `UseCoreSubset` | Use the core 12-class subset (yes, no, up, down, left, right, on, off, stop, go, `_silence_`, `_unknown_`) instead of all 35 classes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NativeSampleRate` | Native sample rate of the Speech Commands corpus (16kHz). |

