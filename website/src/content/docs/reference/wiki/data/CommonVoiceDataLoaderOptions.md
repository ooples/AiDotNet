---
title: "CommonVoiceDataLoaderOptions"
description: "Configuration options for the Mozilla Common Voice data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration options for the Mozilla Common Voice data loader.

## How It Works

Common Voice is a multilingual speech corpus with ~19K+ hours across 100+ languages.
Audio is stored as MP3 files. For use with this loader, pre-convert to WAV format
(e.g., using ffmpeg: `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`).

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `Language` | Language code (e.g., "en", "fr", "de"). |
| `MaxDurationSeconds` | Maximum audio duration in seconds. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `SampleRate` | Sample rate in Hz. |
| `Split` | Dataset split to load. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

