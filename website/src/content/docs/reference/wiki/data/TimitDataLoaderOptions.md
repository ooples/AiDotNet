---
title: "TimitDataLoaderOptions"
description: "Configuration for the TIMIT acoustic-phonetic continuous-speech corpus loader (Garofolo et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration for the TIMIT acoustic-phonetic continuous-speech corpus loader (Garofolo et al. 1993).

## How It Works

TIMIT — 6,300 sentences from 630 speakers (8 dialect regions × ~80 speakers).
The classic phoneme-recognition benchmark; widely used in early speech-recognition
research. Contains sphere-format WAV + word/phoneme alignment files.

**Commercial license required** — TIMIT is distributed by LDC (catalog
LDC93S1) and requires a paid LDC membership. `AutoDownload`
is unavailable; this loader expects the user to manually extract the
distribution.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Always false — TIMIT requires LDC membership. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

