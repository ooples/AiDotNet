---
title: "GtzanDataLoaderOptions"
description: "Configuration options for the GTZAN music genre classification loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Audio.Benchmarks`

Configuration options for the GTZAN music genre classification loader.

## How It Works

GTZAN (Tzanetakis & Cook 2002) — 10 genres × 100 30-second clips each
= 1,000 mono WAV files at 22,050 Hz. The canonical music genre
classification benchmark, despite known label noise. Useful for
MIR research entry-level evaluation.

## Properties

| Property | Summary |
|:-----|:--------|
| `SampleRate` | Sample rate for resampling. |
| `Samples` | Number of samples per clip. |
| `TrainFraction` | Train/test split fraction (per-class deterministic). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

