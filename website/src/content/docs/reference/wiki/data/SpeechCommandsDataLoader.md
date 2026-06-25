---
title: "SpeechCommandsDataLoader<T>"
description: "Loads the Google Speech Commands v2 dataset (~65K clips, 35 words, 16kHz)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Audio.Benchmarks`

Loads the Google Speech Commands v2 dataset (~65K clips, 35 words, 16kHz).

## How It Works

Speech Commands v2 expects the following directory structure after extraction:

Each subdirectory is a word class. WAV files are 16-bit PCM mono at 16kHz, ~1 second each.

**Streaming behaviour:** the loader does NOT preload audio waveforms into memory.
File paths and class indices are scanned eagerly during `LoadAsync`, but the
actual WAV decoding (and resampling, if configured) happens per-batch in
`Int32[])`. This is intentional: the full 35-class dataset would
allocate ~4 GB of float32 features in memory, so eager loading was guaranteed to OOM.
Use `GetBatches` / `GetBatchesAsync` for training; direct
`Features` access on this loader is intentionally not supported (features
are decoded lazily per batch). `Labels` is a small `[N, numClasses]`
one-hot tensor and is materialized during `LoadAsync`, so it remains available.

**Class scheme:** the "core" 12-class subset (default) follows Warden 2018:
the 10 keyword classes (yes, no, up, down, left, right, on, off, stop, go) plus
`_silence_` (sampled from `_background_noise_/`) and `_unknown_`
(collapses every non-core word directory). The full 35-class mode uses all word
directories with no synthetic classes.

**Auto-download:** when `AutoDownload`
is true (default), the loader fetches and extracts the tarball from
`DownloadUrl` if the data directory is empty, mirroring
`LibriSpeechDataLoader`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpeechCommandsDataLoader(SpeechCommandsDataLoaderOptions)` | Creates a new Speech Commands data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AllWords` | All 35 word classes in the full dataset. |
| `CoreWords` | Core 10 spoken words (the standard benchmark subset). |
| `Description` |  |
| `FeatureCount` |  |
| `Name` |  |
| `NumClasses` | Number of word classes (12 in core mode, 35 in full mode). |
| `OutputDimension` |  |
| `TotalCount` |  |
| `WordList` | Gets the word list being used (10/35 word names plus the two synthetic labels in core mode). |

## Methods

| Method | Summary |
|:-----|:--------|
| `DecodeBytesWithResample(Byte[],[],Int32,Int32,Int32,Int32)` | Decodes a WAV byte payload, then writes `targetLen` samples into `dst` starting at `dstOffset`, resampling from `nativeRate` to `targetRate` via linear interpolation when the rates differ. |
| `DecodeSampleInto(SpeechCommandsDataLoader<>.SampleEntry,[],Int32,Int32)` | Decodes a single sample (real WAV or synthesized silence) into the destination span at the given offset, applying resampling when the configured target rate differs from the dataset's native 16kHz rate. |
| `EnsureDatasetPresentAsync(CancellationToken)` | Triggers `CancellationToken)` when the data directory is empty AND `AutoDownload` is enabled. |
| `ExtractBatch(Int32[])` |  |
| `GetOrLoadBackgroundNoiseBuffer(String)` | Returns a cached fully-decoded waveform for the given background-noise file, loading and caching it on first access. |
| `IsInRequestedSplit(String,String,HashSet<String>,HashSet<String>)` | Decides whether a given WAV path falls into the requested split based on the official testing_list.txt / validation_list.txt files (Warden 2018 spec). |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `ResampleIntoDestination([],Int32,[],Int32,Int32,Int32,Int32)` | Linear-interpolation resample of `nativeBuf` starting at `nativeOffset` into `dst`[`dstOffset`..]. |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `BackgroundNoiseDir` | Subdirectory containing the long background-noise WAVs distributed with the dataset. |
| `DownloadUrl` | HTTPS URL of the Speech Commands v2 archive. |
| `SilenceLabel` | Synthetic class label for background-noise / silence in the 12-class subset. |
| `UnknownLabel` | Synthetic class label for non-keyword speech in the 12-class subset. |

