---
title: "Enwik8DataLoaderOptions"
description: "Configuration options for the enwik8 character-level LM data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the enwik8 character-level LM data loader.

## How It Works

enwik8 is the standard character-level Wikipedia language modeling benchmark
(Hutter Prize): the first 100M bytes of an English Wikipedia XML dump.
Models are evaluated in bits-per-character (BPC); SOTA values typically
sit in the 0.95–1.10 range. Canonical split: first 90M chars train,
next 5M val, last 5M test.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `SequenceLength` | Sequence length (characters per sample). |
| `Split` | Dataset split to load. |
| `VocabularySize` | Vocabulary size cap. |

