---
title: "WebDatasetOptions"
description: "Configuration options for the WebDataset loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Formats`

Configuration options for the WebDataset loader.

## Properties

| Property | Summary |
|:-----|:--------|
| `IncludeExtensions` | File extensions to include as data fields (e.g., ".jpg", ".txt", ".json"). |
| `MaxSamples` | Optional maximum number of samples to read. |
| `Seed` | Optional random seed for reproducible shuffling. |
| `Shuffle` | Whether to shuffle samples after reading. |
| `ShuffleBufferSize` | Buffer size for shuffle (number of samples to buffer before shuffling). |

