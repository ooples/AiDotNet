---
title: "XSumDataLoaderOptions"
description: "Configuration options for the XSum extreme summarization loader (Narayan et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the XSum extreme summarization loader (Narayan et al. 2018).

## How It Works

XSum — 226k BBC articles each paired with a single-sentence summary.
The "extreme" abstractive summarization benchmark; targets are highly
compressive (≈ 1:30 compression ratio). Splits: 204k train / 11.3k val / 11.3k test.
AutoDownload pulls a HuggingFace parquet shard.

