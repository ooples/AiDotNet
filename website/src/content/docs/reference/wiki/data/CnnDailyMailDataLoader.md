---
title: "CnnDailyMailDataLoader<T>"
description: "Loads the CNN/DailyMail abstractive-summarization dataset v3.0.0 (Hermann et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the CNN/DailyMail abstractive-summarization dataset v3.0.0
(Hermann et al. 2015 / See et al. 2017) via HuggingFace parquet.

## How It Works

Expects parquet shards under `{DataPath}/cnn_dailymail/`. Auto-download
fetches a small sentinel set: validation parquet + a single train shard.
For the full 287k-train corpus, manually download all shards from
huggingface.co/datasets/abisee/cnn_dailymail and place them in DataPath.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

