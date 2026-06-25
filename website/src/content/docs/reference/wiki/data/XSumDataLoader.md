---
title: "XSumDataLoader<T>"
description: "Loads the XSum extreme abstractive-summarization dataset (Narayan et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text.Benchmarks`

Loads the XSum extreme abstractive-summarization dataset (Narayan et al. 2018).

## How It Works

Mirrors CNN/DailyMail loader shape; reads the HuggingFace
`EdinburghNLP/xsum` parquet conversion. Schema: document, summary, id.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractBatch(Int32[])` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `UnloadDataCore` |  |

