---
title: "CnnDailyMailDataLoaderOptions"
description: "Configuration options for the CNN/DailyMail summarization loader (Hermann et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the CNN/DailyMail summarization loader (Hermann et al. 2015 / See et al. 2017).

## How It Works

CNN/DailyMail v3.0.0 — 287k train / 13.4k val / 11.5k test article-summary
pairs. The canonical English news abstractive-summarization benchmark.
AutoDownload pulls the HuggingFace parquet shards (abisee/cnn_dailymail).

