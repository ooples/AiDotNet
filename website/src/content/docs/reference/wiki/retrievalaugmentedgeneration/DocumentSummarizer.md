---
title: "DocumentSummarizer<T>"
description: "Document summarizer for creating concise summaries of retrieved content."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.ContextCompression`

Document summarizer for creating concise summaries of retrieved content.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DocumentSummarizer(INumericOperations<>,Int32,String,String)` | Initializes a new instance of the `DocumentSummarizer` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompressCore(List<Document<>>,String,Dictionary<String,Object>)` | Compresses documents by summarizing their content with query-aware sentence selection. |
| `Summarize(List<Document<>>)` | Summarizes a list of documents. |
| `SummarizeText(String,List<String>)` | Summarizes text to a maximum length with query-aware sentence selection. |

