---
title: "HeuristicTextFilter"
description: "Filters text documents using simple heuristic rules for quality assessment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Quality`

Filters text documents using simple heuristic rules for quality assessment.

## How It Works

Applies rule-based checks common in web-crawl data cleaning (C4, CCNet, OSCAR).
Checks word count, character ratios, punctuation, and boilerplate phrases.
Fast and requires no training data.

## Methods

| Method | Summary |
|:-----|:--------|
| `Filter(IReadOnlyList<String>)` | Filters documents, returning indices of documents that should be removed. |
| `PassesFilter(String)` | Evaluates a single document against all heuristic rules. |

