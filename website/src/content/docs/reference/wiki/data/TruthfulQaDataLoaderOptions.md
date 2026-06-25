---
title: "TruthfulQaDataLoaderOptions"
description: "Configuration for the TruthfulQA benchmark loader (Lin et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration for the TruthfulQA benchmark loader (Lin et al. 2022).

## How It Works

TruthfulQA tests whether language models avoid common false beliefs in
38 categories — 817 questions, each with a best correct answer and
multiple correct/incorrect distractors. This loader exposes the
generation-style version (best_answer as target).

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

