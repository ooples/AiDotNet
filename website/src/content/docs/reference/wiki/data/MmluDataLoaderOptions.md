---
title: "MmluDataLoaderOptions"
description: "Configuration options for the MMLU (Massive Multitask Language Understanding) loader (Hendrycks et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the MMLU (Massive Multitask Language Understanding) loader (Hendrycks et al. 2021).

## How It Works

MMLU — 57 subjects × ~270 multi-choice questions each, spanning STEM,
humanities, social sciences, professional, and other categories. The
canonical broad-knowledge LLM eval benchmark since GPT-3. 4-way
multiple choice. Splits: dev (5/subject, used for few-shot prompting),
val (~85/subject), test (~14k total).

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxQuestionLength` | Maximum encoded question + choice length in tokens. |
| `MaxSamples` | Optional maximum number of samples to load (for fast iteration / smoke testing). |
| `Split` | Dataset split. |
| `SubjectFilter` | Optional subject filter (case-insensitive substring). |
| `VocabularySize` | Maximum vocabulary size for the BPE-style tokenizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

