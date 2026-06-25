---
title: "HumanEvalDataLoaderOptions"
description: "Configuration options for the HumanEval Python code-generation benchmark (Chen et al."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Text.Benchmarks`

Configuration options for the HumanEval Python code-generation
benchmark (Chen et al. 2021).

## How It Works

HumanEval is a 164-problem hand-curated Python function-completion
benchmark. Each problem provides a function signature + docstring as
the prompt; the model completes the body. Pass@k scores against the
canonical unit tests are the standard metric. Used as the default
code-generation benchmark since GPT-3.5/Codex.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoDownload` | Automatically download if not present. |
| `DataPath` | Root data path. |
| `MaxPromptLength` | Maximum prompt length in tokens. |
| `MaxSamples` | Optional maximum number of samples to load. |
| `MaxSolutionLength` | Maximum solution length in tokens. |
| `Split` | Dataset split to load. |
| `VocabularySize` | Maximum vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

