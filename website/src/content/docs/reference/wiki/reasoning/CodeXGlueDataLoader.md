---
title: "CodeXGlueDataLoader"
description: "Loads CodeXGLUE-style datasets from JSONL (one JSON object per line)."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Reasoning.Benchmarks.Data`

Loads CodeXGLUE-style datasets from JSONL (one JSON object per line).

## For Beginners

A JSONL file is a text file where each line is one JSON record.
That makes it easy to stream or partially load large datasets.

## How It Works

CodeXGLUE is a suite of code understanding/generation datasets. This loader intentionally does not ship any datasets
with the repository; callers provide a path to a local JSONL file.

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadFromFileAsync(String,String,String,String,String)` | Loads a CodeXGLUE dataset from a JSONL file. |

