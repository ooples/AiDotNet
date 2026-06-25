---
title: "IPromptCompressor"
description: "Defines the contract for compressing prompts to reduce token counts and API costs."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for compressing prompts to reduce token counts and API costs.

## For Beginners

A prompt compressor makes your prompts shorter without losing meaning.

Why compress prompts?

- Save money: Shorter prompts = fewer tokens = lower costs
- Fit limits: Some models have maximum token limits
- Faster: Shorter prompts process faster

Example:

The compressed version might be:
"Analyze this document and summarize the main points: [document text]"

## How It Works

A prompt compressor reduces the length of prompts while preserving their semantic meaning.
This is valuable for reducing API costs, fitting within context windows, and optimizing
performance. Different compression strategies include redundancy removal, summarization,
and caching-based approaches.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this compressor implementation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compress(String,CompressionOptions)` | Compresses a prompt to reduce its token count. |
| `CompressAsync(String,CompressionOptions,CancellationToken)` | Compresses a prompt asynchronously. |
| `CompressWithMetrics(String,CompressionOptions)` | Compresses a prompt and returns detailed metrics about the compression. |

