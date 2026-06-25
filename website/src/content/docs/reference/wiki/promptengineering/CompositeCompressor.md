---
title: "CompositeCompressor"
description: "Compressor that chains multiple compressors together in sequence."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Compression`

Compressor that chains multiple compressors together in sequence.

## For Beginners

Combines multiple compressors for better results.

Example:

Order matters:

- Put rule-based compressors first
- Put aggressive compressors last
- Each compressor should work well with the output of the previous one

## How It Works

This compressor applies multiple compression strategies in sequence, with each
compressor working on the output of the previous one. This allows combining
the strengths of different compression approaches.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CompositeCompressor(IEnumerable<IPromptCompressor>,Func<String,Int32>)` | Initializes a new instance of the CompositeCompressor class. |
| `CompositeCompressor(IPromptCompressor[])` | Initializes a new instance of the CompositeCompressor class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Compressors` | Gets the list of compressors in this composite. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildName(IEnumerable<IPromptCompressor>)` | Builds the name of the composite compressor from its components. |
| `CompressAsync(String,CompressionOptions,CancellationToken)` | Compresses the prompt asynchronously by applying all compressors in sequence. |
| `CompressCore(String,CompressionOptions)` | Compresses the prompt by applying all compressors in sequence. |
| `CreateAggressivePipeline` | Creates a composite compressor with an aggressive pipeline for maximum compression. |
| `CreateStandardPipeline` | Creates a composite compressor with a standard pipeline for general use. |

