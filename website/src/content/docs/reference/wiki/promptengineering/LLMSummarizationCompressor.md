---
title: "LLMSummarizationCompressor"
description: "Compressor that uses an LLM to intelligently summarize and compress prompts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Compression`

Compressor that uses an LLM to intelligently summarize and compress prompts.

## For Beginners

Uses AI to make your prompt shorter while keeping the meaning.

Example:

When to use:

- For complex prompts where simple pattern matching won't work
- When semantic understanding is required
- When maximum compression with preserved meaning is needed

## How It Works

This compressor delegates compression to a language model, which can understand context
and produce semantically equivalent but shorter versions of prompts. It's the most
intelligent form of compression but requires an LLM call.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LLMSummarizationCompressor(Func<String,CancellationToken,Task<String>>,String,Func<String,Int32>)` | Initializes a new instance of the LLMSummarizationCompressor class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultSystemPrompt` | Gets the default system prompt used for compression. |
| `SystemPrompt` | Gets the compression prompt template. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildCompressionRequest(String,CompressionOptions)` | Builds the compression request to send to the LLM. |
| `CompressAsync(String,CompressionOptions,CancellationToken)` | Compresses the prompt asynchronously using the LLM. |
| `CompressCore(String,CompressionOptions)` | Compresses the prompt synchronously. |
| `ExtractCompressedPrompt(String)` | Extracts the compressed prompt from the LLM response. |
| `FallbackCompress(String,CompressionOptions)` | Fallback compression using rule-based approach when LLM is not available. |

