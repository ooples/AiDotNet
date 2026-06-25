---
title: "PromptCompressorBase"
description: "Provides a base implementation for prompt compressors with common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.PromptEngineering.Compression`

Provides a base implementation for prompt compressors with common functionality.

## For Beginners

This is the foundation that all prompt compressors build upon.

Think of it like a template for making prompts shorter:

- It handles common tasks (counting tokens, checking inputs)
- Specific compression methods fill in how they compress
- This ensures all compressors work consistently

Derived classes just need to implement CompressCore to define their compression logic.

## How It Works

This abstract class implements the IPromptCompressor interface and provides common functionality
for prompt compression strategies. It handles token counting, validation, and delegates to
derived classes for the core compression logic.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PromptCompressorBase(String,Func<String,Int32>)` | Initializes a new instance of the PromptCompressorBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this compressor implementation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compress(String,CompressionOptions)` | Compresses a prompt to reduce its token count. |
| `CompressAsync(String,CompressionOptions,CancellationToken)` | Compresses a prompt asynchronously. |
| `CompressCore(String,CompressionOptions)` | Core compression logic to be implemented by derived classes. |
| `CompressWithMetrics(String,CompressionOptions)` | Compresses a prompt and returns detailed metrics about the compression. |
| `CountTokens(String)` | Counts tokens in the given text. |
| `ExtractCodeBlocks(String)` | Extracts and preserves code blocks from a prompt. |
| `ExtractVariables(String)` | Extracts and preserves variables from a prompt (like {variable_name}). |
| `ReplaceCodeBlocksWithPlaceholders(String,Dictionary<String,String>)` | Replaces code blocks with placeholders to protect them during compression. |
| `ReplaceVariablesWithPlaceholders(String,Dictionary<String,String>)` | Replaces variables with placeholders to protect them during compression. |
| `RestoreCodeBlocks(String,Dictionary<String,String>)` | Restores code blocks from placeholders after compression. |
| `RestoreVariables(String,Dictionary<String,String>)` | Restores variables from placeholders after compression. |
| `ValidatePrompt(String)` | Validates the prompt input. |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Regex timeout to prevent ReDoS attacks. |

