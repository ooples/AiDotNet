---
title: "ContextWindowManager"
description: "Manages context window limits for LLM prompts, providing token estimation and text truncation utilities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering`

Manages context window limits for LLM prompts, providing token estimation and text truncation utilities.

## For Beginners

Large Language Models have a maximum number of tokens they can process
at once (the "context window"). This class helps you:

Example:
```cs
// Create a manager with 4096 token limit
var manager = new ContextWindowManager(4096);

// Check if your prompt fits
var prompt = "Your long prompt here...";
if (!manager.FitsInWindow(prompt))
{
// Truncate to fit
prompt = manager.TruncateToFit(prompt);
}

// Or split long text into chunks
var chunks = manager.SplitIntoChunks(longDocument);
```

A token is roughly 4 characters or 0.75 words in English, but varies by language and model.

## How It Works

This class helps manage the token limits of large language models by providing utilities
to estimate token counts, check if text fits within the context window, and truncate
or split text that exceeds the limit.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContextWindowManager(Int32)` | Initializes a new instance of the ContextWindowManager with the specified maximum tokens. |
| `ContextWindowManager(Int32,Func<String,Int32>)` | Initializes a new instance of the ContextWindowManager with a custom token estimator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxTokens` | Gets the maximum number of tokens allowed in the context window. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DefaultTokenEstimator(String)` | Default token estimator that approximates tokens as roughly 1 token per 4 characters. |
| `EstimateTokens(String)` | Estimates the number of tokens in the given text. |
| `FitsInWindow(String,Int32)` | Checks if the given text fits within the context window. |
| `RemainingTokens(String,Int32)` | Calculates the remaining tokens available after accounting for the given text. |
| `SplitIntoChunks(String,Int32)` | Splits the text into chunks that each fit within the context window. |
| `TruncateToFit(String,Int32)` | Truncates the text to fit within the context window. |

