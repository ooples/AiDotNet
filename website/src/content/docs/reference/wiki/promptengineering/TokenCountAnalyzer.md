---
title: "TokenCountAnalyzer"
description: "Analyzer that provides accurate token counting and cost estimation for prompts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Analysis`

Analyzer that provides accurate token counting and cost estimation for prompts.

## For Beginners

Counts how many tokens your prompt uses and estimates costs.

Example:

Supports different models with their pricing:

- GPT-4: $0.03/1K tokens
- GPT-3.5-Turbo: $0.001/1K tokens
- Claude: $0.008/1K tokens

## How It Works

This analyzer focuses on token counting and cost estimation, supporting various
tokenization methods and model-specific pricing.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TokenCountAnalyzer(String,Func<String,Int32>)` | Initializes a new instance of the TokenCountAnalyzer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForClaude` | Creates an analyzer pre-configured for Claude. |
| `ForGemini` | Creates an analyzer pre-configured for Gemini. |
| `ForGpt35Turbo` | Creates an analyzer pre-configured for GPT-3.5-Turbo. |
| `ForGpt4` | Creates an analyzer pre-configured for GPT-4. |
| `GetModelPrice(String)` | Gets the price per 1000 tokens for a given model. |
| `GetSupportedModels` | Gets the list of supported models. |

