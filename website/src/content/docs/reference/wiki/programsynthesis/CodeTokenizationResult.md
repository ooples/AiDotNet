---
title: "CodeTokenizationResult"
description: "Represents the result of code-aware tokenization, including token IDs and token-to-source spans."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ProgramSynthesis.Models`

Represents the result of code-aware tokenization, including token IDs and token-to-source spans.

## How It Works

This type is designed to reuse the existing tokenization stack (ITokenizer) while providing
code-specific structural metadata (spans) that is useful for downstream tasks (search, review,
diagnostics, etc.).

## Properties

| Property | Summary |
|:-----|:--------|
| `AstEdges` | Optional relationships between AST nodes (when enabled). |
| `AstNodes` | Optional AST nodes extracted during tokenization (when enabled and supported for the language). |
| `TokenSpans` | Best-effort mapping from each token to a span in the original source code. |

