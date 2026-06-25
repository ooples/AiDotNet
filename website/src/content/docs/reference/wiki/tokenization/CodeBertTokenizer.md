---
title: "CodeBertTokenizer"
description: "CodeBERT-compatible tokenizer for program synthesis and code understanding tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Tokenization.CodeTokenization`

CodeBERT-compatible tokenizer for program synthesis and code understanding tasks.
Combines WordPiece tokenization with code-aware preprocessing.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CodeBertTokenizer(IVocabulary,ProgrammingLanguage,SpecialTokens)` | Creates a new CodeBERT tokenizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Tokenizer` | Gets the underlying tokenizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Decode(List<Int32>,Boolean)` | Decodes token IDs back to code. |
| `EncodeCodeAndNL(String,String,EncodingOptions)` | Encodes code and natural language for CodeBERT. |

