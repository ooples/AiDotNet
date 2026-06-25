---
title: "CodeTokenizer"
description: "Code-aware tokenizer that handles programming language constructs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Tokenization.CodeTokenization`

Code-aware tokenizer that handles programming language constructs.
Supports identifier splitting, keyword recognition, and language-specific patterns.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CodeTokenizer(ITokenizer,ProgrammingLanguage,Boolean)` | Creates a new code tokenizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CleanupTokens(List<String>)` | Cleans up tokens and converts them back to code. |
| `Encode(String,EncodingOptions)` | Encodes code into a tokenization result with best-effort character offsets. |
| `GetLanguageKeywords(ProgrammingLanguage)` | Gets keywords for a programming language. |
| `IsIdentifier(String)` | Checks if a token is an identifier. |
| `PreTokenizeCode(String)` | Pre-tokenizes code by splitting on whitespace and operators while preserving strings and comments. |
| `SplitIdentifier(String)` | Splits an identifier by camelCase, PascalCase, or snake_case. |
| `Tokenize(String)` | Tokenizes code with language-aware handling. |

