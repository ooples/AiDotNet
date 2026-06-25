---
title: "HuggingFaceTokenizerLoader"
description: "Loads HuggingFace pretrained tokenizers."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Tokenization.HuggingFace`

Loads HuggingFace pretrained tokenizers.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSafePath(String,String)` | Safely combines a base path with a filename, preventing path traversal attacks. |
| `LoadBpeTokenizer(String,String,SpecialTokens)` | Loads a BPE tokenizer from HuggingFace format. |
| `LoadFromDirectory(String)` | Loads a HuggingFace tokenizer from a directory. |
| `LoadFromHub(String,String)` | Loads a tokenizer from HuggingFace Hub by model name. |
| `LoadFromHubAsync(String,String)` | Asynchronously loads a tokenizer from HuggingFace Hub. |
| `LoadFromTokenizerJson(String)` | Loads a tokenizer from a tokenizer.json file. |
| `LoadSentencePieceTokenizer(String,SpecialTokens)` | Loads a SentencePiece tokenizer from HuggingFace format. |
| `LoadWordPieceTokenizer(String,SpecialTokens)` | Loads a WordPiece tokenizer from HuggingFace format. |
| `SaveToDirectory(ITokenizer,String)` | Saves a tokenizer to HuggingFace format. |

