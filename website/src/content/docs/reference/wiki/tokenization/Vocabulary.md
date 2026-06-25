---
title: "Vocabulary"
description: "Manages a vocabulary of tokens and their IDs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Tokenization.Vocabulary`

Manages a vocabulary of tokens and their IDs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Vocabulary(Dictionary<String,Int32>,String)` | Creates a vocabulary from an existing token-to-ID mapping. |
| `Vocabulary(String)` | Creates a new vocabulary. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IdToToken` | Gets the ID-to-token mapping. |
| `Size` | Gets the vocabulary size. |
| `TokenToId` | Gets the token-to-ID mapping. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddToken(String)` | Adds a token to the vocabulary. |
| `AddTokens(IEnumerable<String>)` | Adds multiple tokens to the vocabulary. |
| `Clear` | Clears the vocabulary and re-adds the unknown token. |
| `ContainsId(Int32)` | Checks if a token ID exists in the vocabulary. |
| `ContainsToken(String)` | Checks if a token exists in the vocabulary. |
| `GetAllTokens` | Gets all tokens in the vocabulary. |
| `GetToken(Int32)` | Gets the token for a given token ID. |
| `GetTokenId(String)` | Gets the token ID for a given token. |

