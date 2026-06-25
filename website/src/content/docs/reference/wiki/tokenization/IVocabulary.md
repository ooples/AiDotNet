---
title: "IVocabulary"
description: "Interface for vocabulary management."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Tokenization.Interfaces`

Interface for vocabulary management.

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
| `Clear` | Clears the vocabulary. |
| `ContainsId(Int32)` | Checks if a token ID exists in the vocabulary. |
| `ContainsToken(String)` | Checks if a token exists in the vocabulary. |
| `GetAllTokens` | Gets all tokens in the vocabulary. |
| `GetToken(Int32)` | Gets the token for a given token ID. |
| `GetTokenId(String)` | Gets the token ID for a given token. |

