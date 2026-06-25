---
title: "AllowedTokenSetConstraint"
description: "A constraint that always restricts generation to a fixed set of token ids, regardless of context — for example, \"only emit digit tokens\" or \"only emit tokens from this closed label vocabulary\"."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Local`

A constraint that always restricts generation to a fixed set of token ids, regardless of context — for
example, "only emit digit tokens" or "only emit tokens from this closed label vocabulary".

## For Beginners

The simplest gate: a permanent allow-list. Whatever has been generated, only
these tokens are ever permitted next. Handy when the whole answer must come from a small known set.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AllowedTokenSetConstraint(IEnumerable<Int32>)` | Initializes a new constraint permitting only the given token ids. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AllowedNextTokens(IReadOnlyList<Int32>)` |  |

