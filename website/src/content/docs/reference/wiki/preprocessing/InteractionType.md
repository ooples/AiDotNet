---
title: "InteractionType"
description: "Specifies the type of feature interactions to generate."
section: "API Reference"
---

`Enums` · `AiDotNet.Preprocessing.FeatureGeneration`

Specifies the type of feature interactions to generate.

## Fields

| Field | Summary |
|:-----|:--------|
| `AllPairs` | All ordered pairs (a×b and b×a both included). |
| `Pairwise` | Pairwise interactions only (a×b, a×c, b×c). |
| `WithSelf` | Pairwise plus self-interaction (includes a², b², c²). |

