---
title: "Word2VecAggregation"
description: "Specifies how to aggregate word vectors into document vectors."
section: "API Reference"
---

`Enums` · `AiDotNet.Preprocessing.TextVectorizers`

Specifies how to aggregate word vectors into document vectors.

## Fields

| Field | Summary |
|:-----|:--------|
| `Max` | Element-wise maximum across all word vectors. |
| `Mean` | Average of all word vectors in the document. |
| `Sum` | Sum of all word vectors in the document. |
| `TfidfWeighted` | TF-IDF weighted average of word vectors. |

