---
title: "FuzzyMatchStrategy"
description: "Specifies the similarity strategy used for fuzzy entity matching in PSI."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the similarity strategy used for fuzzy entity matching in PSI.

## For Beginners

When entity IDs aren't perfectly identical across parties
(e.g., "John Smith" vs "Jon Smith"), fuzzy matching finds approximate matches.
Each strategy uses a different notion of "similarity":

## Fields

| Field | Summary |
|:-----|:--------|
| `EditDistance` | Levenshtein edit distance. |
| `Exact` | Exact string equality. |
| `Jaccard` | Jaccard similarity coefficient between token or character sets. |
| `NGram` | Character n-gram similarity. |
| `Phonetic` | Phonetic matching using Soundex or Double Metaphone algorithms. |

