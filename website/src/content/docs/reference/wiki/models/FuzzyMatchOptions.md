---
title: "FuzzyMatchOptions"
description: "Configuration options for fuzzy entity matching in Private Set Intersection."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for fuzzy entity matching in Private Set Intersection.

## For Beginners

When entity IDs across parties aren't perfectly identical
(e.g., "John Smith" vs "Jon Smith", or "123-45-6789" vs "123456789"), fuzzy matching
finds approximate matches. These options control how approximate the matching can be.

## How It Works

Example configuration for matching patient names across hospitals:

## Properties

| Property | Summary |
|:-----|:--------|
| `CaseSensitive` | Gets or sets whether string comparisons are case-sensitive. |
| `NGramSize` | Gets or sets the n-gram size for NGram matching strategy. |
| `NormalizeWhitespace` | Gets or sets whether to normalize whitespace before matching. |
| `Strategy` | Gets or sets the fuzzy matching strategy to use. |
| `Threshold` | Gets or sets the similarity threshold for fuzzy matching. |

