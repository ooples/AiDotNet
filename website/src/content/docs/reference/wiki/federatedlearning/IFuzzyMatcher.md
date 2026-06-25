---
title: "IFuzzyMatcher"
description: "Defines the interface for approximate entity matching in PSI."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.PSI`

Defines the interface for approximate entity matching in PSI.

## For Beginners

In an ideal world, "Patient #12345" at Hospital A is exactly
the same string at Hospital B. In practice, one might store "John Smith" and the other
"Jon Smith" or "SMITH, JOHN". Fuzzy matching bridges these gaps by finding IDs that
are similar enough to be the same entity.

## How It Works

Fuzzy matchers handle the common real-world scenario where entity identifiers
aren't perfectly identical across parties due to typos, formatting differences,
transliteration, or data entry errors.

## Properties

| Property | Summary |
|:-----|:--------|
| `StrategyName` | Gets the name of this fuzzy matching strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSimilarity(String,String,FuzzyMatchOptions)` | Computes the similarity score between two identifiers. |
| `FindMatches(String,IReadOnlyList<String>,FuzzyMatchOptions)` | Finds all matches for a given identifier from a set of candidates. |
| `IsMatch(String,String,FuzzyMatchOptions)` | Determines whether two identifiers are a match according to the configured threshold. |
| `Normalize(String,FuzzyMatchOptions)` | Normalizes an identifier before comparison by applying case folding, whitespace normalization, and other transformations. |

