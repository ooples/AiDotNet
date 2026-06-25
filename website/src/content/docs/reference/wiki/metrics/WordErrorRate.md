---
title: "WordErrorRate"
description: "Word Error Rate (WER) metric for speech recognition evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Word Error Rate (WER) metric for speech recognition evaluation.

## How It Works

WER measures the accuracy of automatic speech recognition (ASR) systems by computing
the edit distance between the predicted transcription and the reference transcription.

Formula: WER = (Substitutions + Insertions + Deletions) / Number of words in reference

Typical WER values:

- <5%: Excellent (human-level performance)
- 5-10%: Very good
- 10-20%: Good
- 20-30%: Acceptable
- >30%: Poor

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(String,String)` | Computes the Word Error Rate between a hypothesis and reference transcription. |
| `ComputeBatch(String[],String[])` | Computes WER for a batch of hypothesis-reference pairs. |
| `ComputeDetailed(String,String)` | Computes detailed error statistics including substitutions, insertions, and deletions. |
| `ComputeEditOperations(String[],String[])` | Computes the minimum edit operations using dynamic programming (Levenshtein distance variant). |
| `TokenizeWords(String)` | Tokenizes a string into words. |

