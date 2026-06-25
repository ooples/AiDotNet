---
title: "CharacterErrorRate"
description: "Character Error Rate (CER) metric for speech recognition and OCR evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Character Error Rate (CER) metric for speech recognition and OCR evaluation.

## How It Works

CER is similar to WER but operates at the character level rather than word level.
It's particularly useful for languages without clear word boundaries or for OCR evaluation.

Formula: CER = (Substitutions + Insertions + Deletions) / Number of characters in reference

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(String,String,Boolean)` | Computes the Character Error Rate between a hypothesis and reference. |
| `ComputeBatch(String[],String[],Boolean)` | Computes CER for a batch of hypothesis-reference pairs. |

