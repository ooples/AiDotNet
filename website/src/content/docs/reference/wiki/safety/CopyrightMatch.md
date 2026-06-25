---
title: "CopyrightMatch"
description: "A match between generated text and potentially copyrighted content."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

A match between generated text and potentially copyrighted content.

## For Beginners

CopyrightMatch provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `MatchedText` | The matching text segment. |
| `Similarity` | Similarity score (0.0-1.0). |
| `Source` | Source of the potential copyright match, if known. |
| `StartIndex` | Start character offset in the generated text. |

