---
title: "EncodingOptions"
description: "Options for encoding text into tokens."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Tokenization.Models`

Options for encoding text into tokens.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EncodingOptions` | Creates default encoding options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AddSpecialTokens` | Gets or sets whether to add special tokens (e.g., [CLS], [SEP]). |
| `MaxLength` | Gets or sets the maximum sequence length. |
| `Padding` | Gets or sets whether to pad sequences to MaxLength. |
| `PaddingSide` | Gets or sets the padding side ("right" or "left"). |
| `ReturnAttentionMask` | Gets or sets whether to return attention masks. |
| `ReturnOffsets` | Gets or sets whether to return character offsets. |
| `ReturnPositionIds` | Gets or sets whether to return position IDs. |
| `ReturnTokenTypeIds` | Gets or sets whether to return token type IDs. |
| `Stride` | Gets or sets the stride for overflow handling (used when truncating). |
| `Truncation` | Gets or sets whether to truncate sequences that exceed MaxLength. |
| `TruncationSide` | Gets or sets the truncation side ("right" or "left"). |

