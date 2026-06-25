---
title: "NERModelVariant"
description: "Defines common model size variants for Named Entity Recognition (NER) models."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines common model size variants for Named Entity Recognition (NER) models.

## For Beginners

Think of these like clothing sizes. "Tiny" is the fastest but least
accurate, "Base" is the recommended starting point, and "Large"/"XLarge" are for when
you need maximum accuracy and have powerful hardware.

## How It Works

NER models typically come in multiple sizes trading off speed vs accuracy:

- Smaller variants (Tiny, Small) run faster with lower memory, suitable for real-time applications
- Larger variants (Large, XLarge) produce higher accuracy but require more compute
- Base is the default recommended configuration balancing speed and accuracy

## Fields

| Field | Summary |
|:-----|:--------|
| `Base` | Base variant: default recommended configuration. |
| `Large` | Large variant: increased capacity for higher accuracy. |
| `Small` | Small variant: reduced size for faster inference. |
| `Tiny` | Tiny variant: minimal size for maximum speed. |
| `XLarge` | Extra-large variant: maximum capacity. |

