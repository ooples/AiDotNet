---
title: "SpeculativeMethod"
description: "Selects the speculative decoding method."
section: "API Reference"
---

`Enums` · `AiDotNet.Configuration`

Selects the speculative decoding method.

## Fields

| Field | Summary |
|:-----|:--------|
| `Auto` | Automatically select the best available method (defaults to ClassicDraftModel today). |
| `ClassicDraftModel` | Classic draft-model speculative decoding (standard). |
| `Eagle` | EAGLE-style enhanced draft proposals (hook for future internal implementation). |
| `Medusa` | Medusa-style multi-head proposals (hook for future internal implementation). |

