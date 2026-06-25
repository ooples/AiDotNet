---
title: "TokenizationStrategy"
description: "MIDI tokenization strategies that control how musical notes are converted to tokens."
section: "API Reference"
---

`Enums` · `AiDotNet.Tokenization.Specialized`

MIDI tokenization strategies that control how musical notes are converted to tokens.

## For Beginners

Choose your strategy based on your use case:

- Use REMI for tasks requiring full musical expression (composition, arrangement)
- Use CPWord for models that benefit from smaller vocabularies (faster training)
- Use SimpleNote for melody-focused tasks where dynamics don't matter

## Fields

| Field | Summary |
|:-----|:--------|
| `CPWord` | Compound Word (CPWord): Combines note attributes into single compound tokens. |
| `REMI` | Revamped MIDI (REMI): Position, Bar, Pitch, Velocity, Duration as separate tokens. |
| `SimpleNote` | Simple Note: Basic pitch-duration pairs without velocity or position tracking. |

