---
title: "PIIResult"
description: "Detailed result from PII detection with detected entities and redacted text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detailed result from PII detection with detected entities and redacted text.

## For Beginners

PIIResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `ContainsPII` | Whether any PII was detected. |
| `Entities` | List of detected PII entities with their locations, types, and confidence. |
| `EntityCounts` | Count of PII entities by type. |
| `RedactedText` | The text with PII redacted according to the configured strategy. |

