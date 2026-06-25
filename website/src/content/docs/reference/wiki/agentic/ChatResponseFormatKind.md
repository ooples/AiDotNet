---
title: "ChatResponseFormatKind"
description: "Selects the shape the model's output must take (free text, arbitrary JSON, or schema-constrained JSON)."
section: "API Reference"
---

`Enums` · `AiDotNet.Agentic.Models`

Selects the shape the model's output must take (free text, arbitrary JSON, or schema-constrained JSON).

## For Beginners

By default a model replies with ordinary prose, which is hard for code
to read reliably. These options let you ask for machine-readable output instead:

- **Text**: normal human-readable text (the default).
- **Json**: "reply with valid JSON" (shape not guaranteed).
- **JsonSchema**: "reply with JSON that matches exactly this structure" (shape guaranteed).

## How It Works

Structured output is what makes model responses safe to parse programmatically. When
`JsonSchema` is requested, the accompanying JSON schema is enforced — by the provider
for cloud models, or by constrained decoding for the local in-process engine — so the result is
guaranteed to deserialize.

## Fields

| Field | Summary |
|:-----|:--------|
| `Json` | Syntactically valid JSON, but with no guarantee about which fields are present (often called "JSON mode"). |
| `JsonSchema` | JSON constrained to a supplied JSON schema, so the output is guaranteed to deserialize into the expected type. |
| `Text` | Ordinary free-form text. |

