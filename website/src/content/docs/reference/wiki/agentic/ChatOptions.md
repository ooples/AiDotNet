---
title: "ChatOptions"
description: "Per-request settings for a chat call: sampling controls, tool availability, and output format."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Agentic.Models`

Per-request settings for a chat call: sampling controls, tool availability, and output format.

## For Beginners

Think of this as the knobs on the request. Leave a knob untouched
(`null`) and a reasonable default is used. Turn it to change behavior:

- `Temperature`: higher = more creative/random, lower = more focused.
- `MaxOutputTokens`: cap on reply length.
- `Tools` / `ToolChoice`: which tools the model may call, and how eagerly.
- `ResponseFormat`: ask for plain text or machine-readable JSON.

## How It Works

Every property is nullable. `null` means "use the provider's (or AiDotNet's) sensible default"
rather than forcing callers to specify everything. This follows the library-wide options pattern:
zero-config by default, fully overridable when needed. Connectors apply documented defaults when a
value is `null` (for example, temperature ≈ 0.7).

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxOutputTokens` | Gets or sets the maximum number of tokens to generate. |
| `RequiredToolName` | Gets or sets the specific tool the model must call. |
| `ResponseFormat` | Gets or sets the desired output format. |
| `ResponseJsonSchema` | Gets or sets the JSON schema enforced when `ResponseFormat` is `JsonSchema`. |
| `Seed` | Gets or sets a deterministic sampling seed where the provider supports it. |
| `StopSequences` | Gets or sets sequences that, when generated, cause the model to stop. |
| `Temperature` | Gets or sets the sampling temperature (typically 0.0–2.0). |
| `ToolChoice` | Gets or sets how the model may use the supplied `Tools`. |
| `Tools` | Gets or sets the tools the model is allowed to call this turn. |
| `TopK` | Gets or sets top-K sampling. |
| `TopP` | Gets or sets nucleus-sampling probability mass (0.0–1.0). |

