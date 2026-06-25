---
title: "LocalEngineOptions"
description: "Settings for `LocalEngineChatClient`: the reported model id, the default generation length, and the default sampling behavior (overridable per request via `ChatOptions`)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Agentic.Models.Local`

Settings for `LocalEngineChatClient`: the reported model id, the default generation length,
and the default sampling behavior (overridable per request via `ChatOptions`).

## For Beginners

These are the local model's defaults. The most useful is
`MaxOutputTokens` (how long a reply may get before the engine stops). Leave
`Sampling` unset for safe, near-greedy behavior, or set it to make replies more creative.

## Properties

| Property | Summary |
|:-----|:--------|
| `BeamWidth` | Gets or sets the beam width for beam-search decoding. |
| `Constraint` | Gets or sets a token constraint applied during generation (constrained decoding). |
| `MaxOutputTokens` | Gets or sets the default maximum number of tokens to generate per reply. |
| `ModelId` | Gets or sets the model id reported by `ModelId`. |
| `Sampling` | Gets or sets the default sampling settings. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultMaxOutputTokens` | The default maximum number of tokens generated per reply when none is specified. |

