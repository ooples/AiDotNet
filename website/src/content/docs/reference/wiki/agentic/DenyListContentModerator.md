---
title: "DenyListContentModerator"
description: "A simple `IContentModerator` that blocks content containing any of a configured set of banned terms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

A simple `IContentModerator` that blocks content containing any of a configured set of banned
terms. Zero-config and deterministic — useful as a baseline guardrail or for tests; swap in a classifier
(e.g., an `src/Safety`-backed moderator) for nuanced moderation.

## For Beginners

A blocklist checker: if the text contains a forbidden word, it's blocked.
Fast and predictable, but it only catches exact terms — not paraphrases or intent.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DenyListContentModerator(IEnumerable<String>,Boolean)` | Initializes a new deny-list moderator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CheckAsync(String,CancellationToken)` |  |

