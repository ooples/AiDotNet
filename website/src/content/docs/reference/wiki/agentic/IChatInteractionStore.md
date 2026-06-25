---
title: "IChatInteractionStore"
description: "Stores recorded chat interactions (request key → response) so model calls can be deterministically replayed later without invoking any model."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Pipeline`

Stores recorded chat interactions (request key → response) so model calls can be deterministically
replayed later without invoking any model. This is the backing store for record/replay — the foundation
for reproducible agent runs, cheap re-runs, time-travel debugging, and offline tests.

## For Beginners

A cache of "for this exact request, the model said this". Record once against a
real model, then replay from the store forever — same inputs, same outputs, no API calls.

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of recorded interactions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Save(String,ChatResponse)` | Saves the response for a request key (overwriting any existing entry). |
| `TryGet(String,ChatResponse)` | Tries to get a recorded response for a request key. |

