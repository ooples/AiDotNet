---
title: "RecordingChatClient<T>"
description: "An `IChatClient` decorator that calls a real inner client and records each request/response into an `IChatInteractionStore`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

An `IChatClient` decorator that calls a real inner client and records each request/response
into an `IChatInteractionStore`. Pair it with `ReplayingChatClient` to capture a
run once and replay it deterministically thereafter.

## For Beginners

A tape recorder around the model. It passes your request to the real model and
quietly saves the answer keyed by the request, so you can play it back later without spending another call.
Streaming calls are recorded too: the streamed pieces are reassembled into the final answer and saved when
the stream finishes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RecordingChatClient(IChatClient<>,IChatInteractionStore)` | Initializes a new recording client. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelId` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |
| `GetStreamingResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |

