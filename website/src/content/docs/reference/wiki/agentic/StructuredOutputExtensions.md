---
title: "StructuredOutputExtensions"
description: "Helpers that turn a chat model's reply into a strongly-typed .NET object by constraining the model to a JSON schema derived from the target type and deserializing the result."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.Tools`

Helpers that turn a chat model's reply into a strongly-typed .NET object by constraining the model
to a JSON schema derived from the target type and deserializing the result.

## For Beginners

Instead of getting a blob of text and parsing it yourself, you ask the
model for a specific shape — e.g. `await client.GetStructuredResponseAsync<double, Weather>("Weather in Paris?")` — and get back a filled-in `Weather` object. The library writes the schema, sets the
"reply as JSON matching this shape" flag, and deserializes for you.

## How It Works

These extensions close the loop between `JsonSchemaGenerator` (C# type → JSON Schema) and
the model's structured-output mode: the schema for `TResult` is attached to the
request as `JsonSchema`, and the JSON reply is deserialized back
into `TResult`. Providers that enforce the schema (and the local engine's
constrained decoding) guarantee the reply parses.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetStructuredResponseAsync(IChatClient<>,IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` | Sends a conversation and deserializes the schema-constrained JSON reply into `TResult`. |
| `GetStructuredResponseAsync(IChatClient<>,String,ChatOptions,CancellationToken)` | Sends a single user prompt and deserializes the schema-constrained JSON reply into `TResult`. |

