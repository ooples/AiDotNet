---
title: "OpenAIChatClient<T>"
description: "An `IChatClient` for OpenAI's Chat Completions API with native tool calling, streaming, structured output, and multimodal (image) input."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Connectors`

An `IChatClient` for OpenAI's Chat Completions API with native tool calling, streaming,
structured output, and multimodal (image) input.

## For Beginners

This is the adapter that lets the rest of the library talk to OpenAI models
(GPT-4o and friends) without knowing any OpenAI-specific details.

## How It Works

Translates the provider-neutral `ChatMessage`/`ChatOptions` model into OpenAI's
wire format and back, mapping provider strings (roles, `finish_reason`) onto the library's enums.
The same wire format is used by Azure OpenAI, so that connector derives from this one.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenAIChatClient(String,String,String,HttpClient)` | Initializes a new OpenAI chat client. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ApiKey` | Gets the API key used for authentication. |
| `Endpoint` | Gets the endpoint the request is posted to. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAuthentication(HttpRequestMessage)` | Applies authentication to the outbound request. |
| `BuildRequest(IReadOnlyList<ChatMessage>,ChatOptions,Boolean)` | Builds the OpenAI Chat Completions request body from the conversation and options. |
| `CreateHttpRequest(JObject)` | Creates the HTTP request and applies authentication. |
| `GetResponseCoreAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |
| `GetStreamingResponseCoreAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |

