---
title: "ChatClientBase<T>"
description: "Base class for `IChatClient` implementations that talk to an HTTP chat API."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Agentic.Models.Connectors`

Base class for `IChatClient` implementations that talk to an HTTP chat API. Provides
the shared HTTP client, validation, retry-with-backoff for non-streaming calls, and API-key checks,
leaving the provider-specific request/response mapping to subclasses.

## For Beginners

Every cloud chat provider needs the same plumbing — send an HTTP request,
retry if the network hiccups, give up cleanly on a bad API key. This class does all of that once, so
each provider class only has to describe how its specific API formats requests and responses.

## How It Works

Template-method pattern: this base owns the cross-cutting concerns (transport, retries, timeouts,
error wrapping); a concrete connector implements `CancellationToken)` and
`CancellationToken)` with the provider's wire format. Streaming calls are not
retried (a partially-consumed stream cannot be safely replayed).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChatClientBase(HttpClient)` | Initializes the base client. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HttpClient` | The HTTP client used for API communication. |
| `InitialRetryDelayMs` | The initial retry delay in milliseconds (doubles with each retry). |
| `MaxRetries` | The maximum number of retry attempts for failed non-streaming requests. |
| `ModelId` |  |
| `TimeoutMs` | The request timeout in milliseconds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |
| `GetResponseCoreAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` | Provider-specific non-streaming request/response mapping. |
| `GetStreamingResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |
| `GetStreamingResponseCoreAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` | Provider-specific streaming request/response mapping. |
| `IsRetryable(HttpRequestException)` | Determines whether a failed HTTP request should be retried (network errors, 429, 408, 5xx). |
| `ValidateApiKey(String,String)` | Validates that an API key is present. |

