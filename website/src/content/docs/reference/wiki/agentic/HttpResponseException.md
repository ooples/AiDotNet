---
title: "HttpResponseException"
description: "An `HttpRequestException` that preserves the HTTP status code of a non-success API response on every target framework."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Agentic.Models.Connectors`

An `HttpRequestException` that preserves the HTTP status code of a non-success API response on
every target framework. On .NET Framework, `HttpRequestException` exposes no status code, so
connectors throw this instead — keeping the status available to retry classification
(`ChatClientBase.IsRetryable`) without it being lost into a plain message string.

## How It Works

The property is named `ResponseStatusCode` (not `StatusCode`) to avoid hiding the
nullable `HttpRequestException.StatusCode` that exists on modern frameworks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HttpResponseException(HttpStatusCode,String)` | Initializes a new instance carrying the failing response's status code. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ResponseStatusCode` | Gets the HTTP status code of the non-success response. |

