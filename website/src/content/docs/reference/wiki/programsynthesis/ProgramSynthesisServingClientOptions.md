---
title: "ProgramSynthesisServingClientOptions"
description: "Configuration for calling an AiDotNet.Serving instance for Program Synthesis operations."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ProgramSynthesis.Serving`

Configuration for calling an AiDotNet.Serving instance for Program Synthesis operations.

## Properties

| Property | Summary |
|:-----|:--------|
| `ApiKey` | Optional API key (sent using `ApiKeyHeaderName`). |
| `ApiKeyHeaderName` | Header name used for API key authentication. |
| `BaseAddress` | Base address of the AiDotNet.Serving instance (e.g., http://localhost:52432/). |
| `BearerToken` | Optional bearer token (sent using the Authorization: Bearer header). |
| `HttpClient` | Optional HttpClient to use for requests (recommended for re-use). |
| `PreferServing` | When true, higher-level APIs prefer Serving when configured. |
| `TimeoutMs` | Request timeout in milliseconds. |

