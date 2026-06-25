---
title: "AzureOpenAIChatClient<T>"
description: "An `IChatClient` for Azure OpenAI."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Connectors`

An `IChatClient` for Azure OpenAI. Azure uses the same Chat Completions wire format as
OpenAI, so this derives from `OpenAIChatClient` and only changes the endpoint (a
per-deployment URL with an `api-version`) and the authentication header (`api-key`).

## For Beginners

Azure hosts the same OpenAI models but behind your own Azure resource. You
address a "deployment" you created, the URL includes an API version, and the key goes in an
`api-key` header instead of `Authorization`. Everything else is identical to OpenAI, which is
why this class reuses the OpenAI request/response logic.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AzureOpenAIChatClient(String,String,String,String,HttpClient)` | Initializes a new Azure OpenAI chat client. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAuthentication(HttpRequestMessage)` |  |

