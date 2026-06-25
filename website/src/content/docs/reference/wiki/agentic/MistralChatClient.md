---
title: "MistralChatClient<T>"
description: "An `IChatClient` for the `` platform, whose chat API is OpenAI-compatible."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Connectors`

An `IChatClient` for the `` platform, whose chat
API is OpenAI-compatible. It reuses the OpenAI wire format and only changes the endpoint and key, adding
Mistral's hosted models to the connector lineup with no bespoke code.

## For Beginners

Same agent code as OpenAI, pointed at Mistral's servers with your Mistral key.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MistralChatClient(String,String,String,HttpClient)` | Initializes a new Mistral client. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultEndpoint` | The default Mistral OpenAI-compatible chat endpoint. |

