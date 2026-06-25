---
title: "OllamaChatClient<T>"
description: "An `IChatClient` for a local `` server, which exposes an OpenAI-compatible chat-completions API."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Connectors`

An `IChatClient` for a local `` server, which
exposes an OpenAI-compatible chat-completions API. It reuses the entire OpenAI wire format (messages,
tools, streaming, usage) and only changes the endpoint — so locally-served open models (Llama, Mistral,
Qwen, …) drive the agent stack with no extra code.

## For Beginners

If you run Ollama on your machine, this points the agent at it — a free, local,
private model with the same code you'd use for OpenAI.

## How It Works

Ollama ignores the bearer token, so a placeholder key is sent. The default endpoint targets a local
daemon (`http://localhost:11434/v1/chat/completions`); override it to reach a remote Ollama host.
This complements the in-process `LocalEngineChatClient`: Ollama runs models in a separate local
server process, while the local engine runs them inside AiDotNet itself.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OllamaChatClient(String,String,HttpClient)` | Initializes a new Ollama client. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultEndpoint` | The default Ollama OpenAI-compatible endpoint (local daemon). |

