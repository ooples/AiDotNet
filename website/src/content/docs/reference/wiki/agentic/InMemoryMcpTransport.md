---
title: "InMemoryMcpTransport"
description: "An `IMcpTransport` that forwards requests directly to an in-process `McpServer` — no serialization or network."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Mcp`

An `IMcpTransport` that forwards requests directly to an in-process `McpServer` —
no serialization or network. Useful for tests, for embedding an MCP server in the same process as its
client, and for composing AiDotNet tools as MCP tools without a transport hop.

## For Beginners

A short-circuit pipe: instead of sending MCP messages over stdio or HTTP, it
hands them straight to a server object running in the same program. Great for wiring a client to a server
in one process (and for testing the two together).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InMemoryMcpTransport(McpServer)` | Initializes a new in-memory transport over the given server. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SendRequestAsync(String,JObject,CancellationToken)` |  |

