---
title: "HttpMcpTransport"
description: "An `IMcpTransport` that speaks JSON-RPC 2.0 to a remote MCP server over HTTP POST."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Mcp`

An `IMcpTransport` that speaks JSON-RPC 2.0 to a remote MCP server over HTTP POST. Each request
is sent as a JSON-RPC envelope and the server's `result` is returned (a JSON-RPC `error` becomes
an `McpException`).

## For Beginners

The way to reach an MCP server that lives on the network. It wraps each call in
the standard JSON-RPC envelope, posts it, and unwraps the answer — so `McpClient` works against
a hosted server exactly as it does against an in-process one.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HttpMcpTransport(String,HttpClient)` | Initializes a new HTTP transport. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` | Disposes the internally created `HttpClient` (caller-supplied clients are left alone). |
| `SendRequestAsync(String,JObject,CancellationToken)` |  |

