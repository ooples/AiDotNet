---
title: "IMcpTransport"
description: "Carries Model Context Protocol (MCP) JSON-RPC requests to a server and returns the result."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Mcp`

Carries Model Context Protocol (MCP) JSON-RPC requests to a server and returns the result. Abstracting the
transport keeps `McpClient` independent of how messages are framed — stdio for local server
processes, HTTP/SSE for remote servers, or an in-memory loopback for tests.

## For Beginners

The pipe to an MCP server. The client says "call this method with these
arguments" and the pipe delivers it and brings back the answer — whether the server is a local program or a
remote service.

## How It Works

MCP is JSON-RPC 2.0. An implementation sends a request for `method` with
`parameters` and returns the JSON-RPC `result` object; on a JSON-RPC error it should
throw an `McpException`.

## Methods

| Method | Summary |
|:-----|:--------|
| `SendRequestAsync(String,JObject,CancellationToken)` | Sends a JSON-RPC request and returns its `result` object. |

