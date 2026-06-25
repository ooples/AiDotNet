---
title: "StreamMcpTransport"
description: "An `IMcpTransport` that speaks newline-delimited JSON-RPC 2.0 over a reader/writer pair — the framing MCP uses over stdio."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Mcp`

An `IMcpTransport` that speaks newline-delimited JSON-RPC 2.0 over a reader/writer pair — the
framing MCP uses over stdio. Wire the writer to a child MCP server process's standard input and the reader
to its standard output to drive a local server process.

## For Beginners

The way to talk to an MCP server that runs as a separate program on your
machine: messages go in and out as lines of text over its input/output streams. Point this at that
program's streams and the client works the same as over HTTP.

## How It Works

Each request is written as one JSON line and flushed; responses are read line by line until the one whose
`id` matches the request (interleaved notifications and unrelated responses are skipped). Calls are
serialized with an internal lock so the request/response correlation is safe under concurrent callers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StreamMcpTransport(TextReader,TextWriter)` | Initializes a new stream transport over the given reader/writer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` |  |
| `SendRequestAsync(String,JObject,CancellationToken)` |  |

