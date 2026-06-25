---
title: "McpServer"
description: "A Model Context Protocol (MCP) server that exposes a `ToolCollection` of AiDotNet tools to any MCP client (Claude Desktop, other agent frameworks, or AiDotNet's own `McpClient`)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Mcp`

A Model Context Protocol (MCP) server that exposes a `ToolCollection` of AiDotNet tools to any
MCP client (Claude Desktop, other agent frameworks, or AiDotNet's own `McpClient`). It handles
the MCP JSON-RPC methods (`initialize`, `tools/list`, `tools/call`) against the registered
tools — turning your tools (including model-as-tool and RAG pipelines) into a standard, interoperable
service.

## For Beginners

The flip side of the client. Instead of using someone else's tools, this lets
other AI apps use *your* AiDotNet tools through the same standard protocol — define a tool once and
any MCP-aware client can call it.

## How It Works

This is the inverse of `McpClient`: the client consumes external MCP tools, the server publishes
AiDotNet tools. It is transport-agnostic — `CancellationToken)` processes a parsed JSON-RPC
request, so a stdio/HTTP host (or the in-process `InMemoryMcpTransport`) can drive it.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `McpServer(ToolCollection,String,String)` | Initializes a new MCP server over the given tools. |

## Methods

| Method | Summary |
|:-----|:--------|
| `HandleRequestAsync(String,JObject,CancellationToken)` | Handles one MCP JSON-RPC request and returns its `result` object. |

