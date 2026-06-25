---
title: "McpClient"
description: "A Model Context Protocol (MCP) client: connects to an MCP server over an `IMcpTransport`, lists the tools it offers, and exposes them as `IAgentTool` instances the agent stack can call — so any MCP server's capabilities become available to…"
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Mcp`

A Model Context Protocol (MCP) client: connects to an MCP server over an `IMcpTransport`,
lists the tools it offers, and exposes them as `IAgentTool` instances the agent stack can call
— so any MCP server's capabilities become available to AiDotNet agents with no per-tool code.

## For Beginners

MCP servers publish tools (search the web, read a database, control an app).
This client connects to one, asks "what tools do you have?", and wraps each so your agent can use them
exactly like its own — instantly expanding what it can do.

## How It Works

MCP is the emerging standard (used by Semantic Kernel, LangGraph, and others) for connecting models to
external tools/data. `CancellationToken)` returns a `ToolCollection` of adapters that
forward calls to the server, so an MCP-hosted tool is indistinguishable from a native one to the model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `McpClient(IMcpTransport)` | Initializes a new MCP client over the given transport. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CallToolAsync(String,JObject,CancellationToken)` | Calls a tool on the server (the MCP `tools/call` method) and returns its result as a `ToolInvocationResult`. |
| `GetToolsAsync(CancellationToken)` | Lists the server's tools and wraps each as an `IAgentTool` in a ready-to-use `ToolCollection`. |
| `InitializeAsync(String,CancellationToken)` | Performs the MCP `initialize` handshake and returns the server's capabilities/info object. |
| `ListToolsAsync(CancellationToken)` | Lists the tools the server offers (the MCP `tools/list` method). |

