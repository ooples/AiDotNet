---
title: "McpToolDescriptor"
description: "Describes a tool advertised by an MCP server: its name, description, and JSON-Schema input contract — the data needed to surface it to a model as a callable tool."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Mcp`

Describes a tool advertised by an MCP server: its name, description, and JSON-Schema input contract — the
data needed to surface it to a model as a callable tool.

## For Beginners

One entry from an MCP server's tool menu. `McpClient` turns each of
these into a tool your agent can call, no different from a built-in one.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `McpToolDescriptor(String,String,JObject)` | Initializes a new descriptor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets the tool description. |
| `InputSchema` | Gets a copy of the JSON-Schema describing the tool's arguments (mutating it does not affect the descriptor). |
| `Name` | Gets the tool name. |

