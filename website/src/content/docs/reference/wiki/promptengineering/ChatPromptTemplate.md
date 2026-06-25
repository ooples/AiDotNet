---
title: "ChatPromptTemplate"
description: "ChatPromptTemplate — Models & Types in AiDotNet.PromptEngineering.Templates."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Templates`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChatPromptTemplate(String)` | Initializes a new instance of the ChatPromptTemplate class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Messages` | Gets all messages in the conversation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddAssistantMessage(String)` | Adds an assistant message to the conversation. |
| `AddMessage(String,String)` | Adds a message with a custom role. |
| `AddSystemMessage(String)` | Adds a system message to the conversation. |
| `AddUserMessage(String)` | Adds a user message to the conversation. |
| `CapitalizeFirst(String)` | Capitalizes the first letter of a string. |
| `FormatCore(Dictionary<String,String>)` | Formats the template (chat templates return the conversation as-is). |
| `UpdateTemplate` | Updates the internal template based on current messages. |

