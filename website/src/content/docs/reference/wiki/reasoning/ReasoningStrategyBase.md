---
title: "ReasoningStrategyBase<T>"
description: "Abstract base class for reasoning strategies that solve problems through structured thinking."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Reasoning`

Abstract base class for reasoning strategies that solve problems through structured thinking.
Provides common functionality for all reasoning approaches.

## For Beginners

This base class is like a template for creating different types of reasoning strategies.
Just like AgentBase provides common functionality for all agents, ReasoningStrategyBase provides the shared
foundation that all reasoning strategies need:

- Managing the language model (the "brain")
- Tracking tools that can be used
- Recording the reasoning process
- Handling configuration and timing

Specific reasoning strategies (Chain-of-Thought, Tree-of-Thoughts, etc.) inherit from this class
and implement their unique reasoning logic while getting all the common features for free.

This follows the Template Method design pattern, where the base class defines the structure
and derived classes fill in the specific details.

## How It Works

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReasoningStrategyBase(IChatClient<>,IEnumerable<IAgentTool>)` | Initializes a new instance of the `ReasoningStrategyBase` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChatModel` | Gets the chat model used for reasoning. |
| `Description` |  |
| `ReasoningTrace` | Gets the current reasoning trace. |
| `StrategyName` |  |
| `Tools` | Gets the read-only list of tools available to this strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AppendTrace(String)` | Appends a message to the reasoning trace. |
| `ClearTrace` | Clears the reasoning trace, starting fresh. |
| `ExecuteToolAsync(String,String,CancellationToken)` | Executes a tool with the given input. |
| `FindTool(String)` | Finds a tool by its name. |
| `GetToolDescriptions` | Gets a formatted description of all available tools. |
| `ReasonAsync(String,ReasoningConfig,CancellationToken)` |  |
| `ReasonCoreAsync(String,ReasoningConfig,CancellationToken)` | Core reasoning logic to be implemented by derived strategies. |
| `ValidateConfig(ReasoningConfig)` | Validates that a reasoning configuration is valid. |

