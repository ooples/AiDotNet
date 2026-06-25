---
title: "ExpressionNodeType"
description: "Defines the different types of nodes that can exist in a computational graph."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the different types of nodes that can exist in a computational graph.

## For Beginners

A computational graph is a way to represent mathematical operations as a network
of connected nodes. Think of it like a recipe with steps: some nodes are ingredients (constants and variables),
while others are actions (like add, subtract). This is how AI models internally organize calculations.
Each node in the graph performs a specific operation on its inputs and passes the result to the next node.

## Fields

| Field | Summary |
|:-----|:--------|
| `Add` | A node that performs addition on its inputs. |
| `Constant` | A node that represents a fixed numerical value that doesn't change during computation. |
| `Divide` | A node that performs division on its inputs. |
| `Multiply` | A node that performs multiplication on its inputs. |
| `Subtract` | A node that performs subtraction on its inputs. |
| `Variable` | A node that represents a value that can change during computation, such as an input or parameter. |

