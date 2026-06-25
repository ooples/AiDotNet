---
title: "ExpressionTreeVelocity<T>"
description: "Represents the velocity (rate and direction of change) for an expression tree during optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LinearAlgebra`

Represents the velocity (rate and direction of change) for an expression tree during optimization.

## How It Works

**For Beginners:** Think of this class as tracking how a mathematical formula should change
during optimization. Just like velocity in physics describes how fast and in what direction
an object is moving, this class describes how the formula is "moving" or changing during the
optimization process. It keeps track of which numbers should change and how the structure
of the formula might be modified.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExpressionTreeVelocity` | Initializes a new instance of the ExpressionTreeVelocity class with empty collections. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NodeValueChanges` | A dictionary mapping node IDs to their value changes. |
| `StructureChanges` | A list of structural modifications to apply to the expression tree. |

