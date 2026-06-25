---
title: "NodeModification"
description: "Represents a modification to be applied to a node in a computational graph."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LinearAlgebra`

Represents a modification to be applied to a node in a computational graph.

## For Beginners

In AI and machine learning, a computational graph is like a recipe that shows how 
calculations flow from inputs to outputs. Each step in this recipe is called a "node".

Sometimes, we need to change these nodes - maybe we want to remove a step, add a new one, or change how a step works.
This class helps us keep track of what changes we want to make to which nodes.

Think of it like editing instructions in a recipe: you might want to replace one ingredient with another,
remove a step, or add a new technique. This class helps keep track of those edits before you apply them.

## Properties

| Property | Summary |
|:-----|:--------|
| `NewNodeType` | Gets or sets the new type for the node when changing its functionality. |
| `NodeId` | Gets or sets the unique identifier of the node to be modified. |
| `Type` | Gets or sets the type of modification to apply to the node. |

