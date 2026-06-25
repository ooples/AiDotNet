---
title: "GraphEdge<T>"
description: "Represents a directed edge (relationship) between two nodes in a knowledge graph."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Represents a directed edge (relationship) between two nodes in a knowledge graph.

## For Beginners

Think of an edge as a relationship or connection between two people.

In a social network:

- "Alice WORKS_FOR Microsoft" (Alice is the source, Microsoft is the target)
- "Bob LIVES_IN Seattle" (Bob is the source, Seattle is the target)
- "Charlie KNOWS David" (Charlie knows David, but maybe David doesn't know Charlie - it's directional!)

In a knowledge graph:

- Source: The entity the relationship starts from
- Target: The entity the relationship points to
- Type: The kind of relationship (WORKS_FOR, LIVES_IN, KNOWS)
- Properties: Extra info (since: "2020", strength: 0.9)
- Weight: How important or strong this relationship is (0.0 to 1.0)

For example:
Source: "Albert Einstein" (PERSON)
Target: "Princeton University" (ORGANIZATION)
Type: "WORKED_AT"
Properties: { "from": "1933", "to": "1955" }
Weight: 0.95 (very strong relationship)

## How It Works

A graph edge represents a relationship between two entities, such as "works_for", "located_in", or "friend_of".
Edges are directed (from source to target) and can have properties to store additional relationship metadata.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphEdge(String,String,String,Double)` | Initializes a new instance of the `GraphEdge` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CreatedAt` | Timestamp when this edge was created. |
| `Id` | Unique identifier for this edge. |
| `Properties` | Additional properties and metadata for this relationship. |
| `RelationType` | The relationship type (e.g., WORKS_FOR, LOCATED_IN, FRIEND_OF). |
| `SourceId` | The source node ID (where the relationship starts). |
| `TargetId` | The target node ID (where the relationship points to). |
| `ValidFrom` | Start of the temporal validity window for this relationship. |
| `ValidUntil` | End of the temporal validity window for this relationship. |
| `Weight` | Weight or strength of this relationship (0.0 to 1.0). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetProperty(String)` | Gets a property value from this edge. |
| `IsValidAt(DateTime)` | Checks whether this edge is valid at a specific point in time. |
| `SetProperty(String,Object)` | Adds or updates a property on this edge. |
| `SetTemporalWindow(Nullable<DateTime>,Nullable<DateTime>)` | Sets the temporal validity window for this edge with validation. |

