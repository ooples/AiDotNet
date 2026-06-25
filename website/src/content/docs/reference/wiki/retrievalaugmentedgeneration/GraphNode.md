---
title: "GraphNode<T>"
description: "Represents a node in a knowledge graph, typically an entity extracted from text."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Represents a node in a knowledge graph, typically an entity extracted from text.

## For Beginners

Think of a node as a person in a social network.

Just like a Facebook profile has:

- Name: "John Smith"
- Properties: age, location, interests
- Connections: friends, family, coworkers

A GraphNode has:

- Id: Unique identifier
- Label: Entity type (PERSON, ORGANIZATION, LOCATION)
- Properties: Additional metadata
- Embedding: Numeric representation for similarity search

For example:

- Id: "person_123"
- Label: "PERSON"
- Properties: { "name": "Albert Einstein", "occupation": "Physicist" }
- Embedding: [0.23, -0.45, 0.67, ...] (vector representation)

## How It Works

A graph node stores an entity (person, place, concept, etc.) along with its properties and embeddings.
Nodes are connected by edges to form a knowledge graph that captures relationships between entities.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphNode(String,String)` | Initializes a new instance of the `GraphNode` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CreatedAt` | Timestamp when this node was created. |
| `Embedding` | Vector embedding for similarity search and clustering. |
| `Id` | Unique identifier for this node. |
| `Label` | The entity label or type (e.g., PERSON, ORGANIZATION, LOCATION). |
| `Properties` | Additional properties and metadata for this entity. |
| `UpdatedAt` | Timestamp when this node was last updated. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetProperty(String)` | Gets a property value from this node. |
| `SetProperty(String,Object)` | Adds or updates a property on this node. |

