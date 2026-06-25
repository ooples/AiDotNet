---
title: "GraphQueryMatcher<T>"
description: "Simple pattern matching for graph queries (inspired by Cypher/SPARQL but simplified)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Simple pattern matching for graph queries (inspired by Cypher/SPARQL but simplified).

## For Beginners

Pattern matching is like SQL for graphs.

SQL Example:
```sql
SELECT * FROM persons WHERE name = 'Alice'
```

Graph Pattern Example:
```
(Person {name: "Alice"})-[KNOWS]->(Person)
```
Meaning: Find all people that Alice knows

Another Example:
```
(Person)-[WORKS_AT]->(Company {name: "Google"})
```
Meaning: Find all people who work at Google

This is much more natural for relationship-heavy data!

## How It Works

Supports basic graph pattern matching queries like:

- (Person)-[KNOWS]->(Person)
- (Person {name: "Alice"})-[WORKS_AT]->(Company)
- (a:Person)-[r:KNOWS]->(b:Person)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphQueryMatcher(KnowledgeGraph<>)` | Initializes a new instance of the `GraphQueryMatcher` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AreEqual(Object,Object)` | Compares two objects for equality. |
| `ExecutePattern(String)` | Executes a simple pattern query string. |
| `FindNodes(String,Dictionary<String,Object>)` | Finds nodes matching a label and optional property filters. |
| `FindPaths(String,String,String,Dictionary<String,Object>,Dictionary<String,Object>)` | Finds paths matching a pattern: (source label)-[relationship type]->(target label). |
| `FindPathsOfLength(String,Int32,String)` | Finds all paths of specified length from a source node. |
| `FindShortestPaths(String,String,Int32)` | Finds all shortest paths between two nodes. |
| `IsNumeric(Object)` | Checks if an object is numeric. |
| `MatchesProperties(GraphNode<>,Dictionary<String,Object>)` | Checks if a node matches property filters. |
| `ParseProperties(String)` | Parses property string into dictionary. |

