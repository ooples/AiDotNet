# Junior Developer Implementation Guide: Issue #306
## Graph Database for Production-Scale Knowledge Graphs

**Issue:** [#306 - Build In-House Graph Database for Production-Scale Knowledge Graphs](https://github.com/ooples/AiDotNet/issues/306)

**Target Audience:** Junior developers new to graph databases, knowledge graphs, and graph traversal algorithms

**Estimated Time:** 4-6 days for a junior developer

---

## Table of Contents

1. [Understanding the Problem](#understanding-the-problem)
2. [Core Concepts](#core-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Phase-by-Phase Implementation Guide](#phase-by-phase-implementation-guide)
5. [Testing Strategy](#testing-strategy)
6. [Common Pitfalls](#common-pitfalls)
7. [Resources](#resources)

---

## Understanding the Problem

### What Are We Building?

We're refactoring the existing `KnowledgeGraph<T>` class to support **multiple storage backends** (in-memory and file-based), and implementing a **persistent file-based graph database**. This allows knowledge graphs to survive application restarts and handle millions of entities.

### Why Do We Need This?

The current `KnowledgeGraph<T>` is tightly coupled to in-memory storage. This creates two problems:

1. **Data Loss:** When the application closes, all graph data is lost
2. **Scalability:** Large graphs (millions of nodes) can exceed available RAM

Our solution separates the graph logic from storage, allowing us to plug in different backends.

### Real-World Analogy

Think of it like a library system:

- **Current System:** The librarian memorizes where every book is. Fast, but if the librarian leaves, all knowledge is lost, and they can only remember so many books.
- **New System:** The librarian uses a card catalog (persistent storage). The catalog survives even when the librarian goes home, and can handle millions of books.

The `IGraphStore<T>` interface is the contract that defines how the catalog works, allowing different implementations (index cards, computer database, etc.).

---

## Core Concepts

### 1. Knowledge Graphs Explained

**What is a knowledge graph?**

A knowledge graph represents information as entities (nodes) and relationships (edges). Think of it like a map of how facts connect:

```
NODES (Entities):
- Albert Einstein (PERSON)
- Princeton University (ORGANIZATION)
- Physics (FIELD)
- Germany (COUNTRY)

EDGES (Relationships):
Albert Einstein --[WORKED_AT]--> Princeton University
Albert Einstein --[BORN_IN]--> Germany
Albert Einstein --[STUDIED]--> Physics
Princeton University --[LOCATED_IN]--> USA
```

**Why graphs?**

Graphs excel at representing interconnected data:

```csharp
// Traditional database (relational):
SELECT * FROM people WHERE name = 'Albert Einstein'
// Returns: One row with person data

// Graph database:
MATCH (p:PERSON {name: 'Albert Einstein'})-[:WORKED_AT]->(org)
// Returns: Einstein AND all organizations he worked at
// Can traverse: "Find all physicists who worked at Princeton"
```

### 2. Graph Traversal

**What is traversal?**

Traversal means walking through the graph following edges:

```
Starting from: Albert Einstein

1-hop traversal:
    Einstein --[WORKED_AT]--> Princeton
    Einstein --[BORN_IN]--> Germany
    Einstein --[STUDIED]--> Physics

2-hop traversal:
    Einstein --[WORKED_AT]--> Princeton --[LOCATED_IN]--> USA
    Einstein --[STUDIED]--> Physics --[RELATED_TO]--> Mathematics

3-hop traversal:
    Einstein --[WORKED_AT]--> Princeton --[EMPLOYS]--> Other Scientists --[PUBLISHED]--> Papers
```

**Breadth-First Search (BFS):**

BFS explores all neighbors at the current depth before moving deeper:

```
Start: A
Queue: [A]

Step 1: Visit A, add neighbors (B, C)
Queue: [B, C]

Step 2: Visit B, add neighbors (D, E)
Queue: [C, D, E]

Step 3: Visit C, add neighbors (F)
Queue: [D, E, F]

Result order: A, B, C, D, E, F
```

**Why BFS?**

- Finds shortest path between nodes
- Explores nearby nodes first (useful for "related entities")
- Easy to limit depth (e.g., "find entities within 2 hops")

### 3. Graph Storage Abstraction

**What is `IGraphStore<T>`?**

An interface defining how to store and retrieve graph data, without specifying *where* or *how*:

```csharp
public interface IGraphStore<T>
{
    // Node operations
    void AddNode(GraphNode<T> node);
    GraphNode<T>? GetNode(string nodeId);
    bool RemoveNode(string nodeId);

    // Edge operations
    void AddEdge(GraphEdge<T> edge);
    GraphEdge<T>? GetEdge(string edgeId);
    bool RemoveEdge(string edgeId);

    // Traversal support
    IEnumerable<GraphEdge<T>> GetOutgoingEdges(string nodeId);
    IEnumerable<GraphEdge<T>> GetIncomingEdges(string nodeId);

    // Querying
    IEnumerable<GraphNode<T>> GetAllNodes();
}
```

**Why abstraction?**

Different use cases need different storage:

```csharp
// Development: Fast, temporary
IGraphStore<float> store = new MemoryGraphStore<float>();

// Production: Persistent, scalable
IGraphStore<float> store = new FileGraphStore<float>("./data/graph");

// Enterprise: Distributed, replicated
IGraphStore<float> store = new SQLGraphStore<float>(connectionString);

// Same KnowledgeGraph code works with all!
var graph = new KnowledgeGraph<float>(store);
```

### 4. File-Based Graph Storage

**How do we store graphs in files?**

Similar to the document store, we use multiple files:

```
/data/graph/
├── nodes.dat           # Serialized nodes (binary)
├── edges.dat           # Serialized edges (binary)
├── node_index.db       # B-Tree: nodeId -> file offset
├── edge_index.db       # B-Tree: edgeId -> file offset
├── outgoing_edges.idx  # Map: nodeId -> [edgeIds]
└── incoming_edges.idx  # Map: nodeId -> [edgeIds]
```

**Why separate edge indexes?**

Graph traversal requires fast edge lookups:

```csharp
// To traverse from node A:
1. Look up outgoing edges: outgoing_edges["A"] = ["edge1", "edge2", "edge3"]
2. For each edge ID, get edge details from edge_index.db
3. Get target node from each edge

// Without index: Scan ALL edges to find ones from node A (very slow!)
```

### 5. B-Tree Indexing

**What is a B-Tree? (Detailed)**

A B-Tree is a balanced tree structure that keeps data sorted and allows searches, insertions, and deletions in logarithmic time:

```
B-Tree of order 3 (max 2 keys per node):

                    [G]
                   /   \
              [C, E]   [K, M]
             /  |  \   /  |  \
          [A,B][D][F][H,I,J][L][N,O,P]

To find "J":
1. Start at root: G
2. J > G, go right
3. Compare with K, M: J < K
4. Go to first child
5. Found in [H, I, J]

Only 5 comparisons instead of scanning all 15 items!
```

**For our use case:**

```csharp
// B-Tree maps string keys to file offsets
BTreeIndex nodeIndex;

// Adding a node
nodeIndex.Add("einstein_person", offset: 1024);

// Looking up a node
long? offset = nodeIndex.Get("einstein_person");
// Returns: 1024 (the file position where this node is stored)

// Removing a node
bool removed = nodeIndex.Remove("einstein_person");
```

**B-Tree benefits:**

- **Logarithmic lookup:** O(log N) instead of O(N)
- **Sorted keys:** Can iterate in order
- **Balanced:** Performance doesn't degrade with data distribution
- **Disk-friendly:** Minimizes disk reads (each node is a disk block)

---

## Architecture Overview

### Current Architecture (Before Refactoring)

```
KnowledgeGraph<T>
    |
    +-- private Dictionary<string, GraphNode<T>> _nodes
    +-- private Dictionary<string, GraphEdge<T>> _edges
    +-- private Dictionary<string, HashSet<string>> _outgoingEdges
    +-- private Dictionary<string, HashSet<string>> _incomingEdges
    +-- AddNode(node) { _nodes[node.Id] = node; }
    +-- GetNode(nodeId) { return _nodes[nodeId]; }
```

**Problem:** Storage (dictionaries) is tightly coupled with logic (graph operations).

### New Architecture (After Refactoring)

```
IGraphStore<T>                  KnowledgeGraph<T>
    ↑                               |
    |                               +-- private IGraphStore<T> _store
    |                               +-- AddNode(node) { _store.AddNode(node); }
    |                               +-- GetNode(id) { return _store.GetNode(id); }
    |                               +-- BreadthFirstTraversal(...) { /* logic */ }
    |
    +-- MemoryGraphStore<T>         (Uses dictionaries internally)
    +-- FileGraphStore<T>           (Uses files + B-Tree indexes)
    +-- SQLGraphStore<T>            (Uses SQL database) [future]
```

**Benefits:**

1. **Separation of Concerns:** Graph logic (traversal, pathfinding) separate from storage
2. **Testability:** Easy to mock `IGraphStore<T>` for testing
3. **Flexibility:** Swap storage backends without changing graph logic
4. **Scalability:** File/SQL backends handle larger graphs

### Class Responsibilities

**`IGraphStore<T>`** (Interface)
- Defines contract for graph storage
- No implementation, just method signatures

**`MemoryGraphStore<T>`** (Implementation)
- In-memory storage using dictionaries
- Fast, but not persistent
- Extracted from current `KnowledgeGraph<T>`

**`FileGraphStore<T>`** (Implementation)
- File-based persistent storage
- Uses B-Tree indexes for fast lookups
- Slower than memory, but survives restarts

**`KnowledgeGraph<T>`** (Graph Logic)
- Delegates storage to `IGraphStore<T>`
- Implements graph algorithms (BFS, shortest path, etc.)
- Uses dependency injection for storage

---

## Phase-by-Phase Implementation Guide

### Phase 1: Decouple Graph Logic from Storage

#### AC 1.1: Create Storage Abstraction

**Goal:** Define the interface that all graph stores must implement.

**Step 1: Create the interface file**

Location: `src/RetrievalAugmentedGeneration/Graph/Abstractions/IGraphStore.cs`

```csharp
using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Abstractions;

/// <summary>
/// Defines the contract for graph storage backends.
/// </summary>
/// <remarks>
/// <para>
/// IGraphStore abstracts graph storage operations, allowing KnowledgeGraph to work
/// with different storage implementations (in-memory, file-based, database, etc.).
/// </para>
/// <para><b>For Beginners:</b> This is like a blueprint for building a graph storage system.
///
/// Think of it like defining what a "database" should be able to do:
/// - Add/remove/get nodes (entities)
/// - Add/remove/get edges (relationships)
/// - Find edges connected to a node
///
/// Different implementations (memory, files, SQL) all follow this blueprint,
/// so KnowledgeGraph can work with any of them.
///
/// Example:
/// <code>
/// // All these implement IGraphStore, so they work the same way
/// IGraphStore&lt;float&gt; memoryStore = new MemoryGraphStore&lt;float&gt;();
/// IGraphStore&lt;float&gt; fileStore = new FileGraphStore&lt;float&gt;("./data");
///
/// var graph1 = new KnowledgeGraph&lt;float&gt;(memoryStore);
/// var graph2 = new KnowledgeGraph&lt;float&gt;(fileStore);
/// // Same graph operations, different storage!
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
public interface IGraphStore<T>
{
    /// <summary>
    /// Adds a node to the graph store.
    /// </summary>
    /// <param name="node">The node to add.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Stores an entity in the graph.
    ///
    /// Like adding a contact to your phone:
    /// - If the contact exists, update their info
    /// - If new, create a new entry
    /// </para>
    /// </remarks>
    void AddNode(GraphNode<T> node);

    /// <summary>
    /// Retrieves a node by its unique identifier.
    /// </summary>
    /// <param name="nodeId">The node ID.</param>
    /// <returns>The node if found; otherwise, null.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Looks up an entity by its ID.
    ///
    /// Like searching for a contact by name in your phone.
    /// Returns null if not found.
    /// </para>
    /// </remarks>
    GraphNode<T>? GetNode(string nodeId);

    /// <summary>
    /// Removes a node from the graph store.
    /// </summary>
    /// <param name="nodeId">The node ID to remove.</param>
    /// <returns>True if the node was removed; false if not found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Deletes an entity from the graph.
    ///
    /// Like deleting a contact from your phone.
    /// Returns false if the contact didn't exist.
    /// </para>
    /// </remarks>
    bool RemoveNode(string nodeId);

    /// <summary>
    /// Adds an edge to the graph store.
    /// </summary>
    /// <param name="edge">The edge to add.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a relationship between two entities.
    ///
    /// Like adding "Person A is friends with Person B" to your social network.
    /// Both people must already exist in the graph (added with AddNode first).
    /// </para>
    /// </remarks>
    void AddEdge(GraphEdge<T> edge);

    /// <summary>
    /// Retrieves an edge by its unique identifier.
    /// </summary>
    /// <param name="edgeId">The edge ID.</param>
    /// <returns>The edge if found; otherwise, null.</returns>
    GraphEdge<T>? GetEdge(string edgeId);

    /// <summary>
    /// Removes an edge from the graph store.
    /// </summary>
    /// <param name="edgeId">The edge ID to remove.</param>
    /// <returns>True if the edge was removed; false if not found.</returns>
    bool RemoveEdge(string edgeId);

    /// <summary>
    /// Gets all edges originating from a node (outgoing edges).
    /// </summary>
    /// <param name="nodeId">The source node ID.</param>
    /// <returns>Collection of outgoing edges.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Finds all relationships starting from this entity.
    ///
    /// Example:
    /// For "Albert Einstein", get all edges like:
    /// - Einstein --[WORKED_AT]--> Princeton
    /// - Einstein --[STUDIED]--> Physics
    /// - Einstein --[BORN_IN]--> Germany
    ///
    /// These are "outgoing" because they start at Einstein and point to other entities.
    /// </para>
    /// </remarks>
    IEnumerable<GraphEdge<T>> GetOutgoingEdges(string nodeId);

    /// <summary>
    /// Gets all edges pointing to a node (incoming edges).
    /// </summary>
    /// <param name="nodeId">The target node ID.</param>
    /// <returns>Collection of incoming edges.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Finds all relationships pointing to this entity.
    ///
    /// Example:
    /// For "Princeton University", get all edges like:
    /// - Einstein --[WORKED_AT]--> Princeton
    /// - Feynman --[WORKED_AT]--> Princeton
    /// - Oppenheimer --[WORKED_AT]--> Princeton
    ///
    /// These are "incoming" because they point to Princeton from other entities.
    /// Useful for questions like "Who worked at Princeton?"
    /// </para>
    /// </remarks>
    IEnumerable<GraphEdge<T>> GetIncomingEdges(string nodeId);

    /// <summary>
    /// Gets all nodes in the graph.
    /// </summary>
    /// <returns>Collection of all nodes.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns every entity in the graph.
    ///
    /// Like listing all contacts in your phone.
    /// Warning: This can be slow for large graphs (millions of nodes).
    /// </para>
    /// </remarks>
    IEnumerable<GraphNode<T>> GetAllNodes();

    /// <summary>
    /// Clears all nodes and edges from the store.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Deletes everything from the graph.
    ///
    /// Like factory-resetting your phone - all data is gone!
    /// Use with caution.
    /// </para>
    /// </remarks>
    void Clear();

    /// <summary>
    /// Gets the total number of nodes in the graph.
    /// </summary>
    int NodeCount { get; }

    /// <summary>
    /// Gets the total number of edges in the graph.
    /// </summary>
    int EdgeCount { get; }
}
```

**Understanding the interface:**

1. **Node operations:** Basic CRUD for entities
2. **Edge operations:** Basic CRUD for relationships
3. **Traversal support:** `GetOutgoingEdges` and `GetIncomingEdges` are crucial for graph algorithms
4. **Bulk operations:** `GetAllNodes` for exporting/analyzing the entire graph

#### AC 1.2: Implement In-Memory Store

**Goal:** Extract the current in-memory logic into its own class.

**Step 1: Create MemoryGraphStore**

Location: `src/RetrievalAugmentedGeneration/Graph/MemoryGraphStore.cs`

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Abstractions;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// In-memory graph store using dictionaries for fast access.
/// </summary>
/// <remarks>
/// <para>
/// MemoryGraphStore provides high-performance graph storage in RAM.
/// All data is lost when the application stops. Ideal for development,
/// testing, and session-based graphs.
/// </para>
/// <para><b>For Beginners:</b> This stores the graph in your computer's memory (RAM).
///
/// Think of it like writing notes on a whiteboard:
/// - Very fast to write and read
/// - Everything is erased when you leave (or program stops)
/// - Limited by whiteboard size (RAM)
///
/// Best for:
/// - Development and testing
/// - Temporary graphs (session data)
/// - Small to medium graphs (< 1 million nodes)
///
/// Not suitable for:
/// - Production systems requiring persistence
/// - Very large graphs (> available RAM)
/// - Multi-process systems (each process has its own memory)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
public class MemoryGraphStore<T> : IGraphStore<T>
{
    private readonly Dictionary<string, GraphNode<T>> _nodes;
    private readonly Dictionary<string, GraphEdge<T>> _edges;
    private readonly Dictionary<string, HashSet<string>> _outgoingEdges; // nodeId -> edge IDs
    private readonly Dictionary<string, HashSet<string>> _incomingEdges; // nodeId -> edge IDs

    /// <summary>
    /// Gets the total number of nodes in the graph.
    /// </summary>
    public int NodeCount => _nodes.Count;

    /// <summary>
    /// Gets the total number of edges in the graph.
    /// </summary>
    public int EdgeCount => _edges.Count;

    /// <summary>
    /// Initializes a new instance of the MemoryGraphStore class.
    /// </summary>
    public MemoryGraphStore()
    {
        _nodes = new Dictionary<string, GraphNode<T>>();
        _edges = new Dictionary<string, GraphEdge<T>>();
        _outgoingEdges = new Dictionary<string, HashSet<string>>();
        _incomingEdges = new Dictionary<string, HashSet<string>>();
    }

    /// <summary>
    /// Adds a node to the in-memory store.
    /// </summary>
    public void AddNode(GraphNode<T> node)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        _nodes[node.Id] = node;

        // Initialize edge sets for this node if they don't exist
        if (!_outgoingEdges.ContainsKey(node.Id))
            _outgoingEdges[node.Id] = new HashSet<string>();
        if (!_incomingEdges.ContainsKey(node.Id))
            _incomingEdges[node.Id] = new HashSet<string>();
    }

    /// <summary>
    /// Retrieves a node by its unique identifier.
    /// </summary>
    public GraphNode<T>? GetNode(string nodeId)
    {
        return _nodes.TryGetValue(nodeId, out var node) ? node : null;
    }

    /// <summary>
    /// Removes a node from the in-memory store.
    /// </summary>
    public bool RemoveNode(string nodeId)
    {
        if (!_nodes.ContainsKey(nodeId))
            return false;

        // Remove the node
        _nodes.Remove(nodeId);

        // Remove all edges connected to this node
        if (_outgoingEdges.TryGetValue(nodeId, out var outgoing))
        {
            foreach (var edgeId in outgoing.ToList())
                RemoveEdge(edgeId);
        }

        if (_incomingEdges.TryGetValue(nodeId, out var incoming))
        {
            foreach (var edgeId in incoming.ToList())
                RemoveEdge(edgeId);
        }

        // Clean up edge sets
        _outgoingEdges.Remove(nodeId);
        _incomingEdges.Remove(nodeId);

        return true;
    }

    /// <summary>
    /// Adds an edge to the in-memory store.
    /// </summary>
    public void AddEdge(GraphEdge<T> edge)
    {
        if (edge == null)
            throw new ArgumentNullException(nameof(edge));
        if (!_nodes.ContainsKey(edge.SourceId))
            throw new InvalidOperationException($"Source node '{edge.SourceId}' does not exist");
        if (!_nodes.ContainsKey(edge.TargetId))
            throw new InvalidOperationException($"Target node '{edge.TargetId}' does not exist");

        _edges[edge.Id] = edge;

        // Add to outgoing edges of source node
        if (!_outgoingEdges.ContainsKey(edge.SourceId))
            _outgoingEdges[edge.SourceId] = new HashSet<string>();
        _outgoingEdges[edge.SourceId].Add(edge.Id);

        // Add to incoming edges of target node
        if (!_incomingEdges.ContainsKey(edge.TargetId))
            _incomingEdges[edge.TargetId] = new HashSet<string>();
        _incomingEdges[edge.TargetId].Add(edge.Id);
    }

    /// <summary>
    /// Retrieves an edge by its unique identifier.
    /// </summary>
    public GraphEdge<T>? GetEdge(string edgeId)
    {
        return _edges.TryGetValue(edgeId, out var edge) ? edge : null;
    }

    /// <summary>
    /// Removes an edge from the in-memory store.
    /// </summary>
    public bool RemoveEdge(string edgeId)
    {
        if (!_edges.TryGetValue(edgeId, out var edge))
            return false;

        _edges.Remove(edgeId);

        // Remove from outgoing/incoming edge sets
        if (_outgoingEdges.TryGetValue(edge.SourceId, out var outgoing))
            outgoing.Remove(edgeId);
        if (_incomingEdges.TryGetValue(edge.TargetId, out var incoming))
            incoming.Remove(edgeId);

        return true;
    }

    /// <summary>
    /// Gets all edges originating from a node.
    /// </summary>
    public IEnumerable<GraphEdge<T>> GetOutgoingEdges(string nodeId)
    {
        if (!_outgoingEdges.TryGetValue(nodeId, out var edgeIds))
            return Enumerable.Empty<GraphEdge<T>>();

        return edgeIds.Select(id => _edges[id]);
    }

    /// <summary>
    /// Gets all edges pointing to a node.
    /// </summary>
    public IEnumerable<GraphEdge<T>> GetIncomingEdges(string nodeId)
    {
        if (!_incomingEdges.TryGetValue(nodeId, out var edgeIds))
            return Enumerable.Empty<GraphEdge<T>>();

        return edgeIds.Select(id => _edges[id]);
    }

    /// <summary>
    /// Gets all nodes in the graph.
    /// </summary>
    public IEnumerable<GraphNode<T>> GetAllNodes()
    {
        return _nodes.Values;
    }

    /// <summary>
    /// Clears all nodes and edges from the store.
    /// </summary>
    public void Clear()
    {
        _nodes.Clear();
        _edges.Clear();
        _outgoingEdges.Clear();
        _incomingEdges.Clear();
    }
}
```

**Understanding MemoryGraphStore:**

This is almost identical to the old `KnowledgeGraph<T>` internals, just extracted into its own class that implements `IGraphStore<T>`.

**Why the edge sets?**

```csharp
// Without edge sets (slow):
public IEnumerable<GraphEdge<T>> GetOutgoingEdges(string nodeId)
{
    // Scan ALL edges to find ones from this node
    return _edges.Values.Where(edge => edge.SourceId == nodeId);
    // O(E) where E = total number of edges
}

// With edge sets (fast):
public IEnumerable<GraphEdge<T>> GetOutgoingEdges(string nodeId)
{
    // Direct lookup of edge IDs
    var edgeIds = _outgoingEdges[nodeId];
    return edgeIds.Select(id => _edges[id]);
    // O(1) lookup + O(k) where k = edges from this node
}
```

#### AC 1.3: Refactor KnowledgeGraph

**Goal:** Update `KnowledgeGraph<T>` to use dependency injection for storage.

**Step 1: Modify KnowledgeGraph class**

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Abstractions;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// In-memory knowledge graph for storing and querying entity relationships.
/// </summary>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// A knowledge graph stores entities (nodes) and their relationships (edges) to enable structured information retrieval.
/// This implementation delegates storage to an IGraphStore implementation while providing graph algorithms and querying.
/// </para>
/// <para><b>For Beginners:</b> A knowledge graph is like a map of how information connects together.
///
/// Imagine Wikipedia as a graph:
/// - Each article is a node (Albert Einstein, Physics, Germany, etc.)
/// - Links between articles are edges (Einstein STUDIED Physics, Einstein BORN_IN Germany)
/// - You can traverse the graph to find related information
///
/// This class lets you:
/// 1. Add entities and relationships
/// 2. Find connections between entities
/// 3. Traverse the graph to discover related information
/// 4. Query based on entity types or relationships
///
/// For example, to answer "Who worked at Princeton?":
/// 1. Find all edges with type "WORKED_AT"
/// 2. Filter for target = "Princeton University"
/// 3. Return the source entities (people who worked there)
/// </para>
/// </remarks>
public class KnowledgeGraph<T>
{
    private readonly IGraphStore<T> _store;

    /// <summary>
    /// Gets the total number of nodes in the graph.
    /// </summary>
    public int NodeCount => _store.NodeCount;

    /// <summary>
    /// Gets the total number of edges in the graph.
    /// </summary>
    public int EdgeCount => _store.EdgeCount;

    /// <summary>
    /// Initializes a new instance of the KnowledgeGraph class.
    /// </summary>
    /// <param name="store">The storage backend to use for graph data.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new knowledge graph with a specific storage backend.
    ///
    /// Example:
    /// <code>
    /// // Use in-memory storage (fast, but data lost on restart)
    /// var memoryStore = new MemoryGraphStore&lt;float&gt;();
    /// var graph1 = new KnowledgeGraph&lt;float&gt;(memoryStore);
    ///
    /// // Use file-based storage (slower, but data persists)
    /// var fileStore = new FileGraphStore&lt;float&gt;("./data/graph");
    /// var graph2 = new KnowledgeGraph&lt;float&gt;(fileStore);
    ///
    /// // Both graphs work the same way, just different storage!
    /// graph1.AddNode(new GraphNode&lt;float&gt;("person1", "PERSON"));
    /// graph2.AddNode(new GraphNode&lt;float&gt;("person1", "PERSON"));
    /// </code>
    /// </para>
    /// </remarks>
    public KnowledgeGraph(IGraphStore<T> store)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
    }

    /// <summary>
    /// Adds a node to the graph or updates it if it already exists.
    /// </summary>
    /// <param name="node">The node to add.</param>
    public void AddNode(GraphNode<T> node)
    {
        _store.AddNode(node);
    }

    /// <summary>
    /// Adds an edge to the graph.
    /// </summary>
    /// <param name="edge">The edge to add.</param>
    /// <exception cref="InvalidOperationException">Thrown when source or target nodes don't exist.</exception>
    public void AddEdge(GraphEdge<T> edge)
    {
        _store.AddEdge(edge);
    }

    /// <summary>
    /// Gets a node by its ID.
    /// </summary>
    /// <param name="nodeId">The node ID.</param>
    /// <returns>The node, or null if not found.</returns>
    public GraphNode<T>? GetNode(string nodeId)
    {
        return _store.GetNode(nodeId);
    }

    /// <summary>
    /// Gets all nodes with a specific label.
    /// </summary>
    /// <param name="label">The node label to filter by.</param>
    /// <returns>Collection of nodes with the specified label.</returns>
    public IEnumerable<GraphNode<T>> GetNodesByLabel(string label)
    {
        return _store.GetAllNodes().Where(node => node.Label == label);
    }

    /// <summary>
    /// Gets all outgoing edges from a node.
    /// </summary>
    /// <param name="nodeId">The source node ID.</param>
    /// <returns>Collection of outgoing edges.</returns>
    public IEnumerable<GraphEdge<T>> GetOutgoingEdges(string nodeId)
    {
        return _store.GetOutgoingEdges(nodeId);
    }

    /// <summary>
    /// Gets all incoming edges to a node.
    /// </summary>
    /// <param name="nodeId">The target node ID.</param>
    /// <returns>Collection of incoming edges.</returns>
    public IEnumerable<GraphEdge<T>> GetIncomingEdges(string nodeId)
    {
        return _store.GetIncomingEdges(nodeId);
    }

    /// <summary>
    /// Gets all neighbors of a node (nodes connected by outgoing edges).
    /// </summary>
    /// <param name="nodeId">The node ID.</param>
    /// <returns>Collection of neighbor nodes.</returns>
    public IEnumerable<GraphNode<T>> GetNeighbors(string nodeId)
    {
        var edges = GetOutgoingEdges(nodeId);
        foreach (var edge in edges)
        {
            var target = _store.GetNode(edge.TargetId);
            if (target != null)
                yield return target;
        }
    }

    /// <summary>
    /// Performs breadth-first search traversal starting from a node.
    /// </summary>
    /// <param name="startNodeId">The starting node ID.</param>
    /// <param name="maxDepth">Maximum traversal depth (default: unlimited).</param>
    /// <returns>Collection of nodes in BFS order.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> BFS explores the graph level by level.
    ///
    /// Think of it like ripples in a pond:
    /// - Drop a stone (starting node)
    /// - First ripple: immediate neighbors (depth 1)
    /// - Second ripple: neighbors of neighbors (depth 2)
    /// - Third ripple: neighbors of neighbors of neighbors (depth 3)
    ///
    /// Example:
    /// Starting from "Albert Einstein", maxDepth=2:
    ///
    /// Depth 0: Einstein
    /// Depth 1: Princeton, Physics, Germany (neighbors of Einstein)
    /// Depth 2: USA (neighbor of Princeton), Mathematics (neighbor of Physics)
    ///
    /// Returns all nodes in order: Einstein, Princeton, Physics, Germany, USA, Mathematics
    /// </para>
    /// </remarks>
    public IEnumerable<GraphNode<T>> BreadthFirstTraversal(string startNodeId, int maxDepth = int.MaxValue)
    {
        var startNode = _store.GetNode(startNodeId);
        if (startNode == null)
            yield break;

        var visited = new HashSet<string>();
        var queue = new Queue<(string nodeId, int depth)>();
        queue.Enqueue((startNodeId, 0));
        visited.Add(startNodeId);

        while (queue.Count > 0)
        {
            var (nodeId, depth) = queue.Dequeue();
            var node = _store.GetNode(nodeId);
            if (node != null)
                yield return node;

            if (depth >= maxDepth)
                continue;

            foreach (var edge in _store.GetOutgoingEdges(nodeId))
            {
                if (!visited.Contains(edge.TargetId))
                {
                    visited.Add(edge.TargetId);
                    queue.Enqueue((edge.TargetId, depth + 1));
                }
            }
        }
    }

    /// <summary>
    /// Finds the shortest path between two nodes using BFS.
    /// </summary>
    /// <param name="startNodeId">The starting node ID.</param>
    /// <param name="endNodeId">The target node ID.</param>
    /// <returns>List of node IDs representing the path, or empty if no path exists.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Finds the shortest connection between two entities.
    ///
    /// Like finding the shortest route on a map or the "Six Degrees of Kevin Bacon":
    /// - Start: Kevin Bacon
    /// - End: Random Actor
    /// - Result: [Kevin Bacon, Movie A, Actor B, Movie C, Random Actor]
    ///
    /// The algorithm explores all paths simultaneously, so it finds the shortest.
    ///
    /// Example:
    /// Start: "Albert Einstein"
    /// End: "USA"
    ///
    /// Possible paths:
    /// 1. Einstein -> Princeton -> USA (length: 2)
    /// 2. Einstein -> Germany -> Europe -> USA (length: 3)
    ///
    /// Returns: ["einstein_person", "princeton_org", "usa_country"] (shortest)
    /// </para>
    /// </remarks>
    public List<string> FindShortestPath(string startNodeId, string endNodeId)
    {
        if (_store.GetNode(startNodeId) == null || _store.GetNode(endNodeId) == null)
            return new List<string>();

        var visited = new HashSet<string>();
        var parent = new Dictionary<string, string>();
        var queue = new Queue<string>();

        queue.Enqueue(startNodeId);
        visited.Add(startNodeId);

        while (queue.Count > 0)
        {
            var nodeId = queue.Dequeue();

            if (nodeId == endNodeId)
            {
                // Reconstruct path
                var path = new List<string>();
                var current = endNodeId;
                while (current != startNodeId)
                {
                    path.Add(current);
                    current = parent[current];
                }
                path.Add(startNodeId);
                path.Reverse();
                return path;
            }

            foreach (var edge in _store.GetOutgoingEdges(nodeId))
            {
                if (!visited.Contains(edge.TargetId))
                {
                    visited.Add(edge.TargetId);
                    parent[edge.TargetId] = nodeId;
                    queue.Enqueue(edge.TargetId);
                }
            }
        }

        return new List<string>(); // No path found
    }

    /// <summary>
    /// Finds nodes related to a query by entity name or property matching.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="topK">Maximum number of results to return.</param>
    /// <returns>Collection of matching nodes.</returns>
    public IEnumerable<GraphNode<T>> FindRelatedNodes(string query, int topK = 10)
    {
        var queryLower = query.ToLowerInvariant();

        return _store.GetAllNodes()
            .Where(node =>
            {
                var name = node.GetProperty<string>("name") ?? node.Id;
                return name.ToLowerInvariant().Contains(queryLower) ||
                       node.Label.ToLowerInvariant().Contains(queryLower);
            })
            .Take(topK);
    }

    /// <summary>
    /// Clears all nodes and edges from the graph.
    /// </summary>
    public void Clear()
    {
        _store.Clear();
    }

    /// <summary>
    /// Gets all nodes in the graph.
    /// </summary>
    /// <returns>Collection of all nodes.</returns>
    public IEnumerable<GraphNode<T>> GetAllNodes()
    {
        return _store.GetAllNodes();
    }

    /// <summary>
    /// Gets all edges in the graph.
    /// </summary>
    /// <returns>Collection of all edges.</returns>
    public IEnumerable<GraphEdge<T>> GetAllEdges()
    {
        // This requires GetAllEdges in IGraphStore, or we can enumerate outgoing edges
        // For now, let's enumerate all nodes' outgoing edges
        var allEdges = new HashSet<string>(); // Use HashSet to avoid duplicates
        var edges = new List<GraphEdge<T>>();

        foreach (var node in _store.GetAllNodes())
        {
            foreach (var edge in _store.GetOutgoingEdges(node.Id))
            {
                if (allEdges.Add(edge.Id)) // Returns false if already in set
                    edges.Add(edge);
            }
        }

        return edges;
    }
}
```

**Key changes:**

1. **Constructor now requires `IGraphStore<T>`:** Dependency injection
2. **All storage operations delegated:** `_store.AddNode(...)` instead of `_nodes[...] = ...`
3. **Graph algorithms unchanged:** BFS, shortest path, etc. still work the same way
4. **No direct storage access:** KnowledgeGraph doesn't know HOW data is stored, just that it can call the interface methods

#### AC 1.4: Update Unit Tests

**Goal:** Ensure all existing tests still pass after refactoring.

**Step 1: Update test setup**

```csharp
public class KnowledgeGraphTests
{
    [Fact]
    public void AddNode_ValidNode_Success()
    {
        // Arrange - Use MemoryGraphStore as the storage backend
        var store = new MemoryGraphStore<float>();
        var graph = new KnowledgeGraph<float>(store);

        var node = new GraphNode<float>("person1", "PERSON");
        node.SetProperty("name", "Albert Einstein");

        // Act
        graph.AddNode(node);
        var retrieved = graph.GetNode("person1");

        // Assert
        Assert.NotNull(retrieved);
        Assert.Equal("person1", retrieved.Id);
        Assert.Equal("PERSON", retrieved.Label);
        Assert.Equal("Albert Einstein", retrieved.GetProperty<string>("name"));
    }

    [Fact]
    public void AddEdge_ValidEdge_Success()
    {
        // Arrange
        var store = new MemoryGraphStore<float>();
        var graph = new KnowledgeGraph<float>(store);

        var person = new GraphNode<float>("person1", "PERSON");
        var org = new GraphNode<float>("org1", "ORGANIZATION");
        graph.AddNode(person);
        graph.AddNode(org);

        var edge = new GraphEdge<float>("person1", "org1", "WORKED_AT");

        // Act
        graph.AddEdge(edge);
        var outgoing = graph.GetOutgoingEdges("person1").ToList();

        // Assert
        Assert.Single(outgoing);
        Assert.Equal("WORKED_AT", outgoing[0].RelationType);
        Assert.Equal("org1", outgoing[0].TargetId);
    }
}
```

**Step 2: Create tests for MemoryGraphStore**

```csharp
public class MemoryGraphStoreTests
{
    [Fact]
    public void AddNode_NewNode_Success()
    {
        // Arrange
        var store = new MemoryGraphStore<float>();
        var node = new GraphNode<float>("node1", "LABEL");

        // Act
        store.AddNode(node);
        var retrieved = store.GetNode("node1");

        // Assert
        Assert.NotNull(retrieved);
        Assert.Equal("node1", retrieved.Id);
        Assert.Equal(1, store.NodeCount);
    }

    [Fact]
    public void RemoveNode_ExistingNode_RemovesConnectedEdges()
    {
        // Arrange
        var store = new MemoryGraphStore<float>();

        var node1 = new GraphNode<float>("node1", "LABEL");
        var node2 = new GraphNode<float>("node2", "LABEL");
        store.AddNode(node1);
        store.AddNode(node2);

        var edge = new GraphEdge<float>("node1", "node2", "RELATION");
        store.AddEdge(edge);

        // Act
        var removed = store.RemoveNode("node1");

        // Assert
        Assert.True(removed);
        Assert.Equal(1, store.NodeCount); // Only node2 remains
        Assert.Equal(0, store.EdgeCount); // Edge was removed too
    }

    [Fact]
    public void GetOutgoingEdges_MultipleEdges_ReturnsAll()
    {
        // Arrange
        var store = new MemoryGraphStore<float>();

        var node1 = new GraphNode<float>("node1", "LABEL");
        var node2 = new GraphNode<float>("node2", "LABEL");
        var node3 = new GraphNode<float>("node3", "LABEL");
        store.AddNode(node1);
        store.AddNode(node2);
        store.AddNode(node3);

        store.AddEdge(new GraphEdge<float>("node1", "node2", "REL1"));
        store.AddEdge(new GraphEdge<float>("node1", "node3", "REL2"));

        // Act
        var outgoing = store.GetOutgoingEdges("node1").ToList();

        // Assert
        Assert.Equal(2, outgoing.Count);
    }
}
```

### Phase 2: Implement Persistent On-Disk Store

#### AC 2.1: FileGraphStore Scaffolding

**Goal:** Create the basic file structure and initialization.

**Step 1: Create FileGraphStore class**

Location: `src/RetrievalAugmentedGeneration/Graph/FileGraphStore.cs`

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Abstractions;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// File-based graph store that persists nodes and edges to disk.
/// </summary>
/// <remarks>
/// <para>
/// FileGraphStore provides persistent graph storage using files and B-Tree indexes.
/// All data survives application restarts and can handle large graphs.
/// </para>
/// <para><b>For Beginners:</b> This stores the graph on your hard drive using files.
///
/// Think of it like a filing cabinet:
/// - Nodes stored in folders (nodes.dat)
/// - Edges stored in another folder (edges.dat)
/// - Index cards for fast lookup (B-Tree indexes)
/// - Even if you close the cabinet and come back tomorrow, everything is still there
///
/// Best for:
/// - Production systems requiring persistence
/// - Large graphs (millions of nodes)
/// - Multi-session applications
///
/// Trade-offs:
/// - Slower than memory (disk I/O)
/// - Requires disk space
/// - More complex implementation
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
public class FileGraphStore<T> : IGraphStore<T>
{
    private readonly string _directoryPath;
    private readonly string _nodesFilePath;
    private readonly string _edgesFilePath;

    private readonly BTreeIndex _nodeIndex;
    private readonly BTreeIndex _edgeIndex;

    private int _nodeCount;
    private int _edgeCount;

    // Edge relationship tracking
    private readonly Dictionary<string, List<string>> _outgoingEdges;
    private readonly Dictionary<string, List<string>> _incomingEdges;

    /// <summary>
    /// Gets the total number of nodes in the graph.
    /// </summary>
    public int NodeCount => _nodeCount;

    /// <summary>
    /// Gets the total number of edges in the graph.
    /// </summary>
    public int EdgeCount => _edgeCount;

    /// <summary>
    /// Initializes a new instance of the FileGraphStore class.
    /// </summary>
    /// <param name="directoryPath">The directory where all data files will be stored.</param>
    public FileGraphStore(string directoryPath)
    {
        if (string.IsNullOrWhiteSpace(directoryPath))
            throw new ArgumentException("Directory path cannot be null or empty", nameof(directoryPath));

        _directoryPath = directoryPath;

        // Ensure directory exists
        Directory.CreateDirectory(_directoryPath);

        // Define file paths
        _nodesFilePath = Path.Combine(_directoryPath, "nodes.dat");
        _edgesFilePath = Path.Combine(_directoryPath, "edges.dat");

        var nodeIndexPath = Path.Combine(_directoryPath, "node_index.db");
        var edgeIndexPath = Path.Combine(_directoryPath, "edge_index.db");

        // Initialize B-Tree indexes
        _nodeIndex = new BTreeIndex(nodeIndexPath);
        _edgeIndex = new BTreeIndex(edgeIndexPath);

        // Initialize edge tracking (in-memory for now)
        // TODO: Persist these to files as well
        _outgoingEdges = new Dictionary<string, List<string>>();
        _incomingEdges = new Dictionary<string, List<string>>();

        // Load existing data
        LoadExistingData();
    }

    /// <summary>
    /// Loads existing graph data from files on startup.
    /// </summary>
    private void LoadExistingData()
    {
        // Count nodes and edges from indexes
        _nodeCount = _nodeIndex.Count;
        _edgeCount = _edgeIndex.Count;

        // Load edge relationships
        // For now, we'll rebuild these by reading all edges
        // TODO: Persist edge relationships to avoid this overhead
        if (_edgeCount > 0)
        {
            RebuildEdgeRelationships();
        }
    }

    /// <summary>
    /// Rebuilds the edge relationship dictionaries by reading all edges.
    /// </summary>
    private void RebuildEdgeRelationships()
    {
        // Get all edge IDs from index
        var edgeIds = _edgeIndex.GetAllKeys();

        foreach (var edgeId in edgeIds)
        {
            var edge = GetEdge(edgeId);
            if (edge != null)
            {
                // Add to outgoing edges
                if (!_outgoingEdges.ContainsKey(edge.SourceId))
                    _outgoingEdges[edge.SourceId] = new List<string>();
                _outgoingEdges[edge.SourceId].Add(edgeId);

                // Add to incoming edges
                if (!_incomingEdges.ContainsKey(edge.TargetId))
                    _incomingEdges[edge.TargetId] = new List<string>();
                _incomingEdges[edge.TargetId].Add(edgeId);
            }
        }
    }

    // Implementation continues in next sections...
}
```

**Note on B-Tree Index:**

The `BTreeIndex` class from Issue #305's document store implementation should be reused here. It should have these additional methods:

```csharp
public class BTreeIndex
{
    // Existing methods...
    public void Add(string key, long offset);
    public long? Get(string key);
    public bool Remove(string key);

    // Additional methods needed for graph store
    public int Count { get; } // Total number of keys
    public IEnumerable<string> GetAllKeys(); // For rebuilding relationships
}
```

#### AC 2.2: Implement B-Tree for Indexing

**Goal:** Build a file-based B-Tree to map string keys to file offsets.

This is a complex task. For a junior developer guide, I'll provide a simplified B-Tree implementation. A production system would use a more sophisticated implementation.

**Simple B-Tree Implementation:**

Location: `src/RetrievalAugmentedGeneration/DocumentStores/Helpers/BTreeIndex.cs`

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores.Helpers;

/// <summary>
/// Simplified B-Tree index for mapping string keys to file offsets.
/// </summary>
/// <remarks>
/// <para>
/// This is a simplified implementation that stores the B-Tree as a JSON dictionary.
/// For production use, a proper B-Tree with node-based storage would be more efficient.
/// </para>
/// <para><b>For Beginners:</b> This is like an index at the back of a book.
///
/// Instead of reading every page to find "Einstein", you look in the index:
/// "Einstein: page 42"
///
/// Here, instead of scanning the entire file to find a document:
/// index["doc123"] = 10240 (byte offset in file)
///
/// Jump directly to byte 10240 and read the document. Much faster!
///
/// Why B-Tree?
/// - Fast lookups: O(log N) instead of O(N)
/// - Sorted keys: Can iterate in order
/// - Efficient on disk: Minimizes disk reads
///
/// Our simplified version:
/// - Uses JSON for easy debugging
/// - In-memory dictionary for speed
/// - Persisted to disk for durability
///
/// Production B-Tree would:
/// - Store nodes in separate file blocks
/// - Support very large indexes (> RAM)
/// - Use binary format (faster than JSON)
/// </para>
/// </remarks>
public class BTreeIndex
{
    private readonly string _filePath;
    private Dictionary<string, long> _index;

    /// <summary>
    /// Gets the total number of keys in the index.
    /// </summary>
    public int Count => _index.Count;

    /// <summary>
    /// Initializes a new instance of the BTreeIndex class.
    /// </summary>
    /// <param name="filePath">Path to the index file.</param>
    public BTreeIndex(string filePath)
    {
        _filePath = filePath;
        _index = new Dictionary<string, long>();

        // Load existing index if file exists
        if (File.Exists(_filePath))
        {
            Load();
        }
    }

    /// <summary>
    /// Adds or updates a key-value pair in the index.
    /// </summary>
    /// <param name="key">The key to add/update.</param>
    /// <param name="offset">The file offset value.</param>
    public void Add(string key, long offset)
    {
        _index[key] = offset;
        Save();
    }

    /// <summary>
    /// Gets the offset for a given key.
    /// </summary>
    /// <param name="key">The key to look up.</param>
    /// <returns>The offset if found; otherwise, null.</returns>
    public long? Get(string key)
    {
        return _index.TryGetValue(key, out var offset) ? offset : null;
    }

    /// <summary>
    /// Removes a key from the index.
    /// </summary>
    /// <param name="key">The key to remove.</param>
    /// <returns>True if removed; false if not found.</returns>
    public bool Remove(string key)
    {
        var removed = _index.Remove(key);
        if (removed)
            Save();
        return removed;
    }

    /// <summary>
    /// Gets all keys in the index.
    /// </summary>
    public IEnumerable<string> GetAllKeys()
    {
        return _index.Keys.ToList();
    }

    /// <summary>
    /// Clears all entries from the index.
    /// </summary>
    public void Clear()
    {
        _index.Clear();
        if (File.Exists(_filePath))
            File.Delete(_filePath);
    }

    /// <summary>
    /// Saves the index to disk.
    /// </summary>
    private void Save()
    {
        var json = JsonSerializer.Serialize(_index, new JsonSerializerOptions
        {
            WriteIndented = true
        });
        File.WriteAllText(_filePath, json);
    }

    /// <summary>
    /// Loads the index from disk.
    /// </summary>
    private void Load()
    {
        var json = File.ReadAllText(_filePath);
        _index = JsonSerializer.Deserialize<Dictionary<string, long>>(json) ?? new Dictionary<string, long>();
    }
}
```

**Why this simplified approach?**

For a junior developer, implementing a full B-Tree with nodes, splits, and merges is very complex. This simplified version:

- Uses a dictionary (hash map) for O(1) lookups (even faster than B-Tree's O(log N))
- Stores as JSON for easy debugging
- Provides all needed functionality
- Can be optimized later

**When to use a real B-Tree:**

- Index too large for RAM (> gigabytes)
- Need sorted iteration
- Want to minimize disk I/O (B-Tree nodes map to disk blocks)

#### AC 2.3: Implement Persistent CRUD Operations

**Goal:** Store and retrieve nodes and edges from files.

**Step 1: Serialization helpers**

```csharp
/// <summary>
/// Serializes a GraphNode to binary format.
/// </summary>
private byte[] SerializeNode(GraphNode<T> node)
{
    using var memoryStream = new MemoryStream();
    using var writer = new BinaryWriter(memoryStream);

    // Write size placeholder
    writer.Write(0);

    // Write node data
    writer.Write(node.Id);
    writer.Write(node.Label);

    // Write properties
    writer.Write(node.Properties.Count);
    foreach (var kvp in node.Properties)
    {
        writer.Write(kvp.Key);

        // Type marker and value
        if (kvp.Value is string strValue)
        {
            writer.Write((byte)1);
            writer.Write(strValue);
        }
        else if (kvp.Value is int intValue)
        {
            writer.Write((byte)2);
            writer.Write(intValue);
        }
        else if (kvp.Value is double doubleValue)
        {
            writer.Write((byte)3);
            writer.Write(doubleValue);
        }
        else if (kvp.Value is bool boolValue)
        {
            writer.Write((byte)4);
            writer.Write(boolValue);
        }
        // Add more types as needed
    }

    // Write timestamps
    writer.Write(node.CreatedAt.ToBinary());
    writer.Write(node.UpdatedAt.ToBinary());

    // Write size at beginning
    var totalSize = (int)memoryStream.Length;
    memoryStream.Seek(0, SeekOrigin.Begin);
    writer.Write(totalSize);

    return memoryStream.ToArray();
}

/// <summary>
/// Deserializes a GraphNode from binary format.
/// </summary>
private GraphNode<T> DeserializeNode(byte[] data)
{
    using var memoryStream = new MemoryStream(data);
    using var reader = new BinaryReader(memoryStream);

    // Read size (skip it, we already know the data length)
    var size = reader.ReadInt32();

    // Read node data
    var id = reader.ReadString();
    var label = reader.ReadString();

    var node = new GraphNode<T>(id, label);

    // Read properties
    var propertyCount = reader.ReadInt32();
    for (int i = 0; i < propertyCount; i++)
    {
        var key = reader.ReadString();
        var typeMarker = reader.ReadByte();

        object value = typeMarker switch
        {
            1 => reader.ReadString(),
            2 => reader.ReadInt32(),
            3 => reader.ReadDouble(),
            4 => reader.ReadBoolean(),
            _ => throw new InvalidOperationException($"Unknown type marker: {typeMarker}")
        };

        node.Properties[key] = value;
    }

    // Read timestamps
    node.CreatedAt = DateTime.FromBinary(reader.ReadInt64());
    node.UpdatedAt = DateTime.FromBinary(reader.ReadInt64());

    return node;
}
```

**Similar methods for edges:**

```csharp
private byte[] SerializeEdge(GraphEdge<T> edge) { /* Similar to SerializeNode */ }
private GraphEdge<T> DeserializeEdge(byte[] data) { /* Similar to DeserializeNode */ }
```

**Step 2: Implement AddNode**

```csharp
public void AddNode(GraphNode<T> node)
{
    if (node == null)
        throw new ArgumentNullException(nameof(node));

    // Serialize node
    var nodeBytes = SerializeNode(node);

    // Append to nodes.dat and get offset
    long offset;
    using (var fileStream = new FileStream(_nodesFilePath, FileMode.Append, FileAccess.Write))
    {
        offset = fileStream.Position;
        fileStream.Write(nodeBytes, 0, nodeBytes.Length);
    }

    // Add to index
    _nodeIndex.Add(node.Id, offset);

    // Initialize edge sets for this node
    if (!_outgoingEdges.ContainsKey(node.Id))
        _outgoingEdges[node.Id] = new List<string>();
    if (!_incomingEdges.ContainsKey(node.Id))
        _incomingEdges[node.Id] = new List<string>();

    _nodeCount++;
}
```

**Step 3: Implement GetNode**

```csharp
public GraphNode<T>? GetNode(string nodeId)
{
    // Look up offset in index
    var offset = _nodeIndex.Get(nodeId);
    if (!offset.HasValue)
        return null;

    // Read from file
    using var fileStream = new FileStream(_nodesFilePath, FileMode.Open, FileAccess.Read, FileShare.Read);
    fileStream.Seek(offset.Value, SeekOrigin.Begin);

    // Read size prefix
    var sizeBytes = new byte[4];
    fileStream.Read(sizeBytes, 0, 4);
    var size = BitConverter.ToInt32(sizeBytes, 0);

    // Read entire node
    var nodeBytes = new byte[size];
    fileStream.Seek(offset.Value, SeekOrigin.Begin);
    fileStream.Read(nodeBytes, 0, size);

    return DeserializeNode(nodeBytes);
}
```

**Step 4: Implement RemoveNode**

```csharp
public bool RemoveNode(string nodeId)
{
    // Check if exists
    var offset = _nodeIndex.Get(nodeId);
    if (!offset.HasValue)
        return false;

    // Remove from index (lazy deletion - data stays in file)
    _nodeIndex.Remove(nodeId);

    // Remove all connected edges
    if (_outgoingEdges.TryGetValue(nodeId, out var outgoing))
    {
        foreach (var edgeId in outgoing.ToList())
            RemoveEdge(edgeId);
    }

    if (_incomingEdges.TryGetValue(nodeId, out var incoming))
    {
        foreach (var edgeId in incoming.ToList())
            RemoveEdge(edgeId);
    }

    // Clean up edge sets
    _outgoingEdges.Remove(nodeId);
    _incomingEdges.Remove(nodeId);

    _nodeCount--;
    return true;
}
```

**Step 5: Implement edge operations (similar pattern)**

```csharp
public void AddEdge(GraphEdge<T> edge)
{
    // Validate nodes exist
    if (GetNode(edge.SourceId) == null)
        throw new InvalidOperationException($"Source node '{edge.SourceId}' does not exist");
    if (GetNode(edge.TargetId) == null)
        throw new InvalidOperationException($"Target node '{edge.TargetId}' does not exist");

    // Serialize and append
    var edgeBytes = SerializeEdge(edge);

    long offset;
    using (var fileStream = new FileStream(_edgesFilePath, FileMode.Append, FileAccess.Write))
    {
        offset = fileStream.Position;
        fileStream.Write(edgeBytes, 0, edgeBytes.Length);
    }

    // Add to index
    _edgeIndex.Add(edge.Id, offset);

    // Update edge relationships
    if (!_outgoingEdges.ContainsKey(edge.SourceId))
        _outgoingEdges[edge.SourceId] = new List<string>();
    _outgoingEdges[edge.SourceId].Add(edge.Id);

    if (!_incomingEdges.ContainsKey(edge.TargetId))
        _incomingEdges[edge.TargetId] = new List<string>();
    _incomingEdges[edge.TargetId].Add(edge.Id);

    _edgeCount++;
}

public GraphEdge<T>? GetEdge(string edgeId)
{
    var offset = _edgeIndex.Get(edgeId);
    if (!offset.HasValue)
        return null;

    using var fileStream = new FileStream(_edgesFilePath, FileMode.Open, FileAccess.Read, FileShare.Read);
    fileStream.Seek(offset.Value, SeekOrigin.Begin);

    var sizeBytes = new byte[4];
    fileStream.Read(sizeBytes, 0, 4);
    var size = BitConverter.ToInt32(sizeBytes, 0);

    var edgeBytes = new byte[size];
    fileStream.Seek(offset.Value, SeekOrigin.Begin);
    fileStream.Read(edgeBytes, 0, size);

    return DeserializeEdge(edgeBytes);
}

public bool RemoveEdge(string edgeId)
{
    var edge = GetEdge(edgeId);
    if (edge == null)
        return false;

    _edgeIndex.Remove(edgeId);

    // Remove from relationship tracking
    if (_outgoingEdges.TryGetValue(edge.SourceId, out var outgoing))
        outgoing.Remove(edgeId);
    if (_incomingEdges.TryGetValue(edge.TargetId, out var incoming))
        incoming.Remove(edgeId);

    _edgeCount--;
    return true;
}

public IEnumerable<GraphEdge<T>> GetOutgoingEdges(string nodeId)
{
    if (!_outgoingEdges.TryGetValue(nodeId, out var edgeIds))
        return Enumerable.Empty<GraphEdge<T>>();

    var edges = new List<GraphEdge<T>>();
    foreach (var edgeId in edgeIds)
    {
        var edge = GetEdge(edgeId);
        if (edge != null)
            edges.Add(edge);
    }

    return edges;
}

public IEnumerable<GraphEdge<T>> GetIncomingEdges(string nodeId)
{
    if (!_incomingEdges.TryGetValue(nodeId, out var edgeIds))
        return Enumerable.Empty<GraphEdge<T>>();

    var edges = new List<GraphEdge<T>>();
    foreach (var edgeId in edgeIds)
    {
        var edge = GetEdge(edgeId);
        if (edge != null)
            edges.Add(edge);
    }

    return edges;
}

public IEnumerable<GraphNode<T>> GetAllNodes()
{
    var nodes = new List<GraphNode<T>>();
    var nodeIds = _nodeIndex.GetAllKeys();

    foreach (var nodeId in nodeIds)
    {
        var node = GetNode(nodeId);
        if (node != null)
            nodes.Add(node);
    }

    return nodes;
}

public void Clear()
{
    _nodeIndex.Clear();
    _edgeIndex.Clear();
    _outgoingEdges.Clear();
    _incomingEdges.Clear();

    if (File.Exists(_nodesFilePath))
        File.Delete(_nodesFilePath);
    if (File.Exists(_edgesFilePath))
        File.Delete(_edgesFilePath);

    _nodeCount = 0;
    _edgeCount = 0;
}
```

---

## Testing Strategy

### Unit Tests

**Test File:** `tests/AiDotNet.Tests/RetrievalAugmentedGeneration/Graph/FileGraphStoreTests.cs`

```csharp
public class FileGraphStoreTests
{
    [Fact]
    public void AddNode_NewNode_Success()
    {
        var tempDir = CreateTempDirectory();
        var store = new FileGraphStore<float>(tempDir);

        var node = new GraphNode<float>("person1", "PERSON");
        node.SetProperty("name", "Alice");

        store.AddNode(node);
        var retrieved = store.GetNode("person1");

        Assert.NotNull(retrieved);
        Assert.Equal("person1", retrieved.Id);
        Assert.Equal("PERSON", retrieved.Label);
        Assert.Equal("Alice", retrieved.GetProperty<string>("name"));

        Cleanup(tempDir);
    }

    [Fact]
    public void Persistence_DataSurvivesRestart()
    {
        var tempDir = CreateTempDirectory();

        // First instance: Add data
        var store1 = new FileGraphStore<float>(tempDir);
        var node = new GraphNode<float>("person1", "PERSON");
        store1.AddNode(node);

        // Simulate restart
        var store2 = new FileGraphStore<float>(tempDir);
        var retrieved = store2.GetNode("person1");

        Assert.NotNull(retrieved);
        Assert.Equal("person1", retrieved.Id);
        Assert.Equal(1, store2.NodeCount);

        Cleanup(tempDir);
    }

    [Fact]
    public void AddEdge_ValidEdge_Success()
    {
        var tempDir = CreateTempDirectory();
        var store = new FileGraphStore<float>(tempDir);

        var person = new GraphNode<float>("person1", "PERSON");
        var org = new GraphNode<float>("org1", "ORGANIZATION");
        store.AddNode(person);
        store.AddNode(org);

        var edge = new GraphEdge<float>("person1", "org1", "WORKED_AT");
        store.AddEdge(edge);

        var outgoing = store.GetOutgoingEdges("person1").ToList();
        Assert.Single(outgoing);
        Assert.Equal("WORKED_AT", outgoing[0].RelationType);

        Cleanup(tempDir);
    }

    private string CreateTempDirectory()
    {
        return Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
    }

    private void Cleanup(string directory)
    {
        if (Directory.Exists(directory))
            Directory.Delete(directory, true);
    }
}
```

### Integration Tests

```csharp
[Fact]
public void CompleteWorkflow_AddTraverseRestart()
{
    var tempDir = CreateTempDirectory();

    // Build a small knowledge graph
    var store1 = new FileGraphStore<float>(tempDir);
    var graph1 = new KnowledgeGraph<float>(store1);

    // Add nodes
    var einstein = new GraphNode<float>("einstein", "PERSON");
    einstein.SetProperty("name", "Albert Einstein");
    graph1.AddNode(einstein);

    var princeton = new GraphNode<float>("princeton", "ORGANIZATION");
    princeton.SetProperty("name", "Princeton University");
    graph1.AddNode(princeton);

    var physics = new GraphNode<float>("physics", "FIELD");
    physics.SetProperty("name", "Physics");
    graph1.AddNode(physics);

    // Add edges
    graph1.AddEdge(new GraphEdge<float>("einstein", "princeton", "WORKED_AT"));
    graph1.AddEdge(new GraphEdge<float>("einstein", "physics", "STUDIED"));

    // Traverse
    var neighbors = graph1.GetNeighbors("einstein").ToList();
    Assert.Equal(2, neighbors.Count);

    // Restart
    var store2 = new FileGraphStore<float>(tempDir);
    var graph2 = new KnowledgeGraph<float>(store2);

    // Verify persistence
    Assert.Equal(3, graph2.NodeCount);
    Assert.Equal(2, graph2.EdgeCount);

    var neighbors2 = graph2.GetNeighbors("einstein").ToList();
    Assert.Equal(2, neighbors2.Count);

    Cleanup(tempDir);
}
```

---

## Common Pitfalls

### 1. Edge Relationship Persistence

**Problem:** `_outgoingEdges` and `_incomingEdges` are in-memory dictionaries. They're rebuilt on startup by reading all edges, which is slow for large graphs.

**Solution:**

Persist these to separate files:

```csharp
private void SaveEdgeRelationships()
{
    var outgoingPath = Path.Combine(_directoryPath, "outgoing_edges.json");
    var incomingPath = Path.Combine(_directoryPath, "incoming_edges.json");

    File.WriteAllText(outgoingPath, JsonSerializer.Serialize(_outgoingEdges));
    File.WriteAllText(incomingPath, JsonSerializer.Serialize(_incomingEdges));
}

private void LoadEdgeRelationships()
{
    var outgoingPath = Path.Combine(_directoryPath, "outgoing_edges.json");
    var incomingPath = Path.Combine(_directoryPath, "incoming_edges.json");

    if (File.Exists(outgoingPath))
        _outgoingEdges = JsonSerializer.Deserialize<Dictionary<string, List<string>>>(
            File.ReadAllText(outgoingPath)) ?? new Dictionary<string, List<string>>();

    if (File.Exists(incomingPath))
        _incomingEdges = JsonSerializer.Deserialize<Dictionary<string, List<string>>>(
            File.ReadAllText(incomingPath)) ?? new Dictionary<string, List<string>>();
}
```

### 2. Circular Dependencies in Deserialization

**Problem:** GraphNode and GraphEdge might reference each other, causing infinite loops.

**Solution:** Deserialize nodes and edges separately. Don't embed full node objects in edges - just store IDs.

### 3. File Corruption

**Problem:** Application crashes while writing, leaving partial data in files.

**Solution:**

Write-ahead logging or atomic writes:

```csharp
// Atomic write pattern
private void AtomicWriteNode(GraphNode<T> node, long offset)
{
    var tempFile = _nodesFilePath + ".tmp";
    var nodeBytes = SerializeNode(node);

    // Write to temp file
    using (var fs = new FileStream(tempFile, FileMode.Create, FileAccess.Write))
    {
        fs.Write(nodeBytes, 0, nodeBytes.Length);
        fs.Flush(true); // Flush to disk
    }

    // Append temp file contents to main file
    using (var source = new FileStream(tempFile, FileMode.Open, FileAccess.Read))
    using (var dest = new FileStream(_nodesFilePath, FileMode.Append, FileAccess.Write))
    {
        source.CopyTo(dest);
        dest.Flush(true);
    }

    File.Delete(tempFile);
}
```

### 4. Memory Leaks

**Problem:** File handles not disposed properly.

**Solution:** Always use `using` statements.

### 5. Concurrency Issues

**Problem:** Multiple threads or processes accessing files simultaneously.

**Solution:**

- Use `FileShare.Read` for read operations
- Use locks for write operations
- Consider file locking mechanisms

### 6. Large Graph Performance

**Problem:** Loading all node IDs into memory for `GetAllNodes()` is slow for millions of nodes.

**Solution:**

Implement pagination or streaming:

```csharp
public IEnumerable<GraphNode<T>> GetAllNodesStreaming()
{
    foreach (var nodeId in _nodeIndex.GetAllKeys())
    {
        var node = GetNode(nodeId);
        if (node != null)
            yield return node;
    }
}
```

---

## Resources

### Graph Theory

- [Graph Theory Tutorial](https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/)
- [Breadth-First Search](https://en.wikipedia.org/wiki/Breadth-first_search)
- [Shortest Path Algorithms](https://www.geeksforgeeks.org/shortest-path-algorithms/)

### Knowledge Graphs

- [Knowledge Graph Introduction](https://www.ontotext.com/knowledgehub/fundamentals/what-is-a-knowledge-graph/)
- [Building Knowledge Graphs](https://neo4j.com/developer/graph-data-modeling/)
- [Graph RAG Paper](https://arxiv.org/abs/2404.16130)

### Data Structures

- [B-Tree Tutorial](https://www.cs.usfca.edu/~galles/visualization/BTree.html)
- [B-Tree Wikipedia](https://en.wikipedia.org/wiki/B-tree)

### File I/O

- [C# File I/O Best Practices](https://docs.microsoft.com/en-us/dotnet/standard/io/)
- [BinaryReader/BinaryWriter](https://docs.microsoft.com/en-us/dotnet/api/system.io.binaryreader)

---

## Summary Checklist

Before submitting your PR, ensure:

- [ ] `IGraphStore<T>` interface created with all required methods
- [ ] `MemoryGraphStore<T>` implemented and all existing tests pass
- [ ] `KnowledgeGraph<T>` refactored to use dependency injection
- [ ] `FileGraphStore<T>` class created with proper file structure
- [ ] `BTreeIndex` implemented (simplified version acceptable)
- [ ] Node serialization/deserialization working correctly
- [ ] Edge serialization/deserialization working correctly
- [ ] `AddNode`, `GetNode`, `RemoveNode` implemented
- [ ] `AddEdge`, `GetEdge`, `RemoveEdge` implemented
- [ ] `GetOutgoingEdges` and `GetIncomingEdges` working
- [ ] Edge relationships persisted or efficiently rebuilt
- [ ] Unit tests for all store implementations
- [ ] Integration tests verify persistence across restarts
- [ ] Code coverage >= 90%
- [ ] Documentation includes beginner-friendly explanations
- [ ] File handles properly disposed
- [ ] No memory leaks or resource leaks

Good luck! Remember: Graph databases are complex. Start with the refactoring (Phase 1), ensure all tests pass, then tackle the file-based store (Phase 2). Test frequently and don't hesitate to ask for help!
