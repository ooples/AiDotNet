using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.RetrievalAugmentedGeneration.Graph;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for graph storage backends that manage nodes and edges.
/// </summary>
/// <remarks>
/// <para>
/// A graph store provides persistent or in-memory storage for knowledge graphs,
/// enabling efficient storage and retrieval of entities (nodes) and their relationships (edges).
/// Implementations can range from simple in-memory dictionaries to distributed graph databases.
/// </para>
/// <para><b>For Beginners:</b> A graph store is like a filing system for connected information.
///
/// Think of it like organizing a network of friends:
/// - Nodes are people (Alice, Bob, Charlie)
/// - Edges are relationships (Alice KNOWS Bob, Bob WORKS_WITH Charlie)
/// - The graph store remembers all these connections
///
/// Different implementations might:
/// - MemoryGraphStore: Keep everything in RAM (fast but lost when app closes)
/// - FileGraphStore: Save to disk (slower but survives restarts)
/// - Neo4jGraphStore: Use a professional graph database (production-scale)
///
/// This interface lets you swap storage backends without changing your code!
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector calculations (typically float or double).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("GraphStore")]
public interface IGraphStore<T>
{
    /// <summary>
    /// Gets the total number of nodes in the graph store.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how many entities (people, places, things)
    /// are stored in the graph.
    /// </para>
    /// </remarks>
    int NodeCount { get; }

    /// <summary>
    /// Gets the total number of edges in the graph store.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how many relationships/connections
    /// exist between entities in the graph.
    /// </para>
    /// </remarks>
    int EdgeCount { get; }

    /// <summary>
    /// Adds a node to the graph or updates it if it already exists.
    /// </summary>
    /// <param name="node">The node to add.</param>
    /// <remarks>
    /// <para>
    /// This method stores a node in the graph. If a node with the same ID already exists,
    /// it will be updated with the new data. The node is automatically indexed by its label
    /// for efficient label-based queries.
    /// </para>
    /// <para><b>For Beginners:</b> This adds a new entity to the graph.
    ///
    /// Like adding a person to a social network:
    /// - node.Id = "alice_001"
    /// - node.Label = "PERSON"
    /// - node.Properties = { "name": "Alice Smith", "age": 30 }
    ///
    /// If Alice already exists, her information gets updated.
    /// </para>
    /// </remarks>
    void AddNode(GraphNode<T> node);

    /// <summary>
    /// Adds an edge to the graph representing a relationship between two nodes.
    /// </summary>
    /// <param name="edge">The edge to add.</param>
    /// <remarks>
    /// <para>
    /// This method creates a relationship between two existing nodes. Both the source
    /// and target nodes must already exist in the graph, otherwise an exception is thrown.
    /// The edge is indexed for efficient traversal from both directions.
    /// </para>
    /// <para><b>For Beginners:</b> This adds a connection between two entities.
    ///
    /// Like saying "Alice knows Bob":
    /// - edge.SourceId = "alice_001"
    /// - edge.RelationType = "KNOWS"
    /// - edge.TargetId = "bob_002"
    /// - edge.Weight = 0.9 (how strong the relationship is)
    ///
    /// Both Alice and Bob must already be added as nodes first!
    /// </para>
    /// </remarks>
    void AddEdge(GraphEdge<T> edge);

    /// <summary>
    /// Retrieves a node by its unique identifier.
    /// </summary>
    /// <param name="nodeId">The unique identifier of the node.</param>
    /// <returns>The node if found; otherwise, null.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This gets a specific entity if you know its ID.
    ///
    /// Like asking: "Show me the person with ID 'alice_001'"
    /// </para>
    /// </remarks>
    GraphNode<T>? GetNode(string nodeId);

    /// <summary>
    /// Retrieves an edge by its unique identifier.
    /// </summary>
    /// <param name="edgeId">The unique identifier of the edge.</param>
    /// <returns>The edge if found; otherwise, null.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This gets a specific relationship if you know its ID.
    ///
    /// Edge IDs are usually auto-generated like: "alice_001_KNOWS_bob_002"
    /// </para>
    /// </remarks>
    GraphEdge<T>? GetEdge(string edgeId);

    /// <summary>
    /// Removes a node and all its connected edges from the graph.
    /// </summary>
    /// <param name="nodeId">The unique identifier of the node to remove.</param>
    /// <returns>True if the node was found and removed; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method removes a node and automatically cleans up all edges connected to it
    /// (both incoming and outgoing). This ensures the graph remains consistent.
    /// </para>
    /// <para><b>For Beginners:</b> This deletes an entity and all its connections.
    ///
    /// Like removing Alice from the network:
    /// - Alice's profile is deleted
    /// - All "Alice KNOWS Bob" relationships are deleted
    /// - All "Bob KNOWS Alice" relationships are deleted
    ///
    /// This keeps the graph clean - no broken connections!
    /// </para>
    /// </remarks>
    bool RemoveNode(string nodeId);

    /// <summary>
    /// Removes an edge from the graph.
    /// </summary>
    /// <param name="edgeId">The unique identifier of the edge to remove.</param>
    /// <returns>True if the edge was found and removed; otherwise, false.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This deletes a specific relationship.
    ///
    /// Like saying "Alice no longer knows Bob" - removes just that connection,
    /// but Alice and Bob still exist in the graph.
    /// </para>
    /// </remarks>
    bool RemoveEdge(string edgeId);

    /// <summary>
    /// Gets all outgoing edges from a specific node.
    /// </summary>
    /// <param name="nodeId">The source node ID.</param>
    /// <returns>Collection of outgoing edges from the node.</returns>
    /// <remarks>
    /// <para>
    /// Outgoing edges represent relationships where this node is the source.
    /// For example, if Alice KNOWS Bob, the "KNOWS" edge is outgoing from Alice.
    /// </para>
    /// <para><b>For Beginners:</b> This finds all relationships going OUT from an entity.
    ///
    /// If you ask for Alice's outgoing edges, you get:
    /// - Alice KNOWS Bob
    /// - Alice WORKS_AT CompanyX
    /// - Alice LIVES_IN Seattle
    ///
    /// These are things Alice does or has relationships with.
    /// </para>
    /// </remarks>
    IEnumerable<GraphEdge<T>> GetOutgoingEdges(string nodeId);

    /// <summary>
    /// Gets all incoming edges to a specific node.
    /// </summary>
    /// <param name="nodeId">The target node ID.</param>
    /// <returns>Collection of incoming edges to the node.</returns>
    /// <remarks>
    /// <para>
    /// Incoming edges represent relationships where this node is the target.
    /// For example, if Alice KNOWS Bob, the "KNOWS" edge is incoming to Bob.
    /// </para>
    /// <para><b>For Beginners:</b> This finds all relationships coming IN to an entity.
    ///
    /// If you ask for Bob's incoming edges, you get:
    /// - Alice KNOWS Bob
    /// - Charlie WORKS_WITH Bob
    /// - CompanyY EMPLOYS Bob
    ///
    /// These are relationships others have WITH Bob.
    /// </para>
    /// </remarks>
    IEnumerable<GraphEdge<T>> GetIncomingEdges(string nodeId);

    /// <summary>
    /// Gets all nodes with a specific label.
    /// </summary>
    /// <param name="label">The node label to filter by (e.g., "PERSON", "COMPANY", "LOCATION").</param>
    /// <returns>Collection of nodes with the specified label.</returns>
    /// <remarks>
    /// <para>
    /// Labels are used to categorize nodes by type. This enables efficient queries
    /// like "find all PERSON nodes" or "find all COMPANY nodes".
    /// </para>
    /// <para><b>For Beginners:</b> This finds all entities of a specific type.
    ///
    /// Like asking: "Show me all PERSON nodes"
    /// Returns: Alice, Bob, Charlie (all people in the graph)
    ///
    /// Or: "Show me all COMPANY nodes"
    /// Returns: Microsoft, Google, Amazon (all companies)
    ///
    /// Labels are like categories or tags for organizing your entities.
    /// </para>
    /// </remarks>
    IEnumerable<GraphNode<T>> GetNodesByLabel(string label);

    /// <summary>
    /// Gets all nodes currently stored in the graph.
    /// </summary>
    /// <returns>Collection of all nodes.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves every node without any filtering.
    /// Use with caution on large graphs as it may be memory-intensive.
    /// </para>
    /// <para><b>For Beginners:</b> This gets every single entity in the graph.
    ///
    /// Like asking: "Show me everyone and everything in the network"
    ///
    /// Warning: If you have millions of entities, this could be slow and use lots of memory!
    /// </para>
    /// </remarks>
    IEnumerable<GraphNode<T>> GetAllNodes();

    /// <summary>
    /// Gets all edges currently stored in the graph.
    /// </summary>
    /// <returns>Collection of all edges.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves every edge without any filtering.
    /// Use with caution on large graphs as it may be memory-intensive.
    /// </para>
    /// <para><b>For Beginners:</b> This gets every single relationship in the graph.
    ///
    /// Like asking: "Show me every connection between all entities"
    ///
    /// Warning: Large graphs can have millions of relationships!
    /// </para>
    /// </remarks>
    IEnumerable<GraphEdge<T>> GetAllEdges();

    /// <summary>
    /// Removes all nodes and edges from the graph.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears the entire graph, removing all data. For persistent stores,
    /// this may involve deleting files or database records. Use with extreme caution!
    /// </para>
    /// <para><b>For Beginners:</b> This deletes EVERYTHING from the graph.
    ///
    /// Like wiping the entire social network clean - all people and all connections gone!
    ///
    /// ⚠️ WARNING: This cannot be undone! Make backups first!
    /// </para>
    /// </remarks>
    void Clear();

    // Async methods for I/O-intensive operations

    /// <summary>
    /// Asynchronously adds a node to the graph or updates it if it already exists.
    /// </summary>
    /// <param name="node">The node to add.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    /// <remarks>
    /// <para>
    /// This is the async version of <see cref="AddNode"/>. Use this for file-based or
    /// database-backed stores to avoid blocking the thread during I/O operations.
    /// </para>
    /// <para><b>For Beginners:</b> This does the same as AddNode but doesn't block your app.
    ///
    /// When should you use async?
    /// - FileGraphStore: Yes! (writes to disk)
    /// - MemoryGraphStore: Optional (no I/O, but provided for consistency)
    /// - Database stores: Definitely! (network I/O)
    ///
    /// Example:
    /// ```csharp
    /// await store.AddNodeAsync(node);  // Non-blocking
    /// ```
    /// </para>
    /// </remarks>
    Task AddNodeAsync(GraphNode<T> node);

    /// <summary>
    /// Asynchronously adds an edge to the graph.
    /// </summary>
    /// <param name="edge">The edge to add.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    Task AddEdgeAsync(GraphEdge<T> edge);

    /// <summary>
    /// Asynchronously retrieves a node by its unique identifier.
    /// </summary>
    /// <param name="nodeId">The unique identifier of the node.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the node if found; otherwise, null.</returns>
    Task<GraphNode<T>?> GetNodeAsync(string nodeId);

    /// <summary>
    /// Asynchronously retrieves an edge by its unique identifier.
    /// </summary>
    /// <param name="edgeId">The unique identifier of the edge.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the edge if found; otherwise, null.</returns>
    Task<GraphEdge<T>?> GetEdgeAsync(string edgeId);

    /// <summary>
    /// Asynchronously removes a node and all its connected edges from the graph.
    /// </summary>
    /// <param name="nodeId">The unique identifier of the node to remove.</param>
    /// <returns>A task that represents the asynchronous operation. The task result is true if the node was found and removed; otherwise, false.</returns>
    Task<bool> RemoveNodeAsync(string nodeId);

    /// <summary>
    /// Asynchronously removes an edge from the graph.
    /// </summary>
    /// <param name="edgeId">The unique identifier of the edge to remove.</param>
    /// <returns>A task that represents the asynchronous operation. The task result is true if the edge was found and removed; otherwise, false.</returns>
    Task<bool> RemoveEdgeAsync(string edgeId);

    /// <summary>
    /// Asynchronously gets all outgoing edges from a specific node.
    /// </summary>
    /// <param name="nodeId">The source node ID.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the collection of outgoing edges.</returns>
    Task<IEnumerable<GraphEdge<T>>> GetOutgoingEdgesAsync(string nodeId);

    /// <summary>
    /// Asynchronously gets all incoming edges to a specific node.
    /// </summary>
    /// <param name="nodeId">The target node ID.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the collection of incoming edges.</returns>
    Task<IEnumerable<GraphEdge<T>>> GetIncomingEdgesAsync(string nodeId);

    /// <summary>
    /// Asynchronously gets all nodes with a specific label.
    /// </summary>
    /// <param name="label">The node label to filter by.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the collection of nodes with the specified label.</returns>
    Task<IEnumerable<GraphNode<T>>> GetNodesByLabelAsync(string label);

    /// <summary>
    /// Asynchronously gets all nodes currently stored in the graph.
    /// </summary>
    /// <returns>A task that represents the asynchronous operation. The task result contains all nodes.</returns>
    Task<IEnumerable<GraphNode<T>>> GetAllNodesAsync();

    /// <summary>
    /// Asynchronously gets all edges currently stored in the graph.
    /// </summary>
    /// <returns>A task that represents the asynchronous operation. The task result contains all edges.</returns>
    Task<IEnumerable<GraphEdge<T>>> GetAllEdgesAsync();

    /// <summary>
    /// Asynchronously removes all nodes and edges from the graph.
    /// </summary>
    /// <returns>A task representing the asynchronous operation.</returns>
    Task ClearAsync();
}
