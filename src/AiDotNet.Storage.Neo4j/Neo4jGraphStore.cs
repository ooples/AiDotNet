using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Neo4j.Driver;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Neo4j-backed implementation of <see cref="IGraphStore{T}"/> that runs a GraphRAG knowledge graph on a
/// production property-graph database. Nodes are stored as labeled Neo4j nodes and edges as relationships;
/// writes upsert via <c>MERGE</c> and traversal uses Cypher <c>MATCH</c>. Ships in the opt-in
/// <c>AiDotNet.Storage.Neo4j</c> package so the core stays free of the Neo4j.Driver dependency.
/// </summary>
/// <typeparam name="T">The numeric type used for vector (embedding) operations.</typeparam>
/// <remarks>
/// <para><b>Model mapping.</b> Every node carries a fixed base label (<c>Entity</c> by default; different
/// stores can use different base labels for per-namespace partitioning within one database). The node's
/// entity type (<c>GraphNode.Label</c>, e.g. PERSON) is stored as the <c>label</c> property, and its
/// <c>Properties</c> bag is stored losslessly as a JSON string in the <c>props</c> property. The embedding
/// is stored as a native Neo4j list of doubles, and timestamps as epoch-millisecond longs. Edges are stored
/// as relationships of a fixed type (<c>RELATED</c> by default) whose <c>relationType</c> property holds the
/// original <c>GraphEdge.RelationType</c>. This keeps labels/relationship types as fixed, escaped literals
/// (they cannot be Cypher parameters) so every per-request value is fully parameterized and no user data is
/// ever concatenated into a query — see <see cref="Neo4jCypher"/>.</para>
/// <para><b>Threading / async.</b> The async methods are the primary implementation (Neo4j I/O is
/// inherently async); the synchronous <see cref="IGraphStore{T}"/> members delegate to them with a blocking
/// wait, matching the sync-over-async pattern used by the other opt-in storage backends. The owned driver is
/// thread-safe and disposed by <see cref="Dispose"/>.</para>
/// <para><b>For Beginners:</b> This saves your knowledge graph into Neo4j, a real graph database, so it
/// survives restarts, scales past RAM, and can be shared and queried by many processes.</para>
/// </remarks>
[ComponentType(ComponentType.DocumentStore)]
[PipelineStage(PipelineStage.Indexing)]
public sealed class Neo4jGraphStore<T> : IGraphStore<T>, IDisposable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDriver _driver;
    private readonly bool _ownsDriver;
    private readonly string? _database;
    private readonly Neo4jCypher _cypher;

    /// <summary>
    /// Creates a store that connects to Neo4j using a bolt/neo4j URI and credentials. The driver is owned
    /// and disposed by this instance. Connection is lazy, so construction does not require a live server.
    /// </summary>
    /// <param name="uri">The connection URI, e.g. <c>neo4j://localhost:7687</c> or <c>bolt://host:7687</c>.</param>
    /// <param name="username">The user name. Pass null/empty to connect without authentication.</param>
    /// <param name="password">The password.</param>
    /// <param name="database">Optional Neo4j database name (Neo4j 4+). Null uses the server default.</param>
    /// <param name="nodeLabel">Base label applied to all nodes (default <c>Entity</c>); scopes a namespace.</param>
    /// <param name="relationshipType">Relationship type applied to all edges (default <c>RELATED</c>).</param>
    public Neo4jGraphStore(
        string uri,
        string? username,
        string? password,
        string? database = null,
        string nodeLabel = "Entity",
        string relationshipType = "RELATED")
    {
        if (string.IsNullOrWhiteSpace(uri))
        {
            throw new ArgumentException("Connection URI cannot be null, empty, or whitespace.", nameof(uri));
        }

        var authToken = string.IsNullOrEmpty(username)
            ? AuthTokens.None
            : AuthTokens.Basic(username, password);

        _driver = GraphDatabase.Driver(uri, authToken);
        _ownsDriver = true;
        _database = string.IsNullOrWhiteSpace(database) ? null : database;
        _cypher = new Neo4jCypher(nodeLabel, relationshipType);
    }

    /// <summary>
    /// Creates a store over an existing driver (not owned; the caller disposes it). Useful for sharing a
    /// single tuned driver across stores.
    /// </summary>
    /// <param name="driver">A configured Neo4j driver.</param>
    /// <param name="database">Optional Neo4j database name. Null uses the server default.</param>
    /// <param name="nodeLabel">Base label applied to all nodes (default <c>Entity</c>).</param>
    /// <param name="relationshipType">Relationship type applied to all edges (default <c>RELATED</c>).</param>
    public Neo4jGraphStore(
        IDriver driver,
        string? database = null,
        string nodeLabel = "Entity",
        string relationshipType = "RELATED")
    {
        _driver = driver ?? throw new ArgumentNullException(nameof(driver));
        _ownsDriver = false;
        _database = string.IsNullOrWhiteSpace(database) ? null : database;
        _cypher = new Neo4jCypher(nodeLabel, relationshipType);
    }

    /// <inheritdoc/>
    public int NodeCount => CountAsync(_cypher.NodeCount()).GetAwaiter().GetResult();

    /// <inheritdoc/>
    public int EdgeCount => CountAsync(_cypher.EdgeCount()).GetAwaiter().GetResult();

    // ---- Async implementation (primary). ----

    /// <inheritdoc/>
    public async Task AddNodeAsync(GraphNode<T> node)
    {
        if (node == null)
        {
            throw new ArgumentNullException(nameof(node));
        }

        var parameters = new
        {
            id = node.Id,
            label = node.Label,
            props = Neo4jCypher.SerializeProperties(node.Properties),
            embedding = WriteEmbedding(node.Embedding),
            createdAt = Neo4jCypher.ToEpochMillis(node.CreatedAt),
            updatedAt = Neo4jCypher.ToEpochMillis(node.UpdatedAt),
        };

        await RunWriteAsync(_cypher.UpsertNode(), parameters).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async Task AddEdgeAsync(GraphEdge<T> edge)
    {
        if (edge == null)
        {
            throw new ArgumentNullException(nameof(edge));
        }

        var parameters = new
        {
            id = edge.Id,
            sourceId = edge.SourceId,
            targetId = edge.TargetId,
            relationType = edge.RelationType,
            weight = edge.Weight,
            props = Neo4jCypher.SerializeProperties(edge.Properties),
            createdAt = Neo4jCypher.ToEpochMillis(edge.CreatedAt),
            validFrom = Neo4jCypher.ToEpochMillis(edge.ValidFrom),
            validUntil = Neo4jCypher.ToEpochMillis(edge.ValidUntil),
        };

        var records = await RunWriteAsync(_cypher.UpsertEdge(), parameters).ConfigureAwait(false);
        if (records.Count == 0)
        {
            // A MATCH for one of the endpoints produced no row, so MERGE never ran: the interface
            // contract (see MemoryGraphStore) requires both endpoints to already exist.
            throw new InvalidOperationException(
                $"Cannot add edge '{edge.Id}': source node '{edge.SourceId}' and/or target node '{edge.TargetId}' does not exist.");
        }
    }

    /// <inheritdoc/>
    public async Task<GraphNode<T>?> GetNodeAsync(string nodeId)
    {
        if (string.IsNullOrWhiteSpace(nodeId))
        {
            return null;
        }

        var records = await RunReadAsync(_cypher.GetNode(), new { id = nodeId }).ConfigureAwait(false);
        if (records.Count == 0)
        {
            return null;
        }

        return MapNode(records[0]["n"].As<INode>());
    }

    /// <inheritdoc/>
    public async Task<GraphEdge<T>?> GetEdgeAsync(string edgeId)
    {
        if (string.IsNullOrWhiteSpace(edgeId))
        {
            return null;
        }

        var records = await RunReadAsync(_cypher.GetEdge(), new { id = edgeId }).ConfigureAwait(false);
        return records.Count == 0 ? null : MapEdge(records[0]);
    }

    /// <inheritdoc/>
    public async Task<bool> RemoveNodeAsync(string nodeId)
    {
        if (string.IsNullOrWhiteSpace(nodeId))
        {
            return false;
        }

        var records = await RunWriteAsync(_cypher.RemoveNode(), new { id = nodeId }).ConfigureAwait(false);
        return records.Count > 0 && records[0]["deleted"].As<long>() > 0;
    }

    /// <inheritdoc/>
    public async Task<bool> RemoveEdgeAsync(string edgeId)
    {
        if (string.IsNullOrWhiteSpace(edgeId))
        {
            return false;
        }

        var records = await RunWriteAsync(_cypher.RemoveEdge(), new { id = edgeId }).ConfigureAwait(false);
        return records.Count > 0 && records[0]["deleted"].As<long>() > 0;
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<GraphEdge<T>>> GetOutgoingEdgesAsync(string nodeId)
    {
        var records = await RunReadAsync(_cypher.OutgoingEdges(), new { id = nodeId }).ConfigureAwait(false);
        return records.Select(MapEdge).ToList();
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<GraphEdge<T>>> GetIncomingEdgesAsync(string nodeId)
    {
        var records = await RunReadAsync(_cypher.IncomingEdges(), new { id = nodeId }).ConfigureAwait(false);
        return records.Select(MapEdge).ToList();
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<GraphNode<T>>> GetNodesByLabelAsync(string label)
    {
        var records = await RunReadAsync(_cypher.NodesByLabel(), new { label }).ConfigureAwait(false);
        return records.Select(r => MapNode(r["n"].As<INode>())).ToList();
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<GraphNode<T>>> GetAllNodesAsync()
    {
        var records = await RunReadAsync(_cypher.AllNodes(), new { }).ConfigureAwait(false);
        return records.Select(r => MapNode(r["n"].As<INode>())).ToList();
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<GraphEdge<T>>> GetAllEdgesAsync()
    {
        var records = await RunReadAsync(_cypher.AllEdges(), new { }).ConfigureAwait(false);
        return records.Select(MapEdge).ToList();
    }

    /// <inheritdoc/>
    public async Task ClearAsync()
    {
        await RunWriteAsync(_cypher.Clear(), new { }).ConfigureAwait(false);
    }

    // ---- Synchronous members: sync-over-async, matching the other opt-in storage backends. ----

    /// <inheritdoc/>
    public void AddNode(GraphNode<T> node) => AddNodeAsync(node).GetAwaiter().GetResult();

    /// <inheritdoc/>
    public void AddEdge(GraphEdge<T> edge) => AddEdgeAsync(edge).GetAwaiter().GetResult();

    /// <inheritdoc/>
    public GraphNode<T>? GetNode(string nodeId) => GetNodeAsync(nodeId).GetAwaiter().GetResult();

    /// <inheritdoc/>
    public GraphEdge<T>? GetEdge(string edgeId) => GetEdgeAsync(edgeId).GetAwaiter().GetResult();

    /// <inheritdoc/>
    public bool RemoveNode(string nodeId) => RemoveNodeAsync(nodeId).GetAwaiter().GetResult();

    /// <inheritdoc/>
    public bool RemoveEdge(string edgeId) => RemoveEdgeAsync(edgeId).GetAwaiter().GetResult();

    /// <inheritdoc/>
    public IEnumerable<GraphEdge<T>> GetOutgoingEdges(string nodeId) => GetOutgoingEdgesAsync(nodeId).GetAwaiter().GetResult();

    /// <inheritdoc/>
    public IEnumerable<GraphEdge<T>> GetIncomingEdges(string nodeId) => GetIncomingEdgesAsync(nodeId).GetAwaiter().GetResult();

    /// <inheritdoc/>
    public IEnumerable<GraphNode<T>> GetNodesByLabel(string label) => GetNodesByLabelAsync(label).GetAwaiter().GetResult();

    /// <inheritdoc/>
    public IEnumerable<GraphNode<T>> GetAllNodes() => GetAllNodesAsync().GetAwaiter().GetResult();

    /// <inheritdoc/>
    public IEnumerable<GraphEdge<T>> GetAllEdges() => GetAllEdgesAsync().GetAwaiter().GetResult();

    /// <inheritdoc/>
    public void Clear() => ClearAsync().GetAwaiter().GetResult();

    // ---- Session helpers. ----

    private void ConfigureSession(SessionConfigBuilder builder)
    {
        if (_database != null)
        {
            builder.WithDatabase(_database);
        }
    }

    private async Task<List<IRecord>> RunWriteAsync(string cypher, object parameters)
    {
        await using var session = _driver.AsyncSession(ConfigureSession);
        return await session.ExecuteWriteAsync(async tx =>
        {
            var cursor = await tx.RunAsync(cypher, parameters).ConfigureAwait(false);
            return await cursor.ToListAsync().ConfigureAwait(false);
        }).ConfigureAwait(false);
    }

    private async Task<List<IRecord>> RunReadAsync(string cypher, object parameters)
    {
        await using var session = _driver.AsyncSession(ConfigureSession);
        return await session.ExecuteReadAsync(async tx =>
        {
            var cursor = await tx.RunAsync(cypher, parameters).ConfigureAwait(false);
            return await cursor.ToListAsync().ConfigureAwait(false);
        }).ConfigureAwait(false);
    }

    private async Task<int> CountAsync(string cypher)
    {
        var records = await RunReadAsync(cypher, new { }).ConfigureAwait(false);
        return records.Count == 0 ? 0 : (int)records[0]["c"].As<long>();
    }

    // ---- Record <-> model mapping. ----

    private static GraphNode<T> MapNode(INode node)
    {
        var props = node.Properties;
        var id = ReadString(props, Neo4jCypher.PropId) ?? throw new InvalidOperationException("Neo4j node is missing its 'id' property.");
        var label = ReadString(props, Neo4jCypher.PropLabel);
        if (string.IsNullOrWhiteSpace(label))
        {
            // Labels stored by this backend are never empty; fall back defensively so a malformed row
            // cannot trip the GraphNode constructor's non-empty-label guard.
            label = "UNKNOWN";
        }

        var graphNode = new GraphNode<T>(id, label!)
        {
            Properties = Neo4jCypher.DeserializeProperties(ReadString(props, Neo4jCypher.PropProps)),
            Embedding = ReadEmbedding(props),
        };

        var createdAt = ReadNullableEpoch(props, Neo4jCypher.PropCreatedAt);
        if (createdAt.HasValue)
        {
            graphNode.CreatedAt = createdAt.Value;
        }

        var updatedAt = ReadNullableEpoch(props, Neo4jCypher.PropUpdatedAt);
        if (updatedAt.HasValue)
        {
            graphNode.UpdatedAt = updatedAt.Value;
        }

        return graphNode;
    }

    private static GraphEdge<T> MapEdge(IRecord record)
    {
        var sourceId = record["sourceId"].As<string>();
        var targetId = record["targetId"].As<string>();
        var rel = record["r"].As<IRelationship>();
        var props = rel.Properties;

        var relationType = ReadString(props, Neo4jCypher.PropRelationType);
        if (string.IsNullOrWhiteSpace(relationType))
        {
            relationType = rel.Type;
        }

        // Construct with the default (valid) weight, then assign the stored weight directly to stay
        // lossless without tripping the GraphEdge constructor's [0,1] weight guard.
        var edge = new GraphEdge<T>(sourceId, targetId, relationType!);

        var idValue = ReadString(props, Neo4jCypher.PropId);
        if (!string.IsNullOrEmpty(idValue))
        {
            edge.Id = idValue!;
        }

        if (props.TryGetValue(Neo4jCypher.PropWeight, out var weightRaw) && weightRaw != null)
        {
            edge.Weight = Convert.ToDouble(weightRaw, CultureInfo.InvariantCulture);
        }

        edge.Properties = Neo4jCypher.DeserializeProperties(ReadString(props, Neo4jCypher.PropProps));

        var createdAt = ReadNullableEpoch(props, Neo4jCypher.PropCreatedAt);
        if (createdAt.HasValue)
        {
            edge.CreatedAt = createdAt.Value;
        }

        edge.ValidFrom = ReadNullableEpoch(props, Neo4jCypher.PropValidFrom);
        edge.ValidUntil = ReadNullableEpoch(props, Neo4jCypher.PropValidUntil);
        return edge;
    }

    private static double[]? WriteEmbedding(Vector<T>? embedding)
    {
        if (embedding == null)
        {
            return null;
        }

        var values = new double[embedding.Length];
        for (int i = 0; i < embedding.Length; i++)
        {
            values[i] = NumOps.ToDouble(embedding[i]);
        }

        return values;
    }

    private static Vector<T>? ReadEmbedding(IReadOnlyDictionary<string, object> props)
    {
        if (!props.TryGetValue(Neo4jCypher.PropEmbedding, out var raw) || raw is null)
        {
            return null;
        }

        if (raw is System.Collections.IEnumerable list && raw is not string)
        {
            var values = new List<T>();
            foreach (var item in list)
            {
                values.Add(NumOps.FromDouble(Convert.ToDouble(item, CultureInfo.InvariantCulture)));
            }

            return new Vector<T>(values.ToArray());
        }

        return null;
    }

    private static string? ReadString(IReadOnlyDictionary<string, object> props, string key)
    {
        if (props.TryGetValue(key, out var value) && value != null)
        {
            return Convert.ToString(value, CultureInfo.InvariantCulture);
        }

        return null;
    }

    private static DateTime? ReadNullableEpoch(IReadOnlyDictionary<string, object> props, string key)
    {
        if (props.TryGetValue(key, out var value) && value != null)
        {
            return Neo4jCypher.FromEpochMillis(Convert.ToInt64(value, CultureInfo.InvariantCulture));
        }

        return null;
    }

    /// <summary>Disposes the owned Neo4j driver. No-op when constructed over a shared driver.</summary>
    public void Dispose()
    {
        if (_ownsDriver)
        {
            _driver.Dispose();
        }
    }
}
