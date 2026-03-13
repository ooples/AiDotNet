using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// File-based implementation of <see cref="IGraphStore{T}"/> with persistent storage on disk.
/// </summary>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// This implementation provides persistent graph storage using files:
/// - nodes.dat: Binary file containing serialized nodes
/// - edges.dat: Binary file containing serialized edges
/// - node_index.db: B-Tree index mapping node IDs to file offsets
/// - edge_index.db: B-Tree index mapping edge IDs to file offsets
/// </para>
/// <para><b>For Beginners:</b> This stores your graph on disk so it survives restarts.
///
/// How it works:
/// 1. When you add a node, it's written to nodes.dat
/// 2. The position (offset) is recorded in node_index.db
/// 3. To retrieve a node, we look up its offset and read from that position
/// 4. Everything is saved to disk automatically
///
/// Pros:
/// - 💾 Data persists across restarts
/// - 🔄 Can handle graphs larger than RAM
/// - 📁 Simple file-based storage (no database required)
///
/// Cons:
/// - 🐌 Slower than in-memory (disk I/O overhead)
/// - 🔒 Not suitable for concurrent access from multiple processes
/// - 📦 No compression (files can be large)
///
/// Good for:
/// - Applications that need to save graph state
/// - Graphs up to a few million nodes
/// - Single-process applications
///
/// Not good for:
/// - Real-time systems requiring sub-millisecond latency
/// - Multi-process concurrent access
/// - Distributed systems (use Neo4j or similar instead)
/// </para>
/// </remarks>
public class FileGraphStore<T> : IGraphStore<T>, IDisposable
{
    private readonly string _storageDirectory;
    private readonly string _nodesFilePath;
    private readonly string _edgesFilePath;
    private readonly BTreeIndex _nodeIndex;
    private readonly BTreeIndex _edgeIndex;
    private readonly WriteAheadLog? _wal;
    private readonly SemaphoreSlim _nodesWriteLock = new SemaphoreSlim(1, 1);
    private readonly SemaphoreSlim _edgesWriteLock = new SemaphoreSlim(1, 1);

    // In-memory caches for indices and metadata (thread-safe for concurrent async access)
    private readonly ConcurrentDictionary<string, HashSet<string>> _outgoingEdges; // nodeId -> edge IDs
    private readonly ConcurrentDictionary<string, HashSet<string>> _incomingEdges; // nodeId -> edge IDs
    private readonly ConcurrentDictionary<string, HashSet<string>> _nodesByLabel; // label -> node IDs
    private readonly object _cacheLock = new object(); // Lock for modifying HashSet contents

    private readonly JsonSerializerSettings _jsonSettings;
    private bool _disposed;

    /// <inheritdoc/>
    public int NodeCount => _nodeIndex.Count;

    /// <inheritdoc/>
    public int EdgeCount => _edgeIndex.Count;

    /// <summary>
    /// Initializes a new instance of the <see cref="FileGraphStore{T}"/> class.
    /// </summary>
    /// <param name="storageDirectory">The directory where graph data files will be stored.</param>
    /// <param name="wal">Optional Write-Ahead Log for ACID transactions and crash recovery.</param>
    public FileGraphStore(string storageDirectory, WriteAheadLog? wal = null)
    {
        if (string.IsNullOrWhiteSpace(storageDirectory))
            throw new ArgumentException("Storage directory cannot be null or whitespace", nameof(storageDirectory));

        _storageDirectory = storageDirectory;
        _nodesFilePath = Path.Combine(storageDirectory, "nodes.dat");
        _edgesFilePath = Path.Combine(storageDirectory, "edges.dat");
        _wal = wal;

        // Create directory if it doesn't exist
        if (!Directory.Exists(storageDirectory))
            Directory.CreateDirectory(storageDirectory);

        // Initialize indices
        _nodeIndex = new BTreeIndex(Path.Combine(storageDirectory, "node_index.db"));
        _edgeIndex = new BTreeIndex(Path.Combine(storageDirectory, "edge_index.db"));

        // Initialize in-memory structures (thread-safe)
        _outgoingEdges = new ConcurrentDictionary<string, HashSet<string>>();
        _incomingEdges = new ConcurrentDictionary<string, HashSet<string>>();
        _nodesByLabel = new ConcurrentDictionary<string, HashSet<string>>();

        _jsonSettings = new JsonSerializerSettings
        {
            Formatting = Formatting.None
        };

        // Rebuild in-memory indices from persisted data
        RebuildInMemoryIndices();
    }

    /// <inheritdoc/>
    public void AddNode(GraphNode<T> node)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        try
        {
            // Log to WAL first (durability)
            _wal?.LogAddNode(node);

            // Serialize node to JSON
            var json = JsonConvert.SerializeObject(node, _jsonSettings);
            var bytes = Encoding.UTF8.GetBytes(json);

            _nodesWriteLock.Wait();
            long offset;
            try
            {
                // Get current file position (or reuse existing offset if updating)
                // For updates, we append to the end (old data becomes garbage)
                // In production, you'd implement compaction to reclaim space
                offset = new FileInfo(_nodesFilePath).Exists ? new FileInfo(_nodesFilePath).Length : 0;

                // Write node data to file
                using (var stream = new FileStream(_nodesFilePath, FileMode.Append, FileAccess.Write, FileShare.Read))
                {
                    // Write length prefix (4 bytes)
                    var lengthBytes = BitConverter.GetBytes(bytes.Length);
                    stream.Write(lengthBytes, 0, 4);

                    // Write JSON data
                    stream.Write(bytes, 0, bytes.Length);
                }

                // Update index
                _nodeIndex.Add(node.Id, offset);

                // Flush indices periodically (every 100 operations for performance)
                if (_nodeIndex.Count % 100 == 0)
                    _nodeIndex.Flush();
            }
            finally
            {
                _nodesWriteLock.Release();
            }

            // Update in-memory indices (thread-safe)
            lock (_cacheLock)
            {
                var labelSet = _nodesByLabel.GetOrAdd(node.Label, _ => new HashSet<string>());
                labelSet.Add(node.Id);

                _outgoingEdges.GetOrAdd(node.Id, _ => new HashSet<string>());
                _incomingEdges.GetOrAdd(node.Id, _ => new HashSet<string>());
            }

        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to add node '{node.Id}' to file store", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException($"Failed to add node '{node.Id}' to file store due to unauthorized access", ex);
        }
        catch (JsonSerializationException ex)
        {
            throw new IOException($"Failed to serialize node '{node.Id}' to JSON", ex);
        }
    }

    /// <inheritdoc/>
    public void AddEdge(GraphEdge<T> edge)
    {
        if (edge == null)
            throw new ArgumentNullException(nameof(edge));
        if (!_nodeIndex.Contains(edge.SourceId))
            throw new InvalidOperationException($"Source node '{edge.SourceId}' does not exist");
        if (!_nodeIndex.Contains(edge.TargetId))
            throw new InvalidOperationException($"Target node '{edge.TargetId}' does not exist");

        try
        {
            // Log to WAL first (durability)
            _wal?.LogAddEdge(edge);

            // Serialize edge to JSON
            var json = JsonConvert.SerializeObject(edge, _jsonSettings);
            var bytes = Encoding.UTF8.GetBytes(json);

            _edgesWriteLock.Wait();
            long offset;
            try
            {
                // Get current file position
                offset = new FileInfo(_edgesFilePath).Exists ? new FileInfo(_edgesFilePath).Length : 0;

                // Write edge data to file
                using (var stream = new FileStream(_edgesFilePath, FileMode.Append, FileAccess.Write, FileShare.Read))
                {
                    // Write length prefix (4 bytes)
                    var lengthBytes = BitConverter.GetBytes(bytes.Length);
                    stream.Write(lengthBytes, 0, 4);

                    // Write JSON data
                    stream.Write(bytes, 0, bytes.Length);
                }

                // Update index
                _edgeIndex.Add(edge.Id, offset);

                // Flush indices periodically
                if (_edgeIndex.Count % 100 == 0)
                    _edgeIndex.Flush();
            }
            finally
            {
                _edgesWriteLock.Release();
            }

            // Update in-memory edge indices (thread-safe)
            lock (_cacheLock)
            {
                if (_outgoingEdges.TryGetValue(edge.SourceId, out var outgoingSet))
                    outgoingSet.Add(edge.Id);
                if (_incomingEdges.TryGetValue(edge.TargetId, out var incomingSet))
                    incomingSet.Add(edge.Id);
            }

        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to add edge '{edge.Id}' to file store", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException($"Failed to add edge '{edge.Id}' to file store due to access permissions", ex);
        }
        catch (JsonSerializationException ex)
        {
            throw new IOException($"Failed to serialize edge '{edge.Id}' to JSON", ex);
        }
    }

    /// <inheritdoc/>
    public GraphNode<T>? GetNode(string nodeId)
    {
        if (string.IsNullOrWhiteSpace(nodeId))
            return null;

        var offset = _nodeIndex.Get(nodeId);
        if (offset < 0)
            return null;

        try
        {
            using var stream = new FileStream(_nodesFilePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            stream.Seek(offset, SeekOrigin.Begin);

            // Read length prefix - ensure all 4 bytes are read
            var lengthBytes = new byte[4];
            ReadExactly(stream, lengthBytes, 0, 4);
            var length = BitConverter.ToInt32(lengthBytes, 0);

            // Read JSON data - ensure all bytes are read
            var jsonBytes = new byte[length];
            ReadExactly(stream, jsonBytes, 0, length);
            var json = Encoding.UTF8.GetString(jsonBytes);

            // Deserialize
            return JsonConvert.DeserializeObject<GraphNode<T>>(json, _jsonSettings);
        }
        catch (EndOfStreamException ex)
        {
            throw new IOException($"Failed to read node '{nodeId}' from file store - data may be corrupted", ex);
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to read node '{nodeId}' from file store", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException($"Failed to read node '{nodeId}' from file store", ex);
        }
        catch (JsonSerializationException ex)
        {
            throw new IOException($"Failed to deserialize node '{nodeId}' from JSON", ex);
        }
    }

    /// <inheritdoc/>
    public GraphEdge<T>? GetEdge(string edgeId)
    {
        if (string.IsNullOrWhiteSpace(edgeId))
            return null;

        var offset = _edgeIndex.Get(edgeId);
        if (offset < 0)
            return null;

        try
        {
            using var stream = new FileStream(_edgesFilePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            stream.Seek(offset, SeekOrigin.Begin);

            // Read length prefix - ensure all 4 bytes are read
            var lengthBytes = new byte[4];
            ReadExactly(stream, lengthBytes, 0, 4);
            var length = BitConverter.ToInt32(lengthBytes, 0);

            // Read JSON data - ensure all bytes are read
            var jsonBytes = new byte[length];
            ReadExactly(stream, jsonBytes, 0, length);
            var json = Encoding.UTF8.GetString(jsonBytes);

            // Deserialize
            return JsonConvert.DeserializeObject<GraphEdge<T>>(json, _jsonSettings);
        }
        catch (EndOfStreamException ex)
        {
            throw new IOException($"Failed to read edge '{edgeId}' from file store - data may be corrupted", ex);
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to read edge '{edgeId}' from file store", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException($"Failed to read edge '{edgeId}' from file store", ex);
        }
        catch (JsonSerializationException ex)
        {
            throw new IOException($"Failed to deserialize edge '{edgeId}' from JSON", ex);
        }
    }

    /// <inheritdoc/>
    public bool RemoveNode(string nodeId)
    {
        if (string.IsNullOrWhiteSpace(nodeId) || !_nodeIndex.Contains(nodeId))
            return false;

        try
        {
            var node = GetNode(nodeId);
            if (node == null)
                return false;

            // Log to WAL first (durability)
            _wal?.LogRemoveNode(nodeId);

            // Remove all outgoing edges (thread-safe)
            if (_outgoingEdges.TryGetValue(nodeId, out var outgoing))
            {
                List<string> edgesToRemove;
                lock (_cacheLock)
                {
                    edgesToRemove = outgoing.ToList();
                }
                foreach (var edgeId in edgesToRemove)
                {
                    RemoveEdge(edgeId);
                }
                _outgoingEdges.TryRemove(nodeId, out _);
            }

            // Remove all incoming edges (thread-safe)
            if (_incomingEdges.TryGetValue(nodeId, out var incoming))
            {
                List<string> edgesToRemove;
                lock (_cacheLock)
                {
                    edgesToRemove = incoming.ToList();
                }
                foreach (var edgeId in edgesToRemove)
                {
                    RemoveEdge(edgeId);
                }
                _incomingEdges.TryRemove(nodeId, out _);
            }

            // Remove from label index (thread-safe)
            lock (_cacheLock)
            {
                if (_nodesByLabel.TryGetValue(node.Label, out var nodeIds))
                {
                    nodeIds.Remove(nodeId);
                    if (nodeIds.Count == 0)
                        _nodesByLabel.TryRemove(node.Label, out _);
                }
            }

            // Remove from node index (marks as deleted, actual data remains)
            _nodeIndex.Remove(nodeId);
            _nodeIndex.Flush();

            return true;
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to remove node '{nodeId}' from file store", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException($"Failed to remove node '{nodeId}' from file store", ex);
        }
    }

    /// <inheritdoc/>
    public bool RemoveEdge(string edgeId)
    {
        if (string.IsNullOrWhiteSpace(edgeId) || !_edgeIndex.Contains(edgeId))
            return false;

        try
        {
            var edge = GetEdge(edgeId);
            if (edge == null)
                return false;

            // Log to WAL first (durability)
            _wal?.LogRemoveEdge(edgeId);

            // Remove from in-memory indices (thread-safe)
            lock (_cacheLock)
            {
                if (_outgoingEdges.TryGetValue(edge.SourceId, out var outgoing))
                    outgoing.Remove(edgeId);
                if (_incomingEdges.TryGetValue(edge.TargetId, out var incoming))
                    incoming.Remove(edgeId);
            }

            // Remove from edge index
            _edgeIndex.Remove(edgeId);
            _edgeIndex.Flush();

            return true;
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to remove edge '{edgeId}' from file store", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException($"Failed to remove edge '{edgeId}' from file store", ex);
        }
    }

    /// <inheritdoc/>
    public IEnumerable<GraphEdge<T>> GetOutgoingEdges(string nodeId)
    {
        if (!_outgoingEdges.TryGetValue(nodeId, out var edgeIds))
            return Enumerable.Empty<GraphEdge<T>>();

        // Take snapshot of edge IDs under lock to avoid concurrent modification
        List<string> snapshot;
        lock (_cacheLock)
        {
            snapshot = edgeIds.ToList();
        }
        return snapshot.Select(id => GetEdge(id)).Where(e => e != null).Cast<GraphEdge<T>>();
    }

    /// <inheritdoc/>
    public IEnumerable<GraphEdge<T>> GetIncomingEdges(string nodeId)
    {
        if (!_incomingEdges.TryGetValue(nodeId, out var edgeIds))
            return Enumerable.Empty<GraphEdge<T>>();

        // Take snapshot of edge IDs under lock to avoid concurrent modification
        List<string> snapshot;
        lock (_cacheLock)
        {
            snapshot = edgeIds.ToList();
        }
        return snapshot.Select(id => GetEdge(id)).Where(e => e != null).Cast<GraphEdge<T>>();
    }

    /// <inheritdoc/>
    public IEnumerable<GraphNode<T>> GetNodesByLabel(string label)
    {
        if (!_nodesByLabel.TryGetValue(label, out var nodeIds))
            return Enumerable.Empty<GraphNode<T>>();

        // Take snapshot of node IDs under lock to avoid concurrent modification
        List<string> snapshot;
        lock (_cacheLock)
        {
            snapshot = nodeIds.ToList();
        }
        return snapshot.Select(id => GetNode(id)).Where(n => n != null).Cast<GraphNode<T>>();
    }

    /// <inheritdoc/>
    public IEnumerable<GraphNode<T>> GetAllNodes()
    {
        return _nodeIndex.GetAllKeys().Select(id => GetNode(id)).Where(n => n != null).Cast<GraphNode<T>>();
    }

    /// <inheritdoc/>
    public IEnumerable<GraphEdge<T>> GetAllEdges()
    {
        return _edgeIndex.GetAllKeys().Select(id => GetEdge(id)).Where(e => e != null).Cast<GraphEdge<T>>();
    }

    /// <inheritdoc/>
    public void Clear()
    {
        try
        {
            // Clear in-memory structures
            _outgoingEdges.Clear();
            _incomingEdges.Clear();
            _nodesByLabel.Clear();

            // Clear indices
            _nodeIndex.Clear();
            _edgeIndex.Clear();
            _nodeIndex.Flush();
            _edgeIndex.Flush();

            // Delete data files
            if (File.Exists(_nodesFilePath))
                File.Delete(_nodesFilePath);
            if (File.Exists(_edgesFilePath))
                File.Delete(_edgesFilePath);
        }
        catch (IOException ex)
        {
            throw new IOException("Failed to clear file store", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException("Failed to clear file store", ex);
        }
    }

    /// <summary>
    /// Rebuilds in-memory indices by scanning all nodes and edges.
    /// </summary>
    /// <remarks>
    /// This is called during initialization to restore the in-memory indices
    /// from persisted data. It scans all nodes to rebuild label indices and
    /// all edges to rebuild outgoing/incoming edge indices.
    /// </remarks>
    private void RebuildInMemoryIndices()
    {
        try
        {
            // Rebuild node-related indices (thread-safe using GetOrAdd)
            foreach (var node in _nodeIndex.GetAllKeys().Select(GetNode).OfType<GraphNode<T>>())
            {
                // Rebuild label index
                var labelSet = _nodesByLabel.GetOrAdd(node.Label, _ => new HashSet<string>());
                lock (_cacheLock)
                {
                    labelSet.Add(node.Id);
                }

                // Initialize edge indices
                _outgoingEdges.GetOrAdd(node.Id, _ => new HashSet<string>());
                _incomingEdges.GetOrAdd(node.Id, _ => new HashSet<string>());
            }

            // Rebuild edge indices (thread-safe)
            foreach (var edge in _edgeIndex.GetAllKeys().Select(GetEdge).OfType<GraphEdge<T>>())
            {
                lock (_cacheLock)
                {
                    if (_outgoingEdges.TryGetValue(edge.SourceId, out var outgoingSet))
                        outgoingSet.Add(edge.Id);
                    if (_incomingEdges.TryGetValue(edge.TargetId, out var incomingSet))
                        incomingSet.Add(edge.Id);
                }
            }
        }
        catch (IOException ex)
        {
            throw new IOException("Failed to rebuild in-memory indices", ex);
        }
        catch (InvalidDataException ex)
        {
            throw new IOException("Failed to rebuild in-memory indices", ex);
        }
    }

    // Async methods for non-blocking I/O operations

    /// <inheritdoc/>
    public async Task AddNodeAsync(GraphNode<T> node)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        try
        {
            // Log to WAL first (durability)
            _wal?.LogAddNode(node);

            // Serialize node to JSON
            var json = JsonConvert.SerializeObject(node, _jsonSettings);
            var bytes = Encoding.UTF8.GetBytes(json);

            await _nodesWriteLock.WaitAsync().ConfigureAwait(false);
            long offset;
            try
            {
                // Get current file position
                offset = new FileInfo(_nodesFilePath).Exists ? new FileInfo(_nodesFilePath).Length : 0;

                // Write node data to file asynchronously
                using (var stream = new FileStream(_nodesFilePath, FileMode.Append, FileAccess.Write, FileShare.Read, 4096, useAsync: true))
                {
                    // Write length prefix (4 bytes)
                    var lengthBytes = BitConverter.GetBytes(bytes.Length);
                    await stream.WriteAsync(lengthBytes, 0, 4).ConfigureAwait(false);

                    // Write JSON data
                    await stream.WriteAsync(bytes, 0, bytes.Length).ConfigureAwait(false);
                }

                // Update index
                _nodeIndex.Add(node.Id, offset);

                // Flush indices periodically
                if (_nodeIndex.Count % 100 == 0)
                    _nodeIndex.Flush();
            }
            finally
            {
                _nodesWriteLock.Release();
            }

            // Update in-memory indices (thread-safe)
            lock (_cacheLock)
            {
                var labelSet = _nodesByLabel.GetOrAdd(node.Label, _ => new HashSet<string>());
                labelSet.Add(node.Id);

                _outgoingEdges.GetOrAdd(node.Id, _ => new HashSet<string>());
                _incomingEdges.GetOrAdd(node.Id, _ => new HashSet<string>());
            }

        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to add node '{node.Id}' to file store", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException($"Failed to add node '{node.Id}' to file store due to access permissions", ex);
        }
        catch (JsonSerializationException ex)
        {
            throw new IOException($"Failed to serialize node '{node.Id}' to JSON", ex);
        }
    }

    /// <inheritdoc/>
    public async Task AddEdgeAsync(GraphEdge<T> edge)
    {
        if (edge == null)
            throw new ArgumentNullException(nameof(edge));
        if (!_nodeIndex.Contains(edge.SourceId))
            throw new InvalidOperationException($"Source node '{edge.SourceId}' does not exist");
        if (!_nodeIndex.Contains(edge.TargetId))
            throw new InvalidOperationException($"Target node '{edge.TargetId}' does not exist");

        try
        {
            // Log to WAL first (durability)
            _wal?.LogAddEdge(edge);

            // Serialize edge to JSON
            var json = JsonConvert.SerializeObject(edge, _jsonSettings);
            var bytes = Encoding.UTF8.GetBytes(json);

            await _edgesWriteLock.WaitAsync().ConfigureAwait(false);
            long offset;
            try
            {
                // Get current file position
                offset = new FileInfo(_edgesFilePath).Exists ? new FileInfo(_edgesFilePath).Length : 0;

                // Write edge data to file asynchronously
                using (var stream = new FileStream(_edgesFilePath, FileMode.Append, FileAccess.Write, FileShare.Read, 4096, useAsync: true))
                {
                    // Write length prefix (4 bytes)
                    var lengthBytes = BitConverter.GetBytes(bytes.Length);
                    await stream.WriteAsync(lengthBytes, 0, 4).ConfigureAwait(false);

                    // Write JSON data
                    await stream.WriteAsync(bytes, 0, bytes.Length).ConfigureAwait(false);
                }

                // Update index
                _edgeIndex.Add(edge.Id, offset);

                // Flush indices periodically
                if (_edgeIndex.Count % 100 == 0)
                    _edgeIndex.Flush();
            }
            finally
            {
                _edgesWriteLock.Release();
            }

            // Update in-memory edge indices (thread-safe)
            lock (_cacheLock)
            {
                if (_outgoingEdges.TryGetValue(edge.SourceId, out var outgoingSet))
                    outgoingSet.Add(edge.Id);
                if (_incomingEdges.TryGetValue(edge.TargetId, out var incomingSet))
                    incomingSet.Add(edge.Id);
            }

        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to add edge '{edge.Id}' to file store", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException($"Failed to add edge '{edge.Id}' to file store due to access permissions", ex);
        }
    }

    /// <inheritdoc/>
    public async Task<GraphNode<T>?> GetNodeAsync(string nodeId)
    {
        if (string.IsNullOrWhiteSpace(nodeId))
            return null;

        var offset = _nodeIndex.Get(nodeId);
        if (offset < 0)
            return null;

        try
        {
            using var stream = new FileStream(_nodesFilePath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: true);
            stream.Seek(offset, SeekOrigin.Begin);

            // Read length prefix - ensure all 4 bytes are read
            var lengthBytes = new byte[4];
            await ReadExactlyAsync(stream, lengthBytes, 0, 4);
            var length = BitConverter.ToInt32(lengthBytes, 0);

            // Read JSON data - ensure all bytes are read
            var jsonBytes = new byte[length];
            await ReadExactlyAsync(stream, jsonBytes, 0, length);
            var json = Encoding.UTF8.GetString(jsonBytes);

            // Deserialize
            return JsonConvert.DeserializeObject<GraphNode<T>>(json, _jsonSettings);
        }
        catch (EndOfStreamException ex)
        {
            throw new IOException($"Failed to read node '{nodeId}' from file store - data may be corrupted", ex);
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to read node '{nodeId}' from file store", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException($"Failed to read node '{nodeId}' from file store", ex);
        }
        catch (JsonSerializationException ex)
        {
            throw new IOException($"Failed to deserialize node '{nodeId}' from JSON", ex);
        }
    }

    /// <inheritdoc/>
    public async Task<GraphEdge<T>?> GetEdgeAsync(string edgeId)
    {
        if (string.IsNullOrWhiteSpace(edgeId))
            return null;

        var offset = _edgeIndex.Get(edgeId);
        if (offset < 0)
            return null;

        try
        {
            using var stream = new FileStream(_edgesFilePath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: true);
            stream.Seek(offset, SeekOrigin.Begin);

            // Read length prefix - ensure all 4 bytes are read
            var lengthBytes = new byte[4];
            await ReadExactlyAsync(stream, lengthBytes, 0, 4);
            var length = BitConverter.ToInt32(lengthBytes, 0);

            // Read JSON data - ensure all bytes are read
            var jsonBytes = new byte[length];
            await ReadExactlyAsync(stream, jsonBytes, 0, length);
            var json = Encoding.UTF8.GetString(jsonBytes);

            // Deserialize
            return JsonConvert.DeserializeObject<GraphEdge<T>>(json, _jsonSettings);
        }
        catch (EndOfStreamException ex)
        {
            throw new IOException($"Failed to read edge '{edgeId}' from file store - data may be corrupted", ex);
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to read edge '{edgeId}' from file store", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException($"Failed to read edge '{edgeId}' from file store", ex);
        }
        catch (JsonSerializationException ex)
        {
            throw new IOException($"Failed to deserialize edge '{edgeId}' from JSON", ex);
        }
    }

    /// <inheritdoc/>
    public async Task<bool> RemoveNodeAsync(string nodeId)
    {
        if (string.IsNullOrWhiteSpace(nodeId) || !_nodeIndex.Contains(nodeId))
            return false;

        try
        {
            var node = await GetNodeAsync(nodeId);
            if (node == null)
                return false;

            // Log to WAL first (durability)
            _wal?.LogRemoveNode(nodeId);

            // Remove all outgoing edges (thread-safe)
            if (_outgoingEdges.TryGetValue(nodeId, out var outgoing))
            {
                List<string> edgesToRemove;
                lock (_cacheLock)
                {
                    edgesToRemove = outgoing.ToList();
                }
                foreach (var edgeId in edgesToRemove)
                {
                    await RemoveEdgeAsync(edgeId);
                }
                _outgoingEdges.TryRemove(nodeId, out _);
            }

            // Remove all incoming edges (thread-safe)
            if (_incomingEdges.TryGetValue(nodeId, out var incoming))
            {
                List<string> edgesToRemove;
                lock (_cacheLock)
                {
                    edgesToRemove = incoming.ToList();
                }
                foreach (var edgeId in edgesToRemove)
                {
                    await RemoveEdgeAsync(edgeId);
                }
                _incomingEdges.TryRemove(nodeId, out _);
            }

            // Remove from label index (thread-safe)
            lock (_cacheLock)
            {
                if (_nodesByLabel.TryGetValue(node.Label, out var nodeIds))
                {
                    nodeIds.Remove(nodeId);
                    if (nodeIds.Count == 0)
                        _nodesByLabel.TryRemove(node.Label, out _);
                }
            }

            // Remove from node index
            _nodeIndex.Remove(nodeId);
            _nodeIndex.Flush();

            return true;
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to remove node '{nodeId}' from file store", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException($"Failed to remove node '{nodeId}' from file store", ex);
        }
    }

    /// <inheritdoc/>
    public Task<bool> RemoveEdgeAsync(string edgeId)
    {
        return Task.FromResult(RemoveEdge(edgeId));
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<GraphEdge<T>>> GetOutgoingEdgesAsync(string nodeId)
    {
        if (!_outgoingEdges.TryGetValue(nodeId, out var edgeIds))
            return Enumerable.Empty<GraphEdge<T>>();

        // Take snapshot of edgeIds to avoid race condition during enumeration
        List<string> snapshot;
        lock (_cacheLock)
        {
            snapshot = edgeIds.ToList();
        }

        var tasks = snapshot.Select(id => GetEdgeAsync(id));
        var results = await Task.WhenAll(tasks);
        return results.OfType<GraphEdge<T>>().ToList();
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<GraphEdge<T>>> GetIncomingEdgesAsync(string nodeId)
    {
        if (!_incomingEdges.TryGetValue(nodeId, out var edgeIds))
            return Enumerable.Empty<GraphEdge<T>>();

        // Take snapshot of edgeIds to avoid race condition during enumeration
        List<string> snapshot;
        lock (_cacheLock)
        {
            snapshot = edgeIds.ToList();
        }

        var tasks = snapshot.Select(id => GetEdgeAsync(id));
        var results = await Task.WhenAll(tasks);
        return results.OfType<GraphEdge<T>>().ToList();
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<GraphNode<T>>> GetNodesByLabelAsync(string label)
    {
        if (!_nodesByLabel.TryGetValue(label, out var nodeIds))
            return Enumerable.Empty<GraphNode<T>>();

        // Take snapshot of nodeIds to avoid race condition during enumeration
        List<string> snapshot;
        lock (_cacheLock)
        {
            snapshot = nodeIds.ToList();
        }

        var tasks = snapshot.Select(id => GetNodeAsync(id));
        var results = await Task.WhenAll(tasks);
        return results.OfType<GraphNode<T>>().ToList();
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<GraphNode<T>>> GetAllNodesAsync()
    {
        var tasks = _nodeIndex.GetAllKeys().Select(id => GetNodeAsync(id));
        var results = await Task.WhenAll(tasks);
        return results.OfType<GraphNode<T>>().ToList();
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<GraphEdge<T>>> GetAllEdgesAsync()
    {
        var tasks = _edgeIndex.GetAllKeys().Select(id => GetEdgeAsync(id));
        var results = await Task.WhenAll(tasks);
        return results.OfType<GraphEdge<T>>().ToList();
    }

    /// <inheritdoc/>
    public Task ClearAsync()
    {
        Clear();
        return Task.CompletedTask;
    }

    /// <summary>
    /// Disposes the file graph store, ensuring all changes are flushed to disk.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        try
        {
            _nodeIndex.Flush();
            _edgeIndex.Flush();
            _nodeIndex.Dispose();
            _edgeIndex.Dispose();
            _nodesWriteLock.Dispose();
            _edgesWriteLock.Dispose();
        }
        finally
        {
            _disposed = true;
        }
    }

    /// <summary>
    /// Reads exactly the specified number of bytes from the stream.
    /// </summary>
    /// <param name="stream">The stream to read from.</param>
    /// <param name="buffer">The buffer to read into.</param>
    /// <param name="offset">The offset in the buffer to start writing at.</param>
    /// <param name="count">The number of bytes to read.</param>
    /// <exception cref="EndOfStreamException">Thrown if the stream ends before all bytes are read.</exception>
    private static void ReadExactly(Stream stream, byte[] buffer, int offset, int count)
    {
        int totalRead = 0;
        while (totalRead < count)
        {
            int bytesRead = stream.Read(buffer, offset + totalRead, count - totalRead);
            if (bytesRead == 0)
                throw new EndOfStreamException($"Unexpected end of stream. Expected {count} bytes, got {totalRead}.");
            totalRead += bytesRead;
        }
    }

    /// <summary>
    /// Asynchronously reads exactly the specified number of bytes from the stream.
    /// </summary>
    /// <param name="stream">The stream to read from.</param>
    /// <param name="buffer">The buffer to read into.</param>
    /// <param name="offset">The offset in the buffer to start writing at.</param>
    /// <param name="count">The number of bytes to read.</param>
    /// <exception cref="EndOfStreamException">Thrown if the stream ends before all bytes are read.</exception>
    private static async Task ReadExactlyAsync(Stream stream, byte[] buffer, int offset, int count)
    {
        int totalRead = 0;
        while (totalRead < count)
        {
            int bytesRead = await stream.ReadAsync(buffer, offset + totalRead, count - totalRead);
            if (bytesRead == 0)
                throw new EndOfStreamException($"Unexpected end of stream. Expected {count} bytes, got {totalRead}.");
            totalRead += bytesRead;
        }
    }
}
