using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;

using Newtonsoft.Json;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// A durable, file-based vector document store with HNSW indexing and write-ahead logging.
/// </summary>
/// <remarks>
/// <para>
/// FileDocumentStore persists documents to disk while maintaining an in-memory HNSW index
/// for O(log n) approximate nearest neighbor search. It uses a write-ahead log (WAL) for
/// crash recovery and tombstone-based soft deletes with periodic compaction.
/// </para>
/// <para><b>For Beginners:</b> This is like a persistent library that survives application restarts.
///
/// Key features:
/// - Data is saved to disk files so it survives crashes and restarts
/// - Fast similarity search using HNSW graph index (same as the in-memory store)
/// - Write-ahead log ensures no data loss even during crashes
/// - Deleted documents are marked (not immediately removed) and cleaned up periodically
///
/// File layout on disk:
/// <code>
/// your-store-directory/
///   store.meta       - Store header (version, dimensions, count, config)
///   documents.json   - All document content and metadata
///   vectors.bin      - Binary vector data (compact, fast to read)
///   hnsw.bin         - Clean-shutdown marker (HNSW is rebuilt from vectors on load)
///   wal.jsonl        - Write-ahead log (journal of recent operations)
/// </code>
///
/// Best used for:
/// - Applications that need persistent vector search
/// - Medium datasets (up to ~1M documents depending on RAM)
/// - Single-process applications (not for multi-process concurrent access)
/// - Scenarios where you want an embedded vector database without external dependencies
///
/// <b>Thread-safety:</b> While <see cref="ConcurrentDictionary{TKey, TValue}"/> is used
/// internally for document/metadata storage, HNSW index operations (AddVectors, RemoveVectors,
/// Search/Query) are <b>not</b> protected by internal locks. Concurrent mutations from multiple
/// threads are unsupported. Concurrent read-only/search operations may work depending on the
/// HNSW implementation but are not guaranteed. For safe in-process concurrency, serialize writes
/// with an external lock or implement internal synchronization around HNSW accesses.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
public class FileDocumentStore<T> : DocumentStoreBase<T>, IDisposable
{
    // File names within the store directory
    private const string MetaFileName = "store.meta";
    private const string DocumentsFileName = "documents.json";
    private const string VectorsFileName = "vectors.bin";
    private const string HnswFileName = "hnsw.bin";
    private const string WalFileName = "wal.jsonl";

    // Store header version for format compatibility
    private const int FormatVersion = 1;

    private readonly FileDocumentStoreOptions _options;
    private readonly string _directoryPath;
    private readonly ConcurrentDictionary<string, VectorDocument<T>> _store;
    private readonly HNSWIndex<T> _hnswIndex;
    private readonly HashSet<string> _tombstones;
    private readonly object _walLock = new();
    private readonly object _persistLock = new();

    private StreamWriter? _walWriter;
    private long _walSize;
    private int _vectorDimension;
    private bool _disposed;

    /// <inheritdoc/>
    public override int DocumentCount => _store.Count;

    /// <inheritdoc/>
    public override int VectorDimension => _vectorDimension;

    /// <summary>
    /// Gets the number of tombstoned (soft-deleted) documents awaiting compaction.
    /// </summary>
    public int TombstoneCount
    {
        get
        {
            lock (_walLock)
            {
                return _tombstones.Count;
            }
        }
    }

    /// <summary>
    /// Gets the directory path where store files are located.
    /// </summary>
    public string DirectoryPath => _directoryPath;

    /// <summary>
    /// Initializes a new FileDocumentStore, creating or loading from the specified directory.
    /// </summary>
    /// <param name="vectorDimension">The dimension of vector embeddings.</param>
    /// <param name="options">Configuration options for the store.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when vectorDimension is less than or equal to zero.</exception>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="ArgumentException">Thrown when options.DirectoryPath is null or empty.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new file-based document store or loads an existing one.
    ///
    /// Example:
    /// <code>
    /// var options = new FileDocumentStoreOptions
    /// {
    ///     DirectoryPath = "./my-vector-store"
    /// };
    /// var store = new FileDocumentStore&lt;float&gt;(384, options);
    ///
    /// // Add documents - they're automatically saved to disk
    /// store.Add(new VectorDocument&lt;float&gt;(doc, embedding));
    ///
    /// // Search - uses fast HNSW index
    /// var results = store.GetSimilar(queryVector, topK: 5);
    ///
    /// // Data persists across restarts
    /// store.Dispose(); // Flushes all pending data
    ///
    /// // Later, reopen the same store
    /// var store2 = new FileDocumentStore&lt;float&gt;(384, options);
    /// // All documents are still there
    /// </code>
    /// </para>
    /// </remarks>
    public FileDocumentStore(int vectorDimension, FileDocumentStoreOptions options)
    {
        if (vectorDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(vectorDimension), "Vector dimension must be positive");
        if (options == null)
            throw new ArgumentNullException(nameof(options));
        if (string.IsNullOrWhiteSpace(options.DirectoryPath))
            throw new ArgumentException("DirectoryPath must be specified", nameof(options));

        _options = options;
        _vectorDimension = vectorDimension;
        _directoryPath = options.DirectoryPath;
        _store = new ConcurrentDictionary<string, VectorDocument<T>>();
        _tombstones = new HashSet<string>();
        _walSize = 0;

        _hnswIndex = new HNSWIndex<T>(
            new CosineSimilarityMetric<T>(),
            maxConnections: options.HnswMaxConnections,
            efConstruction: options.HnswEfConstruction,
            efSearch: options.HnswEfSearch,
            seed: options.HnswSeed);

        // Ensure directory exists
        Directory.CreateDirectory(_directoryPath);

        // Load existing data if present, then replay WAL
        if (File.Exists(Path.Combine(_directoryPath, MetaFileName)))
        {
            LoadFromDisk();

            if (_vectorDimension != vectorDimension)
            {
                throw new InvalidOperationException(
                    $"Vector dimension mismatch between existing store ({_vectorDimension}) and requested dimension ({vectorDimension}).");
            }
        }

        ReplayWal();
        OpenWalWriter();
    }

    /// <summary>
    /// Core logic for adding a single vector document.
    /// Writes to WAL first for durability, then updates in-memory structures.
    /// </summary>
    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        ThrowIfDisposed();

        // Validate embedding dimension matches the store
        if (vectorDocument.Embedding.Length != _vectorDimension)
        {
            throw new ArgumentException(
                $"Vector dimension mismatch. Expected {_vectorDimension}, got {vectorDocument.Embedding.Length} for document {vectorDocument.Document.Id}",
                nameof(vectorDocument));
        }

        string docId = vectorDocument.Document.Id;

        // Update in-memory structures first so that if they throw,
        // no WAL entry is written and the failed add won't be replayed on reopen.
        lock (_walLock)
        {
            _tombstones.Remove(docId);
        }

        _hnswIndex.Add(docId, vectorDocument.Embedding);
        _store[docId] = vectorDocument;

        // Write to WAL after in-memory success for durability
        WriteWalEntry(new WalEntry
        {
            Operation = WalOperation.Add,
            DocumentId = docId,
            Document = new SerializableDocument
            {
                Id = vectorDocument.Document.Id,
                Content = vectorDocument.Document.Content,
                Metadata = vectorDocument.Document.Metadata
            },
            VectorData = VectorToDoubleArray(vectorDocument.Embedding)
        });

        CheckAutoFlush();
    }

    /// <summary>
    /// Core logic for batch-adding vector documents.
    /// </summary>
    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        ThrowIfDisposed();

        // Validate dimensions
        foreach (var vd in vectorDocuments)
        {
            if (vd.Embedding.Length != _vectorDimension)
            {
                throw new ArgumentException(
                    $"Vector dimension mismatch. Expected {_vectorDimension}, got {vd.Embedding.Length} for document {vd.Document.Id}",
                    nameof(vectorDocuments));
            }
        }

        // Update in-memory structures first so that if they throw,
        // no WAL entries are written and the failed batch won't be replayed on reopen.
        var hnswBatch = new Dictionary<string, Vector<T>>();

        lock (_walLock)
        {
            foreach (var vd in vectorDocuments)
            {
                string docId = vd.Document.Id;
                hnswBatch[docId] = vd.Embedding;
                _tombstones.Remove(docId);
            }
        }

        _hnswIndex.AddBatch(hnswBatch);

        foreach (var vd in vectorDocuments)
        {
            _store[vd.Document.Id] = vd;
        }

        // Write WAL entries after in-memory success for durability
        foreach (var vd in vectorDocuments)
        {
            WriteWalEntry(new WalEntry
            {
                Operation = WalOperation.Add,
                DocumentId = vd.Document.Id,
                Document = new SerializableDocument
                {
                    Id = vd.Document.Id,
                    Content = vd.Document.Content,
                    Metadata = vd.Document.Metadata
                },
                VectorData = VectorToDoubleArray(vd.Embedding)
            });
        }

        CheckAutoFlush();
    }

    /// <summary>
    /// Core logic for similarity search using HNSW index with optional metadata filtering.
    /// </summary>
    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        ThrowIfDisposed();

        int availableCount = Math.Min(_store.Count, _hnswIndex.Count);
        if (availableCount == 0)
            return Enumerable.Empty<Document<T>>();

        bool hasFilters = metadataFilters.Count > 0;
        int fetchCount = hasFilters
            ? Math.Min(topK * 10, availableCount)
            : Math.Min(topK, availableCount);

        var hnswResults = _hnswIndex.Search(queryVector, fetchCount);
        var results = new List<Document<T>>();

        foreach (var (id, score) in hnswResults)
        {
            if (!_store.TryGetValue(id, out var vd))
                continue;

            if (hasFilters && !MatchesFilters(vd.Document, metadataFilters))
                continue;

            var newDocument = new Document<T>(vd.Document.Id, vd.Document.Content, vd.Document.Metadata)
            {
                RelevanceScore = score,
                HasRelevanceScore = true
            };
            results.Add(newDocument);

            if (results.Count >= topK)
                break;
        }

        return results;
    }

    /// <summary>
    /// Core logic for retrieving a document by ID.
    /// </summary>
    protected override Document<T>? GetByIdCore(string documentId)
    {
        ThrowIfDisposed();
        return _store.TryGetValue(documentId, out var vd) ? vd.Document : null;
    }

    /// <summary>
    /// Core logic for removing a document. Uses tombstone-based soft delete.
    /// </summary>
    protected override bool RemoveCore(string documentId)
    {
        ThrowIfDisposed();

        // Atomically attempt to remove the document from the in-memory store
        if (!_store.TryRemove(documentId, out _))
            return false;

        // Update HNSW index and tombstones
        _hnswIndex.Remove(documentId);

        lock (_walLock)
        {
            _tombstones.Add(documentId);
        }

        // Write to WAL after in-memory success for durability
        WriteWalEntry(new WalEntry
        {
            Operation = WalOperation.Remove,
            DocumentId = documentId
        });

        CheckAutoFlush();
        return true;
    }

    /// <summary>
    /// Core logic for retrieving all documents.
    /// </summary>
    protected override IEnumerable<Document<T>> GetAllCore()
    {
        ThrowIfDisposed();
        return _store.Values.Select(vd => vd.Document).ToList();
    }

    /// <summary>
    /// Removes all documents from the store and deletes all files.
    /// </summary>
    public override void Clear()
    {
        ThrowIfDisposed();
        _store.Clear();
        _hnswIndex.Clear();

        lock (_walLock)
        {
            _tombstones.Clear();
        }

        // Close WAL writer
        CloseWalWriter();

        // Delete all files
        DeleteFileIfExists(Path.Combine(_directoryPath, MetaFileName));
        DeleteFileIfExists(Path.Combine(_directoryPath, DocumentsFileName));
        DeleteFileIfExists(Path.Combine(_directoryPath, VectorsFileName));
        DeleteFileIfExists(Path.Combine(_directoryPath, HnswFileName));
        DeleteFileIfExists(Path.Combine(_directoryPath, WalFileName));

        // Reopen WAL
        OpenWalWriter();
    }

    /// <summary>
    /// Flushes all in-memory data to disk, creating a consistent snapshot.
    /// This writes documents, vectors, HNSW graph, and metadata, then clears the WAL.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This saves everything to disk right now.
    /// Normally the WAL handles durability, but Flush creates a clean checkpoint
    /// that's faster to load on next startup.</para>
    /// </remarks>
    public void Flush()
    {
        ThrowIfDisposed();
        lock (_persistLock)
        {
            lock (_walLock)
            {
                PersistToDisk();
                ClearWal();
                _tombstones.Clear();
            }
        }
    }

    /// <summary>
    /// Performs compaction: rebuilds store files removing tombstoned entries and reclaiming space.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Over time, deleted documents leave behind "tombstones"
    /// (markers saying "this was deleted"). Compaction cleans these up and makes the store
    /// files smaller. This happens automatically when the tombstone ratio exceeds the threshold,
    /// but you can trigger it manually.</para>
    /// </remarks>
    public void Compact()
    {
        ThrowIfDisposed();
        lock (_persistLock)
        {
            lock (_walLock)
            {
                _tombstones.Clear();
                PersistToDisk();
                ClearWal();
            }
        }
    }

    /// <summary>
    /// Releases resources used by the store. Flushes pending data to disk.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources used by the store.
    /// </summary>
    /// <param name="disposing">True if called from Dispose(); false if from finalizer.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (disposing)
        {
            // Flush all data to disk before closing
            try
            {
                lock (_persistLock)
                {
                    lock (_walLock)
                    {
                        PersistToDisk();
                        ClearWal();
                    }
                }
            }
            catch (Exception ex)
            {
                // Best-effort flush on dispose - log but don't throw
                System.Diagnostics.Debug.WriteLine($"FileDocumentStore: Flush on dispose failed: {ex.Message}");
            }

            CloseWalWriter();
        }

        _disposed = true;
    }

    #region Persistence - Disk I/O

    /// <summary>
    /// Persists all in-memory data to disk files.
    /// </summary>
    private void PersistToDisk()
    {
        var allDocuments = _store.ToArray();

        // Always write a snapshot, even when the store is empty.
        // For an empty store, this writes metadata with count = 0,
        // overwrites documents as an empty collection, and ensures vector
        // and index files are consistent with the empty state.
        WriteMetadata(allDocuments.Length);
        WriteDocuments(allDocuments);
        WriteVectors(allDocuments);
        WriteHnswGraph();
    }

    /// <summary>
    /// Loads all data from disk files into memory.
    /// </summary>
    private void LoadFromDisk()
    {
        // Read metadata to validate
        var meta = ReadMetadata();
        if (meta == null)
            return;

        if (meta.Version != FormatVersion)
        {
            throw new InvalidOperationException(
                $"Unsupported store format version {meta.Version}. Expected {FormatVersion}.");
        }

        _vectorDimension = meta.VectorDimension;

        // Read documents
        var documents = ReadDocuments();
        if (documents == null || documents.Count == 0)
            return;

        // Read vectors
        var vectors = ReadVectors(documents.Count, _vectorDimension);
        if (vectors == null)
            return;

        // Reconstruct in-memory store and HNSW index
        for (int i = 0; i < documents.Count; i++)
        {
            var doc = documents[i];
            var vector = vectors[i];
            var vd = new VectorDocument<T>(doc, vector);
            _store[doc.Id] = vd;
            _hnswIndex.Add(doc.Id, vector);
        }
    }

    /// <summary>
    /// Writes the store metadata header file.
    /// </summary>
    private void WriteMetadata(int documentCount)
    {
        var meta = new StoreMetadata
        {
            Version = FormatVersion,
            VectorDimension = _vectorDimension,
            DocumentCount = documentCount,
            HnswMaxConnections = _options.HnswMaxConnections,
            HnswEfConstruction = _options.HnswEfConstruction,
            HnswEfSearch = _options.HnswEfSearch
        };

        string json = JsonConvert.SerializeObject(meta, Formatting.Indented);
        File.WriteAllText(Path.Combine(_directoryPath, MetaFileName), json);
    }

    /// <summary>
    /// Reads the store metadata header file.
    /// </summary>
    private StoreMetadata? ReadMetadata()
    {
        string path = Path.Combine(_directoryPath, MetaFileName);
        if (!File.Exists(path))
            return null;

        try
        {
            string json = File.ReadAllText(path);
            if (string.IsNullOrWhiteSpace(json))
                return null;

            return JsonConvert.DeserializeObject<StoreMetadata>(json);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"FileDocumentStore: Failed to read metadata from '{path}': {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Writes all documents (content + metadata) as JSON.
    /// </summary>
    private void WriteDocuments(KeyValuePair<string, VectorDocument<T>>[] allDocuments)
    {
        var docs = allDocuments.Select(kvp => new SerializableDocument
        {
            Id = kvp.Value.Document.Id,
            Content = kvp.Value.Document.Content,
            Metadata = kvp.Value.Document.Metadata
        }).ToList();

        string json = JsonConvert.SerializeObject(docs, Formatting.None);
        File.WriteAllText(Path.Combine(_directoryPath, DocumentsFileName), json);
    }

    /// <summary>
    /// Reads all documents from the JSON file.
    /// </summary>
    private List<Document<T>>? ReadDocuments()
    {
        string path = Path.Combine(_directoryPath, DocumentsFileName);
        if (!File.Exists(path))
            return null;

        try
        {
            string json = File.ReadAllText(path);
            if (string.IsNullOrWhiteSpace(json))
                return null;

            var serializedDocs = JsonConvert.DeserializeObject<List<SerializableDocument>>(json);
            if (serializedDocs == null)
                return null;

            return serializedDocs.Select(sd => new Document<T>(sd.Id, sd.Content, sd.Metadata)).ToList();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"FileDocumentStore: Failed to read documents from '{path}': {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Writes all vectors as compact binary data.
    /// Format: [double, double, double, ...] per vector, sequentially.
    /// </summary>
    private void WriteVectors(KeyValuePair<string, VectorDocument<T>>[] allDocuments)
    {
        string path = Path.Combine(_directoryPath, VectorsFileName);
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var writer = new BinaryWriter(stream);

        foreach (var kvp in allDocuments)
        {
            var embedding = kvp.Value.Embedding;
            for (int i = 0; i < embedding.Length; i++)
            {
                double val = NumOps.ToDouble(embedding[i]);
                writer.Write(val);
            }
        }
    }

    /// <summary>
    /// Reads vectors from binary file.
    /// </summary>
    private List<Vector<T>>? ReadVectors(int documentCount, int vectorDimension)
    {
        string path = Path.Combine(_directoryPath, VectorsFileName);
        if (!File.Exists(path))
            return null;

        // Validate file has expected size to detect truncation from crash during write
        var fileInfo = new FileInfo(path);
        long expectedSize = (long)documentCount * vectorDimension * sizeof(double);
        if (fileInfo.Length < expectedSize)
            return null;

        var vectors = new List<Vector<T>>(documentCount);
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var reader = new BinaryReader(stream);

        for (int doc = 0; doc < documentCount; doc++)
        {
            var vector = new Vector<T>(vectorDimension);
            for (int i = 0; i < vectorDimension; i++)
            {
                double val = reader.ReadDouble();
                vector[i] = NumOps.FromDouble(val);
            }
            vectors.Add(vector);
        }

        return vectors;
    }

    /// <summary>
    /// Writes the HNSW graph marker file.
    /// Currently, the HNSW index is rebuilt from vectors on load for simplicity.
    /// A future optimization could serialize the graph adjacency lists directly
    /// to speed up load times for large stores.
    /// </summary>
    private void WriteHnswGraph()
    {
        string path = Path.Combine(_directoryPath, HnswFileName);
        File.WriteAllText(path, "HNSW_OK");
    }

    #endregion

    #region Write-Ahead Log

    /// <summary>
    /// Opens the WAL writer for append operations.
    /// </summary>
    private void OpenWalWriter()
    {
        string path = Path.Combine(_directoryPath, WalFileName);

        // When SyncWalWrites is enabled, use WriteThrough to guarantee data reaches
        // the physical disk (not just the OS buffer) on every write.
        var fileOptions = _options.SyncWalWrites ? FileOptions.WriteThrough : FileOptions.None;
        var stream = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read, 4096, fileOptions);
        _walSize = stream.Length;
        _walWriter = new StreamWriter(stream) { AutoFlush = true };
    }

    /// <summary>
    /// Closes the WAL writer.
    /// </summary>
    private void CloseWalWriter()
    {
        if (_walWriter != null)
        {
            try
            {
                _walWriter.Flush();
                _walWriter.Dispose();
            }
            catch (Exception ex)
            {
                // Best-effort close - log but don't throw
                System.Diagnostics.Debug.WriteLine($"FileDocumentStore: Failed to close WAL writer: {ex.Message}");
            }
            _walWriter = null;
        }
    }

    /// <summary>
    /// Writes a single entry to the WAL.
    /// </summary>
    private void WriteWalEntry(WalEntry entry)
    {
        lock (_walLock)
        {
            if (_walWriter == null)
            {
                throw new InvalidOperationException(
                    "WAL writer is not initialized. The store may have been disposed or not fully constructed.");
            }

            string json = JsonConvert.SerializeObject(entry, Formatting.None);
            _walWriter.WriteLine(json);
            _walWriter.Flush();
            _walSize = _walWriter.BaseStream.Length;
        }
    }

    /// <summary>
    /// Replays WAL entries to recover any uncommitted operations.
    /// </summary>
    private void ReplayWal()
    {
        string path = Path.Combine(_directoryPath, WalFileName);
        if (!File.Exists(path))
            return;

        StreamReader? reader;
        try
        {
            reader = new StreamReader(path);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"FileDocumentStore: WAL file read failed: {ex.Message}");
            return; // WAL file may be corrupted or locked
        }

        using (reader)
        {
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                WalEntry? entry;
                try
                {
                    entry = JsonConvert.DeserializeObject<WalEntry>(line);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"FileDocumentStore: Corrupted WAL entry skipped: {ex.Message}");
                    continue; // Skip corrupted WAL entries
                }

                if (entry == null)
                    continue;

                switch (entry.Operation)
                {
                    case WalOperation.Add:
                        if (entry.Document != null && entry.VectorData != null)
                        {
                            var doc = new Document<T>(entry.DocumentId, entry.Document.Content, entry.Document.Metadata);
                            var vector = DoubleArrayToVector(entry.VectorData);
                            var vd = new VectorDocument<T>(doc, vector);

                            bool alreadyExists = _store.ContainsKey(entry.DocumentId);
                            _tombstones.Remove(entry.DocumentId);
                            _store[entry.DocumentId] = vd;

                            // Remove from HNSW first if updating existing entry
                            if (alreadyExists)
                            {
                                _hnswIndex.Remove(entry.DocumentId);
                            }
                            _hnswIndex.Add(entry.DocumentId, vector);
                        }
                        break;

                    case WalOperation.Remove:
                        _store.TryRemove(entry.DocumentId, out _);
                        _hnswIndex.Remove(entry.DocumentId);
                        _tombstones.Add(entry.DocumentId);
                        break;
                }
            }
        }
    }

    /// <summary>
    /// Clears the WAL file after a successful flush.
    /// </summary>
    private void ClearWal()
    {
        CloseWalWriter();

        string path = Path.Combine(_directoryPath, WalFileName);

        // First, attempt to delete the WAL file.
        DeleteFileIfExists(path);

        // If deletion failed (e.g., file locked by AV scan), fall back to truncating it.
        try
        {
            if (File.Exists(path))
            {
                using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
                // FileMode.Create truncates the file to zero length
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"FileDocumentStore: Failed to truncate WAL file '{path}': {ex.Message}");
        }

        _walSize = 0;
        OpenWalWriter();
    }

    /// <summary>
    /// Checks if auto-flush or auto-compaction should be triggered.
    /// </summary>
    private void CheckAutoFlush()
    {
        // Flush on every write if configured for maximum durability
        if (_options.FlushOnEveryWrite)
        {
            Flush();
            return;
        }

        // Auto-flush when WAL exceeds max size
        long currentWalSize;
        int tombstoneCount;
        lock (_walLock)
        {
            currentWalSize = _walSize;
            tombstoneCount = _tombstones.Count;
        }

        if (currentWalSize > _options.MaxWalSizeBytes)
        {
            Flush();
            return;
        }

        // Auto-compact when tombstone ratio exceeds threshold.
        // Only compact when there are live documents (compacting an empty store is pointless)
        // and when we have enough total entries to avoid thrashing on small stores.
        int liveCount = _store.Count;
        int totalDocs = liveCount + tombstoneCount;
        if (liveCount > 0 && tombstoneCount > 0 && totalDocs >= _options.MinimumDocumentCountForCompaction)
        {
            double ratio = (double)tombstoneCount / totalDocs;
            if (ratio >= _options.CompactionTombstoneRatio)
            {
                Compact();
            }
        }
    }

    /// <summary>
    /// Throws ObjectDisposedException if the store has been disposed.
    /// </summary>
    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(FileDocumentStore<T>));
        }
    }

    #endregion

    #region Helpers

    /// <summary>
    /// Converts a Vector&lt;T&gt; to a double array for WAL serialization.
    /// </summary>
    private double[] VectorToDoubleArray(Vector<T> vector)
    {
        var result = new double[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = NumOps.ToDouble(vector[i]);
        }
        return result;
    }

    /// <summary>
    /// Converts a double array back to a Vector&lt;T&gt;.
    /// </summary>
    private Vector<T> DoubleArrayToVector(double[] data)
    {
        var vector = new Vector<T>(data.Length);
        for (int i = 0; i < data.Length; i++)
        {
            vector[i] = NumOps.FromDouble(data[i]);
        }
        return vector;
    }

    /// <summary>
    /// Deletes a file if it exists, ignoring errors.
    /// </summary>
    private static void DeleteFileIfExists(string path)
    {
        try
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"FileDocumentStore: Failed to delete file '{path}': {ex.Message}");
        }
    }

    #endregion

    #region Internal Types

    /// <summary>
    /// Store metadata header persisted to disk.
    /// </summary>
    private class StoreMetadata
    {
        public int Version { get; set; }
        public int VectorDimension { get; set; }
        public int DocumentCount { get; set; }
        public int HnswMaxConnections { get; set; }
        public int HnswEfConstruction { get; set; }
        public int HnswEfSearch { get; set; }
    }

    /// <summary>
    /// Simplified document for JSON serialization (without generic T fields).
    /// </summary>
    private class SerializableDocument
    {
        public string Id { get; set; } = string.Empty;
        public string Content { get; set; } = string.Empty;
        public Dictionary<string, object> Metadata { get; set; } = new();
    }

    /// <summary>
    /// Write-ahead log entry.
    /// </summary>
    private class WalEntry
    {
        public WalOperation Operation { get; set; }
        public string DocumentId { get; set; } = string.Empty;
        public SerializableDocument? Document { get; set; }
        public double[]? VectorData { get; set; }
    }

    /// <summary>
    /// WAL operation types.
    /// </summary>
    private enum WalOperation
    {
        Add = 0,
        Remove = 1
    }

    #endregion
}
