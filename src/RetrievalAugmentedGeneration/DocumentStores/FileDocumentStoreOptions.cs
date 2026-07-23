namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// Configuration options for the file-based document store.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how the file-based document store
/// operates, including where files are stored, HNSW index parameters, and WAL behavior.
/// The defaults work well for most use cases.</para>
/// </remarks>
public class FileDocumentStoreOptions
{
    /// <summary>
    /// Gets or sets the directory path where store files are written.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the folder on disk where all document store data
    /// is saved. It will be created automatically if it doesn't exist.</para>
    /// </remarks>
    public string DirectoryPath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the HNSW M parameter (max connections per node).
    /// Higher values improve recall but use more memory.
    /// </summary>
    public int HnswMaxConnections { get; set; } = 16;

    /// <summary>
    /// Gets or sets the HNSW efConstruction parameter (search depth during index building).
    /// Higher values build better indexes but slower.
    /// </summary>
    public int HnswEfConstruction { get; set; } = 200;

    /// <summary>
    /// Gets or sets the HNSW efSearch parameter (search depth during queries).
    /// Higher values give better recall but slower search.
    /// </summary>
    public int HnswEfSearch { get; set; } = 50;

    /// <summary>
    /// Gets or sets the maximum WAL size in bytes before auto-flushing to main files.
    /// Default is 10 MB.
    /// </summary>
    public long MaxWalSizeBytes { get; set; } = 10 * 1024 * 1024;

    /// <summary>
    /// Gets or sets the ratio of tombstones to total documents that triggers automatic compaction.
    /// Default is 0.3 (30%).
    /// </summary>
    public double CompactionTombstoneRatio { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets whether to flush data to disk after every write operation.
    /// When true, ensures durability at the cost of write performance.
    /// Default is false (WAL provides durability).
    /// </summary>
    public bool FlushOnEveryWrite { get; set; }

    /// <summary>
    /// Gets or sets whether WAL writes should be synchronous (fsync on every write).
    /// When true, uses <see cref="System.IO.FileOptions.WriteThrough"/> to guarantee data
    /// reaches the physical disk, not just the OS buffer. Default is false for better performance.
    /// </summary>
    public bool SyncWalWrites { get; set; }

    /// <summary>
    /// Gets or sets the minimum total document count (live + tombstones) required before
    /// automatic compaction is considered. Prevents thrashing on small stores.
    /// Default is 10.
    /// </summary>
    public int MinimumDocumentCountForCompaction { get; set; } = 10;

    /// <summary>
    /// Gets or sets the random seed for the HNSW index.
    /// Use a fixed value for reproducible behavior.
    /// </summary>
    public int HnswSeed { get; set; } = 42;
}
