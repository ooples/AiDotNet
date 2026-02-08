using System.IO.Compression;
using System.Security.Cryptography;

namespace AiDotNet.Data.Pipeline;

/// <summary>
/// Persists an entire processed pipeline to disk for fast reload across epochs,
/// with automatic invalidation when source data or pipeline configuration changes.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Inspired by TensorFlow's tf.data snapshot pattern. After the first epoch processes
/// data through expensive transforms (decode, resize, augment, tokenize), the entire
/// output is saved to disk. Subsequent epochs load the cached result directly,
/// skipping all preprocessing.
/// </para>
/// <para>
/// Cache invalidation is automatic: the pipeline configuration and source data metadata
/// are hashed, and the cache is rebuilt when the hash changes.
/// </para>
/// <para><b>For Beginners:</b> Imagine you have a pipeline that reads images, resizes them,
/// and normalizes pixel values. On the first run, this is slow. SnapshotPipeline saves the
/// processed results so that the second run is instant:
/// <code>
/// var pipeline = new DataPipeline&lt;Tensor&lt;float&gt;&gt;(loadImages)
///     .Map(img => Resize(img, 224, 224))
///     .Map(img => Normalize(img));
///
/// var snapshot = new SnapshotPipeline&lt;float&gt;(pipeline, new DiskCacheOptions
/// {
///     CacheDirectory = "./cache/images"
/// });
///
/// // First epoch: processes and caches all data
/// // Second epoch: loads from cache (instant)
/// foreach (var batch in snapshot.GetCachedPipeline())
/// {
///     model.Train(batch);
/// }
/// </code>
/// </para>
/// </remarks>
public class SnapshotPipeline<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly DataPipeline<Tensor<T>> _sourcePipeline;
    private readonly DiskCacheOptions _options;
    private readonly string _pipelineHash;
    private string? _activeCacheDir;

    /// <summary>
    /// Gets whether the cache is currently valid and up-to-date.
    /// </summary>
    public bool IsCacheValid => _activeCacheDir is not null && IsCacheDirectoryValid(_activeCacheDir);

    /// <summary>
    /// Gets the path to the active cache directory, or null if no cache exists.
    /// </summary>
    public string? ActiveCacheDirectory => _activeCacheDir;

    /// <summary>
    /// Creates a new snapshot pipeline.
    /// </summary>
    /// <param name="sourcePipeline">The data pipeline to cache.</param>
    /// <param name="options">Cache configuration options.</param>
    /// <param name="pipelineId">Optional unique identifier for this pipeline configuration.
    /// If null, a hash is computed from the pipeline structure.</param>
    public SnapshotPipeline(
        DataPipeline<Tensor<T>> sourcePipeline,
        DiskCacheOptions? options = null,
        string? pipelineId = null)
    {
        _sourcePipeline = sourcePipeline ?? throw new ArgumentNullException(nameof(sourcePipeline));
        _options = options ?? new DiskCacheOptions();
        _pipelineHash = pipelineId ?? ComputePipelineHash();

        // Check for existing valid cache
        string candidateDir = Path.Combine(_options.CacheDirectory, _pipelineHash);
        if (Directory.Exists(candidateDir) && IsCacheDirectoryValid(candidateDir))
        {
            _activeCacheDir = candidateDir;
        }
    }

    /// <summary>
    /// Gets a pipeline that reads from cache if available, or processes and caches the source pipeline.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A data pipeline reading from the snapshot cache.</returns>
    public DataPipeline<Tensor<T>> GetCachedPipeline(CancellationToken cancellationToken = default)
    {
        if (_activeCacheDir is not null && IsCacheDirectoryValid(_activeCacheDir))
        {
            return CreateReadPipeline(_activeCacheDir);
        }

        // Build cache from source
        string cacheDir = Path.Combine(_options.CacheDirectory, _pipelineHash);
        BuildCache(cacheDir, cancellationToken);
        _activeCacheDir = cacheDir;

        return CreateReadPipeline(cacheDir);
    }

    /// <summary>
    /// Forces a rebuild of the cache, even if a valid cache exists.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    public void RebuildCache(CancellationToken cancellationToken = default)
    {
        string cacheDir = Path.Combine(_options.CacheDirectory, _pipelineHash);

        // Clean existing cache
        if (Directory.Exists(cacheDir))
        {
            Directory.Delete(cacheDir, true);
        }

        BuildCache(cacheDir, cancellationToken);
        _activeCacheDir = cacheDir;
    }

    /// <summary>
    /// Invalidates and removes the cache.
    /// </summary>
    public void InvalidateCache()
    {
        if (_activeCacheDir is not null && Directory.Exists(_activeCacheDir))
        {
            Directory.Delete(_activeCacheDir, true);
        }

        _activeCacheDir = null;
    }

    /// <summary>
    /// Gets information about the current cache state.
    /// </summary>
    /// <returns>Cache information including size, entry count, and validity.</returns>
    public CacheInfo GetCacheInfo()
    {
        if (_activeCacheDir is null || !Directory.Exists(_activeCacheDir))
        {
            return new CacheInfo
            {
                IsValid = false,
                EntryCount = 0,
                TotalSizeBytes = 0,
                CacheDirectory = _activeCacheDir ?? string.Empty
            };
        }

        var files = Directory.GetFiles(_activeCacheDir, "*.bin");
        long totalSize = 0;
        foreach (var file in files)
        {
            totalSize += new FileInfo(file).Length;
        }

        return new CacheInfo
        {
            IsValid = IsCacheDirectoryValid(_activeCacheDir),
            EntryCount = files.Length,
            TotalSizeBytes = totalSize,
            CacheDirectory = _activeCacheDir
        };
    }

    /// <summary>
    /// Cleans up old cache entries across all pipeline caches based on the eviction policy.
    /// </summary>
    public void CleanupCache()
    {
        if (!Directory.Exists(_options.CacheDirectory)) return;

        if (_options.MaxAge.HasValue)
        {
            CleanupByAge(_options.MaxAge.Value);
        }

        if (_options.MaxCacheSizeBytes > 0)
        {
            CleanupBySize(_options.MaxCacheSizeBytes);
        }
    }

    private void BuildCache(string cacheDir, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(cacheDir);

        int index = 0;
        foreach (var tensor in _sourcePipeline)
        {
            cancellationToken.ThrowIfCancellationRequested();

            string filePath = Path.Combine(cacheDir, $"{index:D8}.bin");
            WriteTensorToFile(filePath, tensor);
            index++;
        }

        // Write metadata
        WriteMetadata(cacheDir, index);
    }

    private DataPipeline<Tensor<T>> CreateReadPipeline(string cacheDir)
    {
        return new DataPipeline<Tensor<T>>(() => ReadCacheEntries(cacheDir));
    }

    private IEnumerable<Tensor<T>> ReadCacheEntries(string cacheDir)
    {
        string metaPath = Path.Combine(cacheDir, "metadata.txt");
        if (!File.Exists(metaPath))
            throw new InvalidOperationException($"Cache metadata not found in {cacheDir}");

        string[] metaLines = File.ReadAllLines(metaPath);
        int entryCount = 0;
        foreach (string line in metaLines)
        {
            if (line.StartsWith("count=", StringComparison.Ordinal))
            {
                entryCount = int.Parse(line.Substring(6));
            }
        }

        for (int i = 0; i < entryCount; i++)
        {
            string filePath = Path.Combine(cacheDir, $"{i:D8}.bin");
            if (File.Exists(filePath))
            {
                yield return ReadTensorFromFile(filePath);
            }
        }
    }

    private void WriteTensorToFile(string filePath, Tensor<T> tensor)
    {
        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        Stream writeStream = stream;

        if (_options.CompressData)
        {
            writeStream = new GZipStream(stream, CompressionLevel.Fastest, leaveOpen: true);
        }

        using var writer = new BinaryWriter(writeStream);

        // Write shape
        writer.Write(tensor.Shape.Length);
        for (int d = 0; d < tensor.Shape.Length; d++)
        {
            writer.Write(tensor.Shape[d]);
        }

        // Write data as doubles
        var span = tensor.Data.Span;
        writer.Write(span.Length);
        for (int i = 0; i < span.Length; i++)
        {
            writer.Write(NumOps.ToDouble(span[i]));
        }

        if (_options.CompressData && writeStream is GZipStream gzip)
        {
            gzip.Dispose();
        }

        // Write checksum if integrity verification enabled
        if (_options.VerifyIntegrity)
        {
            string checksumPath = filePath + ".sha256";
            stream.Position = 0;
            byte[] hash = ComputeSha256(stream);
            File.WriteAllText(checksumPath, Convert.ToBase64String(hash));
        }
    }

    private Tensor<T> ReadTensorFromFile(string filePath)
    {
        // Verify integrity if enabled
        if (_options.VerifyIntegrity)
        {
            string checksumPath = filePath + ".sha256";
            if (File.Exists(checksumPath))
            {
                string expectedHash = File.ReadAllText(checksumPath);
                using var checkStream = File.OpenRead(filePath);
                byte[] actualHash = ComputeSha256(checkStream);
                string actualHashStr = Convert.ToBase64String(actualHash);

                if (!string.Equals(expectedHash, actualHashStr, StringComparison.Ordinal))
                {
                    throw new InvalidDataException(
                        $"Cache file integrity check failed for {filePath}. " +
                        "Cache may be corrupted. Delete the cache directory and retry.");
                }
            }
        }

        using var stream = File.OpenRead(filePath);
        Stream readStream = stream;

        if (_options.CompressData)
        {
            readStream = new GZipStream(stream, CompressionMode.Decompress, leaveOpen: true);
        }

        using var reader = new BinaryReader(readStream);

        // Read shape
        int shapeDims = reader.ReadInt32();
        var shape = new int[shapeDims];
        for (int d = 0; d < shapeDims; d++)
        {
            shape[d] = reader.ReadInt32();
        }

        // Read data
        int dataLength = reader.ReadInt32();
        var data = new T[dataLength];
        for (int i = 0; i < dataLength; i++)
        {
            data[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        return new Tensor<T>(data, shape);
    }

    private void WriteMetadata(string cacheDir, int entryCount)
    {
        string metaPath = Path.Combine(cacheDir, "metadata.txt");
        var lines = new[]
        {
            $"count={entryCount}",
            $"hash={_pipelineHash}",
            $"created={DateTime.UtcNow:O}",
            $"compressed={_options.CompressData}"
        };
        File.WriteAllLines(metaPath, lines);
    }

    private bool IsCacheDirectoryValid(string cacheDir)
    {
        string metaPath = Path.Combine(cacheDir, "metadata.txt");
        if (!File.Exists(metaPath)) return false;

        string[] lines = File.ReadAllLines(metaPath);

        // Check hash matches
        if (_options.AutoInvalidateOnSourceChange)
        {
            foreach (string line in lines)
            {
                if (line.StartsWith("hash=", StringComparison.Ordinal))
                {
                    string cachedHash = line.Substring(5);
                    if (!string.Equals(cachedHash, _pipelineHash, StringComparison.Ordinal))
                        return false;
                }
            }
        }

        // Check age
        if (_options.MaxAge.HasValue)
        {
            foreach (string line in lines)
            {
                if (line.StartsWith("created=", StringComparison.Ordinal))
                {
                    if (DateTime.TryParse(line.Substring(8), out DateTime created))
                    {
                        if (DateTime.UtcNow - created > _options.MaxAge.Value)
                            return false;
                    }
                }
            }
        }

        // Check that at least one data file exists
        int count = 0;
        foreach (string line in lines)
        {
            if (line.StartsWith("count=", StringComparison.Ordinal))
            {
                count = int.Parse(line.Substring(6));
            }
        }

        if (count > 0)
        {
            string firstFile = Path.Combine(cacheDir, "00000000.bin");
            if (!File.Exists(firstFile)) return false;
        }

        return true;
    }

    private string ComputePipelineHash()
    {
        // Deterministic hash based on pipeline type info so the same pipeline
        // configuration produces the same cache key across runs.
        string input = $"{typeof(T).FullName}_{_sourcePipeline.GetType().FullName}_{_options.CacheDirectory}";
        using var sha = SHA256.Create();
        byte[] hash = sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(input));
        return Convert.ToBase64String(hash).Replace("/", "_").Replace("+", "-").Substring(0, 16);
    }

    private static byte[] ComputeSha256(Stream stream)
    {
        using var sha = SHA256.Create();
        return sha.ComputeHash(stream);
    }

    private void CleanupByAge(TimeSpan maxAge)
    {
        foreach (string subDir in Directory.GetDirectories(_options.CacheDirectory))
        {
            string metaPath = Path.Combine(subDir, "metadata.txt");
            if (!File.Exists(metaPath)) continue;

            string[] lines = File.ReadAllLines(metaPath);
            foreach (string line in lines)
            {
                if (line.StartsWith("created=", StringComparison.Ordinal))
                {
                    if (DateTime.TryParse(line.Substring(8), out DateTime created))
                    {
                        if (DateTime.UtcNow - created > maxAge)
                        {
                            Directory.Delete(subDir, true);
                        }
                    }
                }
            }
        }
    }

    private void CleanupBySize(long maxSizeBytes)
    {
        var cacheEntries = new List<(string Dir, long Size, DateTime Created)>();
        long totalSize = 0;

        foreach (string subDir in Directory.GetDirectories(_options.CacheDirectory))
        {
            long dirSize = 0;
            DateTime created = DateTime.MinValue;

            foreach (string file in Directory.GetFiles(subDir))
            {
                dirSize += new FileInfo(file).Length;
            }

            string metaPath = Path.Combine(subDir, "metadata.txt");
            if (File.Exists(metaPath))
            {
                string[] lines = File.ReadAllLines(metaPath);
                foreach (string line in lines)
                {
                    if (line.StartsWith("created=", StringComparison.Ordinal))
                    {
                        DateTime.TryParse(line.Substring(8), out created);
                    }
                }
            }

            cacheEntries.Add((subDir, dirSize, created));
            totalSize += dirSize;
        }

        if (totalSize <= maxSizeBytes) return;

        // Sort by eviction policy
        switch (_options.EvictionPolicy)
        {
            case CacheEvictionPolicy.OldestFirst:
                cacheEntries.Sort((a, b) => a.Created.CompareTo(b.Created));
                break;
            case CacheEvictionPolicy.LargestFirst:
                cacheEntries.Sort((a, b) => b.Size.CompareTo(a.Size));
                break;
            case CacheEvictionPolicy.LeastRecentlyUsed:
            default:
                // Use last write time of metadata as proxy for last access
                cacheEntries.Sort((a, b) =>
                {
                    string metaA = Path.Combine(a.Dir, "metadata.txt");
                    string metaB = Path.Combine(b.Dir, "metadata.txt");
                    DateTime accessA = File.Exists(metaA) ? File.GetLastWriteTimeUtc(metaA) : DateTime.MinValue;
                    DateTime accessB = File.Exists(metaB) ? File.GetLastWriteTimeUtc(metaB) : DateTime.MinValue;
                    return accessA.CompareTo(accessB);
                });
                break;
        }

        // Evict entries until within budget (don't evict active cache)
        foreach (var (dir, size, _) in cacheEntries)
        {
            if (totalSize <= maxSizeBytes) break;

            // Don't evict the active cache
            if (string.Equals(dir, _activeCacheDir, StringComparison.OrdinalIgnoreCase))
                continue;

            Directory.Delete(dir, true);
            totalSize -= size;
        }
    }
}

