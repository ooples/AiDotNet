namespace AiDotNet.Inference.PagedAttention;

/// <summary>
/// Paged key-value cache for efficient LLM serving memory management.
/// </summary>
/// <remarks>
/// <para>
/// PagedKVCache implements the vLLM-style paged attention memory management system.
/// Instead of pre-allocating contiguous memory for each sequence's maximum length,
/// it dynamically allocates fixed-size blocks as sequences grow.
/// </para>
/// <para><b>For Beginners:</b> Traditional KV-cache is like reserving a whole hotel floor per guest.
///
/// PagedKVCache is like renting hotel rooms individually:
/// - Guest arrives: Get them a room (allocate 1 block)
/// - Guest needs more space: Give them another room (allocate more blocks)
/// - Guest leaves: Rooms become available (free blocks)
///
/// Benefits:
/// - 8-9x more sequences can fit in memory
/// - No wasted space for short sequences
/// - Efficient beam search with copy-on-write
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor computations.</typeparam>
internal class PagedKVCache<T> : IDisposable
{
    private readonly PagedKVCacheConfig _config;
    private readonly BlockManager<T> _blockManager;
    private readonly BlockTableManager<T> _blockTableManager;

    // Physical storage for K and V tensors
    // Shape: [num_blocks, num_layers, 2 (K/V), block_size, num_heads, head_dim]
    private readonly T[] _kvStorage;
    private readonly long _elementsPerBlock;

    // Tracking
    private readonly Dictionary<long, SequenceMetadata> _sequenceMetadata;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// Gets the configuration.
    /// </summary>
    public PagedKVCacheConfig Config => _config;

    /// <summary>
    /// Gets the block manager.
    /// </summary>
    public BlockManager<T> BlockManager => _blockManager;

    /// <summary>
    /// Gets the block table manager.
    /// </summary>
    public BlockTableManager<T> BlockTableManager => _blockTableManager;

    /// <summary>
    /// Gets the number of active sequences.
    /// </summary>
    public int ActiveSequenceCount
    {
        get { lock (_lock) return _sequenceMetadata.Count; }
    }

    /// <summary>
    /// Creates a new paged KV cache.
    /// </summary>
    public PagedKVCache(PagedKVCacheConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));

        // Create block manager
        var blockConfig = new BlockManagerConfig
        {
            BlockSize = config.BlockSize,
            NumBlocks = config.NumBlocks,
            NumLayers = config.NumLayers,
            NumHeads = config.NumHeads,
            HeadDimension = config.HeadDimension
        };
        _blockManager = new BlockManager<T>(blockConfig);
        _blockTableManager = new BlockTableManager<T>(_blockManager);

        // Calculate elements per block
        // Each block stores: block_size tokens x num_layers x 2 (K,V) x num_heads x head_dim
        _elementsPerBlock = (long)config.BlockSize * config.NumLayers * 2 * config.NumHeads * config.HeadDimension;

        // Allocate physical storage
        long totalElements = _elementsPerBlock * config.NumBlocks;
        if (totalElements > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(config), $"PagedKVCache requires totalElements <= {int.MaxValue}, but got {totalElements}. Reduce NumBlocks or memory size.");

        try
        {
            _kvStorage = new T[(int)totalElements];
        }
        catch (OutOfMemoryException ex)
        {
            throw new InvalidOperationException(
                $"Failed to allocate PagedKVCache storage ({totalElements} elements). " +
                "This can happen when requesting very large contiguous memory blocks (e.g., multi-GB) in environments with tighter single-object limits. " +
                "Reduce available memory/NumBlocks or use a runtime that supports larger allocations.",
                ex);
        }

        _sequenceMetadata = new Dictionary<long, SequenceMetadata>();
    }

    /// <summary>
    /// Creates a new paged KV cache from memory size.
    /// </summary>
    public static PagedKVCache<T> FromMemorySize(
        long availableBytes,
        int numLayers,
        int numHeads,
        int headDim,
        int blockSize = 16)
    {
        var config = PagedKVCacheConfig.FromMemorySize(
            availableBytes, numLayers, numHeads, headDim, blockSize);
        return new PagedKVCache<T>(config);
    }

    /// <summary>
    /// Allocates cache space for a new sequence.
    /// </summary>
    /// <param name="sequenceId">The sequence ID.</param>
    /// <param name="initialTokens">Number of initial tokens (e.g., prompt length).</param>
    /// <returns>True if allocation succeeded.</returns>
    public bool AllocateSequence(long sequenceId, int initialTokens)
    {
        lock (_lock)
        {
            if (_sequenceMetadata.ContainsKey(sequenceId))
                return false;

            // Allocate at least one block up-front so the first token write (position 0) always has capacity.
            // Actual "current length" bookkeeping still starts at initialTokens.
            int blocksNeeded = _blockManager.BlocksForTokens(Math.Max(1, initialTokens));
            blocksNeeded = Math.Max(1, blocksNeeded);
            var table = _blockTableManager.CreateBlockTable(sequenceId, blocksNeeded);

            if (table == null)
                return false;

            _sequenceMetadata[sequenceId] = new SequenceMetadata
            {
                SequenceId = sequenceId,
                CurrentLength = initialTokens,
                CreatedAt = DateTime.UtcNow
            };

            return true;
        }
    }

    /// <summary>
    /// Extends a sequence's cache for additional tokens.
    /// </summary>
    /// <param name="sequenceId">The sequence ID.</param>
    /// <param name="additionalTokens">Number of additional tokens.</param>
    /// <returns>True if extension succeeded.</returns>
    public bool ExtendSequence(long sequenceId, int additionalTokens)
    {
        lock (_lock)
        {
            if (!_sequenceMetadata.TryGetValue(sequenceId, out var metadata))
                return false;

            int newLength = metadata.CurrentLength + additionalTokens;
            if (!_blockTableManager.EnsureCapacity(sequenceId, newLength))
                return false;

            metadata.CurrentLength = newLength;
            return true;
        }
    }

    /// <summary>
    /// Frees all cache blocks for a sequence.
    /// </summary>
    public void FreeSequence(long sequenceId)
    {
        lock (_lock)
        {
            _blockTableManager.FreeBlockTable(sequenceId);
            _sequenceMetadata.Remove(sequenceId);
        }
    }

    /// <summary>
    /// Forks a sequence's cache for beam search.
    /// </summary>
    /// <param name="sourceSequenceId">The source sequence ID.</param>
    /// <param name="newSequenceId">The new sequence ID.</param>
    /// <returns>True if fork succeeded.</returns>
    public bool ForkSequence(long sourceSequenceId, long newSequenceId)
    {
        lock (_lock)
        {
            if (!_sequenceMetadata.TryGetValue(sourceSequenceId, out var sourceMetadata))
                return false;

            var forkedTable = _blockTableManager.ForkBlockTable(sourceSequenceId, newSequenceId);
            if (forkedTable == null)
                return false;

            _sequenceMetadata[newSequenceId] = new SequenceMetadata
            {
                SequenceId = newSequenceId,
                CurrentLength = sourceMetadata.CurrentLength,
                CreatedAt = DateTime.UtcNow,
                ParentSequenceId = sourceSequenceId
            };

            return true;
        }
    }

    /// <summary>
    /// Gets the storage offset for a specific position in the cache.
    /// </summary>
    /// <param name="blockId">Physical block ID.</param>
    /// <param name="layer">Layer index.</param>
    /// <param name="isValue">True for V, false for K.</param>
    /// <param name="tokenOffset">Offset within the block.</param>
    /// <param name="head">Head index.</param>
    /// <returns>The offset into the storage array.</returns>
    public long GetStorageOffset(int blockId, int layer, bool isValue, int tokenOffset, int head)
    {
        // Layout: [block][layer][kv][token][head][dim]
        long offset = blockId * _elementsPerBlock;
        offset += (long)layer * 2 * _config.BlockSize * _config.NumHeads * _config.HeadDimension;
        offset += (isValue ? 1 : 0) * (long)_config.BlockSize * _config.NumHeads * _config.HeadDimension;
        offset += (long)tokenOffset * _config.NumHeads * _config.HeadDimension;
        offset += (long)head * _config.HeadDimension;
        return offset;
    }

    /// <summary>
    /// Writes key tensor for a token position.
    /// </summary>
    public void WriteKey(long sequenceId, int tokenPosition, int layer, ReadOnlySpan<T> keyData)
    {
        var table = _blockTableManager.GetBlockTable(sequenceId);
        if (table == null)
            throw new InvalidOperationException($"No block table for sequence {sequenceId}");

        var (blockId, offset) = table.GetBlockAndOffset(tokenPosition);

        // Check for copy-on-write
        if (_blockManager.GetReferenceCount(blockId) > 1)
        {
            _blockTableManager.CopyOnWrite(sequenceId, tokenPosition / _config.BlockSize, CopyBlockData);
            table = _blockTableManager.GetBlockTable(sequenceId)!;
            (blockId, offset) = table.GetBlockAndOffset(tokenPosition);
        }

        // Write key data for all heads
        for (int head = 0; head < _config.NumHeads; head++)
        {
            long storageOffset = GetStorageOffset(blockId, layer, isValue: false, offset, head);
            int dataOffset = head * _config.HeadDimension;
            keyData.Slice(dataOffset, _config.HeadDimension).CopyTo(
                _kvStorage.AsSpan((int)storageOffset, _config.HeadDimension));
        }
    }

    /// <summary>
    /// Writes value tensor for a token position.
    /// </summary>
    public void WriteValue(long sequenceId, int tokenPosition, int layer, ReadOnlySpan<T> valueData)
    {
        var table = _blockTableManager.GetBlockTable(sequenceId);
        if (table == null)
            throw new InvalidOperationException($"No block table for sequence {sequenceId}");

        var (blockId, offset) = table.GetBlockAndOffset(tokenPosition);

        // Check for copy-on-write
        if (_blockManager.GetReferenceCount(blockId) > 1)
        {
            _blockTableManager.CopyOnWrite(sequenceId, tokenPosition / _config.BlockSize, CopyBlockData);
            table = _blockTableManager.GetBlockTable(sequenceId)!;
            (blockId, offset) = table.GetBlockAndOffset(tokenPosition);
        }

        // Write value data for all heads
        for (int head = 0; head < _config.NumHeads; head++)
        {
            long storageOffset = GetStorageOffset(blockId, layer, isValue: true, offset, head);
            int dataOffset = head * _config.HeadDimension;
            valueData.Slice(dataOffset, _config.HeadDimension).CopyTo(
                _kvStorage.AsSpan((int)storageOffset, _config.HeadDimension));
        }
    }

    /// <summary>
    /// Reads key tensor for a token position.
    /// </summary>
    public void ReadKey(long sequenceId, int tokenPosition, int layer, Span<T> keyData)
    {
        var table = _blockTableManager.GetBlockTable(sequenceId);
        if (table == null)
            throw new InvalidOperationException($"No block table for sequence {sequenceId}");

        var (blockId, offset) = table.GetBlockAndOffset(tokenPosition);

        for (int head = 0; head < _config.NumHeads; head++)
        {
            long storageOffset = GetStorageOffset(blockId, layer, isValue: false, offset, head);
            int dataOffset = head * _config.HeadDimension;
            _kvStorage.AsSpan((int)storageOffset, _config.HeadDimension).CopyTo(
                keyData.Slice(dataOffset, _config.HeadDimension));
        }
    }

    /// <summary>
    /// Reads value tensor for a token position.
    /// </summary>
    public void ReadValue(long sequenceId, int tokenPosition, int layer, Span<T> valueData)
    {
        var table = _blockTableManager.GetBlockTable(sequenceId);
        if (table == null)
            throw new InvalidOperationException($"No block table for sequence {sequenceId}");

        var (blockId, offset) = table.GetBlockAndOffset(tokenPosition);

        for (int head = 0; head < _config.NumHeads; head++)
        {
            long storageOffset = GetStorageOffset(blockId, layer, isValue: true, offset, head);
            int dataOffset = head * _config.HeadDimension;
            _kvStorage.AsSpan((int)storageOffset, _config.HeadDimension).CopyTo(
                valueData.Slice(dataOffset, _config.HeadDimension));
        }
    }

    /// <summary>
    /// Gets the block table for a sequence (for paged attention kernel).
    /// </summary>
    public int[]? GetBlockTable(long sequenceId)
    {
        return _blockTableManager.GetBlockTableArray(sequenceId);
    }

    /// <summary>
    /// Gets the current length of a sequence.
    /// </summary>
    public int GetSequenceLength(long sequenceId)
    {
        lock (_lock)
        {
            return _sequenceMetadata.TryGetValue(sequenceId, out var metadata) ? metadata.CurrentLength : 0;
        }
    }

    /// <summary>
    /// Checks if more tokens can be added to a sequence without new allocation.
    /// </summary>
    public bool HasCapacityFor(long sequenceId, int additionalTokens)
    {
        lock (_lock)
        {
            if (!_sequenceMetadata.TryGetValue(sequenceId, out var metadata))
                return false;

            var table = _blockTableManager.GetBlockTable(sequenceId);
            if (table == null)
                return false;

            int newLength = metadata.CurrentLength + additionalTokens;
            int blocksNeeded = _blockManager.BlocksForTokens(newLength);
            int additionalBlocks = blocksNeeded - table.NumLogicalBlocks;

            return additionalBlocks <= 0 || _blockManager.CanAllocate(additionalBlocks);
        }
    }

    /// <summary>
    /// Gets statistics about the cache.
    /// </summary>
    public PagedKVCacheStats GetStats()
    {
        lock (_lock)
        {
            var blockStats = _blockManager.GetStats();
            long totalTokens = _sequenceMetadata.Values.Sum(m => m.CurrentLength);

            return new PagedKVCacheStats
            {
                ActiveSequences = _sequenceMetadata.Count,
                TotalTokensCached = totalTokens,
                BlockStats = blockStats,
                AverageSequenceLength = _sequenceMetadata.Count > 0
                    ? (double)totalTokens / _sequenceMetadata.Count
                    : 0,
                MemoryEfficiency = CalculateMemoryEfficiency()
            };
        }
    }

    /// <summary>
    /// Gets the underlying storage array (for GPU transfer).
    /// </summary>
    public T[] GetStorage() => _kvStorage;

    /// <summary>
    /// Releases resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        lock (_lock)
        {
            _sequenceMetadata.Clear();
            _blockTableManager.Clear();
        }
    }

    private void CopyBlockData(int sourceBlockId, int destBlockId)
    {
        long sourceOffset = sourceBlockId * _elementsPerBlock;
        long destOffset = destBlockId * _elementsPerBlock;

        Array.Copy(_kvStorage, sourceOffset, _kvStorage, destOffset, _elementsPerBlock);
    }

    private double CalculateMemoryEfficiency()
    {
        // Calculate how efficiently we're using memory compared to traditional allocation
        lock (_lock)
        {
            if (_sequenceMetadata.Count == 0)
                return 1.0;

            // Traditional: each sequence reserves max_seq_len tokens
            long traditionalTokens = _sequenceMetadata.Count * _config.MaxSeqLen;

            // Paged: actual blocks allocated * block size
            var blockStats = _blockManager.GetStats();
            long pagedTokenCapacity = blockStats.AllocatedBlocks * _config.BlockSize;

            if (traditionalTokens == 0)
                return 1.0;

            return (double)pagedTokenCapacity / traditionalTokens;
        }
    }

    private class SequenceMetadata
    {
        public long SequenceId { get; set; }
        public int CurrentLength { get; set; }
        public DateTime CreatedAt { get; set; }
        public long? ParentSequenceId { get; set; }
    }
}

/// <summary>
/// Configuration for PagedKVCache.
/// </summary>
internal class PagedKVCacheConfig
{
    /// <summary>
    /// Number of tokens per block.
    /// </summary>
    public int BlockSize { get; set; } = 16;

    /// <summary>
    /// Total number of blocks.
    /// </summary>
    public int NumBlocks { get; set; } = 1024;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    public int NumLayers { get; set; } = 32;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    public int NumHeads { get; set; } = 32;

    /// <summary>
    /// Dimension of each head.
    /// </summary>
    public int HeadDimension { get; set; } = 128;

    /// <summary>
    /// Maximum sequence length (for efficiency calculation).
    /// </summary>
    public int MaxSeqLen { get; set; } = 2048;

    /// <summary>
    /// Creates configuration from available memory.
    /// </summary>
    public static PagedKVCacheConfig FromMemorySize(
        long availableBytes,
        int numLayers,
        int numHeads,
        int headDim,
        int blockSize = 16)
    {
        // Calculate bytes per block
        // Each block: block_size * num_layers * 2 (K,V) * num_heads * head_dim * sizeof(float)
        long bytesPerBlock = (long)blockSize * numLayers * 2 * numHeads * headDim * sizeof(float);
        int numBlocks = (int)(availableBytes / bytesPerBlock);

        return new PagedKVCacheConfig
        {
            BlockSize = blockSize,
            NumBlocks = numBlocks,
            NumLayers = numLayers,
            NumHeads = numHeads,
            HeadDimension = headDim
        };
    }

    /// <summary>
    /// Creates configuration for a specific model.
    /// </summary>
    public static PagedKVCacheConfig ForModel(string modelName, long availableBytes, int blockSize = 16)
    {
        return modelName.ToLowerInvariant() switch
        {
            "llama-7b" => FromMemorySize(availableBytes, 32, 32, 128, blockSize),
            "llama-13b" => FromMemorySize(availableBytes, 40, 40, 128, blockSize),
            "llama-70b" => FromMemorySize(availableBytes, 80, 64, 128, blockSize),
            "gpt-2" => FromMemorySize(availableBytes, 12, 12, 64, blockSize),
            "mistral-7b" => FromMemorySize(availableBytes, 32, 32, 128, blockSize),
            _ => FromMemorySize(availableBytes, 32, 32, 128, blockSize)
        };
    }
}

/// <summary>
/// Statistics about the paged KV cache.
/// </summary>
internal class PagedKVCacheStats
{
    /// <summary>Number of active sequences.</summary>
    public int ActiveSequences { get; set; }

    /// <summary>Total tokens currently cached.</summary>
    public long TotalTokensCached { get; set; }

    /// <summary>Average sequence length.</summary>
    public double AverageSequenceLength { get; set; }

    /// <summary>Memory efficiency compared to traditional allocation.</summary>
    public double MemoryEfficiency { get; set; }

    /// <summary>Underlying block manager statistics.</summary>
    public BlockManagerStats BlockStats { get; set; } = new();
}
