namespace AiDotNet.Inference.PagedAttention;

/// <summary>
/// Manages physical memory blocks for PagedAttention KV cache.
/// </summary>
/// <remarks>
/// <para>
/// The BlockManager maintains a pool of fixed-size memory blocks that can be
/// dynamically allocated and freed. This enables efficient memory utilization
/// by avoiding pre-allocation of maximum sequence length for each request.
/// </para>
/// <para><b>For Beginners:</b> Think of memory management like a parking lot.
///
/// Traditional KV-cache: Each car (request) gets a reserved section of spaces
/// equal to the maximum parking time, even if they leave early. Wasteful!
///
/// PagedAttention: Cars share a common pool of parking spaces. When they arrive,
/// they get spaces from the free pool. When they leave, spaces return to the pool.
///
/// Benefits:
/// - No wasted space for requests shorter than max length
/// - Can serve more requests with the same memory
/// - Memory is only allocated when needed
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor computations.</typeparam>
internal class BlockManager<T>
{
    private readonly BlockManagerConfig _config;
    private readonly object _lock = new();

    // Physical block pool
    private readonly Queue<int> _freeBlocks;
    private readonly HashSet<int> _allocatedBlocks;

    // Block reference counting for copy-on-write
    private readonly Dictionary<int, int> _refCounts;

    // Statistics
    private long _totalAllocations;
    private long _totalFrees;
    private long _copyOnWriteCount;

    /// <summary>
    /// Gets the configuration.
    /// </summary>
    public BlockManagerConfig Config => _config;

    /// <summary>
    /// Gets the number of free blocks available.
    /// </summary>
    public int FreeBlockCount
    {
        get { lock (_lock) return _freeBlocks.Count; }
    }

    /// <summary>
    /// Gets the number of allocated blocks.
    /// </summary>
    public int AllocatedBlockCount
    {
        get { lock (_lock) return _allocatedBlocks.Count; }
    }

    /// <summary>
    /// Gets the total number of blocks.
    /// </summary>
    public int TotalBlocks => _config.NumBlocks;

    /// <summary>
    /// Gets the memory utilization (0-1).
    /// </summary>
    public double MemoryUtilization
    {
        get { lock (_lock) return (double)_allocatedBlocks.Count / _config.NumBlocks; }
    }

    /// <summary>
    /// Creates a new block manager with the specified configuration.
    /// </summary>
    public BlockManager(BlockManagerConfig config)
    {
        Guard.NotNull(config);
        _config = config;

        _freeBlocks = new Queue<int>(_config.NumBlocks);
        _allocatedBlocks = new HashSet<int>();
        _refCounts = new Dictionary<int, int>();

        // Initialize all blocks as free
        for (int i = 0; i < _config.NumBlocks; i++)
        {
            _freeBlocks.Enqueue(i);
        }
    }

    /// <summary>
    /// Creates a block manager for a specific model configuration.
    /// </summary>
    public BlockManager(int totalMemoryBytes, int blockSize, int numLayers, int numHeads, int headDim)
        : this(BlockManagerConfig.FromMemorySize(totalMemoryBytes, blockSize, numLayers, numHeads, headDim))
    {
    }

    /// <summary>
    /// Allocates a single block.
    /// </summary>
    /// <returns>The block ID, or -1 if no blocks available.</returns>
    public int AllocateBlock()
    {
        lock (_lock)
        {
            if (_freeBlocks.Count == 0)
                return -1;

            int blockId = _freeBlocks.Dequeue();
            _allocatedBlocks.Add(blockId);
            _refCounts[blockId] = 1;
            _totalAllocations++;

            return blockId;
        }
    }

    /// <summary>
    /// Allocates multiple blocks.
    /// </summary>
    /// <param name="count">Number of blocks to allocate.</param>
    /// <returns>Array of block IDs, or null if not enough blocks available.</returns>
    public int[]? AllocateBlocks(int count)
    {
        lock (_lock)
        {
            if (_freeBlocks.Count < count)
                return null;

            var blocks = new int[count];
            for (int i = 0; i < count; i++)
            {
                blocks[i] = _freeBlocks.Dequeue();
                _allocatedBlocks.Add(blocks[i]);
                _refCounts[blocks[i]] = 1;
            }

            _totalAllocations += count;
            return blocks;
        }
    }

    /// <summary>
    /// Frees a single block.
    /// </summary>
    /// <param name="blockId">The block ID to free.</param>
    public void FreeBlock(int blockId)
    {
        lock (_lock)
        {
            if (!_allocatedBlocks.Contains(blockId))
                return;

            // Decrement reference count
            if (_refCounts.TryGetValue(blockId, out int count))
            {
                if (count > 1)
                {
                    _refCounts[blockId] = count - 1;
                    return; // Block still referenced
                }
            }

            _allocatedBlocks.Remove(blockId);
            _refCounts.Remove(blockId);
            _freeBlocks.Enqueue(blockId);
            _totalFrees++;
        }
    }

    /// <summary>
    /// Frees multiple blocks.
    /// </summary>
    /// <param name="blockIds">The block IDs to free.</param>
    public void FreeBlocks(IEnumerable<int> blockIds)
    {
        lock (_lock)
        {
            foreach (int blockId in blockIds)
            {
                if (!_allocatedBlocks.Contains(blockId))
                    continue;

                // Decrement reference count
                if (_refCounts.TryGetValue(blockId, out int count))
                {
                    if (count > 1)
                    {
                        _refCounts[blockId] = count - 1;
                        continue; // Block still referenced
                    }
                }

                _allocatedBlocks.Remove(blockId);
                _refCounts.Remove(blockId);
                _freeBlocks.Enqueue(blockId);
                _totalFrees++;
            }
        }
    }

    /// <summary>
    /// Increments the reference count for a block (for copy-on-write).
    /// </summary>
    /// <param name="blockId">The block ID to reference.</param>
    public void AddReference(int blockId)
    {
        lock (_lock)
        {
            if (_refCounts.TryGetValue(blockId, out int count))
            {
                _refCounts[blockId] = count + 1;
            }
        }
    }

    /// <summary>
    /// Gets the reference count for a block.
    /// </summary>
    public int GetReferenceCount(int blockId)
    {
        lock (_lock)
        {
            return _refCounts.TryGetValue(blockId, out int count) ? count : 0;
        }
    }

    /// <summary>
    /// Performs copy-on-write for a block if it has multiple references.
    /// </summary>
    /// <param name="blockId">The block ID to potentially copy.</param>
    /// <param name="copyData">Action to copy data from old block to new block.</param>
    /// <returns>The block ID to use (original if ref count == 1, new copy otherwise).</returns>
    public int CopyOnWrite(int blockId, Action<int, int>? copyData = null)
    {
        lock (_lock)
        {
            if (!_refCounts.TryGetValue(blockId, out int count) || count <= 1)
                return blockId; // No copy needed

            // Need to copy - allocate new block
            if (_freeBlocks.Count == 0)
                return -1; // No space for copy

            int newBlockId = _freeBlocks.Dequeue();
            _allocatedBlocks.Add(newBlockId);
            _refCounts[newBlockId] = 1;
            _totalAllocations++;

            // Decrement reference on original
            _refCounts[blockId] = count - 1;

            _copyOnWriteCount++;

            // Copy data if callback provided
            copyData?.Invoke(blockId, newBlockId);

            return newBlockId;
        }
    }

    /// <summary>
    /// Checks if the manager can allocate the specified number of blocks.
    /// </summary>
    public bool CanAllocate(int count)
    {
        lock (_lock)
        {
            return _freeBlocks.Count >= count;
        }
    }

    /// <summary>
    /// Calculates how many tokens can fit in the specified number of blocks.
    /// </summary>
    public int TokensForBlocks(int numBlocks) => numBlocks * _config.BlockSize;

    /// <summary>
    /// Calculates how many blocks are needed for the specified number of tokens.
    /// </summary>
    public int BlocksForTokens(int numTokens)
    {
        return (numTokens + _config.BlockSize - 1) / _config.BlockSize;
    }

    /// <summary>
    /// Gets statistics about the block manager.
    /// </summary>
    public BlockManagerStats GetStats()
    {
        lock (_lock)
        {
            return new BlockManagerStats
            {
                TotalBlocks = _config.NumBlocks,
                AllocatedBlocks = _allocatedBlocks.Count,
                FreeBlocks = _freeBlocks.Count,
                MemoryUtilization = (double)_allocatedBlocks.Count / _config.NumBlocks,
                TotalAllocations = _totalAllocations,
                TotalFrees = _totalFrees,
                CopyOnWriteCount = _copyOnWriteCount,
                BlockSizeTokens = _config.BlockSize,
                BytesPerBlock = _config.BytesPerBlock,
                TotalMemoryBytes = (long)_config.NumBlocks * _config.BytesPerBlock
            };
        }
    }

    /// <summary>
    /// Resets the block manager, freeing all blocks.
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _allocatedBlocks.Clear();
            _refCounts.Clear();
            _freeBlocks.Clear();

            for (int i = 0; i < _config.NumBlocks; i++)
            {
                _freeBlocks.Enqueue(i);
            }

            _totalAllocations = 0;
            _totalFrees = 0;
            _copyOnWriteCount = 0;
        }
    }
}

/// <summary>
/// Configuration for the block manager.
/// </summary>
internal class BlockManagerConfig
{
    /// <summary>
    /// Number of tokens per block.
    /// </summary>
    public int BlockSize { get; set; } = 16;

    /// <summary>
    /// Total number of blocks to allocate.
    /// </summary>
    public int NumBlocks { get; set; } = 1024;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    public int NumLayers { get; set; } = 32;

    /// <summary>
    /// Number of attention heads per layer.
    /// </summary>
    public int NumHeads { get; set; } = 32;

    /// <summary>
    /// Dimension of each attention head.
    /// </summary>
    public int HeadDimension { get; set; } = 128;

    /// <summary>
    /// Whether to use GPU memory.
    /// </summary>
    public bool UseGpuMemory { get; set; } = false;

    /// <summary>
    /// GPU device ID (if using GPU memory).
    /// </summary>
    public int GpuDeviceId { get; set; } = 0;

    /// <summary>
    /// Bytes per block (calculated based on model configuration).
    /// </summary>
    public long BytesPerBlock => (long)BlockSize * NumLayers * NumHeads * HeadDimension * sizeof(float) * 2; // K and V

    /// <summary>
    /// Total memory required in bytes.
    /// </summary>
    public long TotalMemoryBytes => BytesPerBlock * NumBlocks;

    /// <summary>
    /// Creates a configuration from available memory size.
    /// </summary>
    public static BlockManagerConfig FromMemorySize(
        long availableBytes,
        int blockSize,
        int numLayers,
        int numHeads,
        int headDim)
    {
        var config = new BlockManagerConfig
        {
            BlockSize = blockSize,
            NumLayers = numLayers,
            NumHeads = numHeads,
            HeadDimension = headDim
        };

        // Calculate bytes per block
        long bytesPerBlock = (long)blockSize * numLayers * numHeads * headDim * sizeof(float) * 2;

        // Calculate number of blocks that fit
        config.NumBlocks = (int)(availableBytes / bytesPerBlock);

        return config;
    }

    /// <summary>
    /// Creates a configuration for a specific model.
    /// </summary>
    public static BlockManagerConfig ForModel(string modelName, long availableMemoryBytes, int blockSize = 16)
    {
        return modelName.ToLowerInvariant() switch
        {
            "llama-7b" => FromMemorySize(availableMemoryBytes, blockSize, 32, 32, 128),
            "llama-13b" => FromMemorySize(availableMemoryBytes, blockSize, 40, 40, 128),
            "llama-70b" => FromMemorySize(availableMemoryBytes, blockSize, 80, 64, 128),
            "gpt-2" => FromMemorySize(availableMemoryBytes, blockSize, 12, 12, 64),
            "gpt-2-medium" => FromMemorySize(availableMemoryBytes, blockSize, 24, 16, 64),
            "gpt-2-large" => FromMemorySize(availableMemoryBytes, blockSize, 36, 20, 64),
            "gpt-2-xl" => FromMemorySize(availableMemoryBytes, blockSize, 48, 25, 64),
            _ => FromMemorySize(availableMemoryBytes, blockSize, 32, 32, 128)
        };
    }
}

/// <summary>
/// Statistics about the block manager state.
/// </summary>
internal class BlockManagerStats
{
    /// <summary>Total number of blocks in the pool.</summary>
    public int TotalBlocks { get; set; }

    /// <summary>Number of currently allocated blocks.</summary>
    public int AllocatedBlocks { get; set; }

    /// <summary>Number of free blocks.</summary>
    public int FreeBlocks { get; set; }

    /// <summary>Memory utilization (0-1).</summary>
    public double MemoryUtilization { get; set; }

    /// <summary>Total number of allocations performed.</summary>
    public long TotalAllocations { get; set; }

    /// <summary>Total number of frees performed.</summary>
    public long TotalFrees { get; set; }

    /// <summary>Number of copy-on-write operations.</summary>
    public long CopyOnWriteCount { get; set; }

    /// <summary>Number of tokens per block.</summary>
    public int BlockSizeTokens { get; set; }

    /// <summary>Bytes per block.</summary>
    public long BytesPerBlock { get; set; }

    /// <summary>Total memory in bytes.</summary>
    public long TotalMemoryBytes { get; set; }

    /// <summary>Used memory in bytes.</summary>
    public long UsedMemoryBytes => (long)AllocatedBlocks * BytesPerBlock;

    /// <summary>Free memory in bytes.</summary>
    public long FreeMemoryBytes => (long)FreeBlocks * BytesPerBlock;
}
