namespace AiDotNet.Inference.PagedAttention;

/// <summary>
/// Maps logical block indices to physical block IDs for a sequence.
/// </summary>
/// <remarks>
/// <para>
/// The BlockTable provides the indirection layer between logical sequence positions
/// and physical memory blocks. Each sequence has its own block table that grows
/// as more tokens are generated.
/// </para>
/// <para><b>For Beginners:</b> Think of the block table like a book's table of contents.
///
/// The book (your sequence) has logical chapters (logical blocks) numbered 0, 1, 2...
/// But the actual pages (physical blocks) might be scattered throughout the library.
/// The table of contents tells you: "Chapter 0 is on shelf A, Chapter 1 is on shelf Z..."
///
/// This indirection allows:
/// - Efficient memory allocation (use any available block)
/// - Copy-on-write for beam search (share chapters between book copies)
/// - Swapping to disk (move a chapter to storage, update the table)
/// </para>
/// </remarks>
internal class BlockTable
{
    private readonly int _blockSize;
    private readonly List<int> _physicalBlockIds;
    private readonly long _sequenceId;

    /// <summary>
    /// Gets the sequence ID this table belongs to.
    /// </summary>
    public long SequenceId => _sequenceId;

    /// <summary>
    /// Gets the block size (tokens per block).
    /// </summary>
    public int BlockSize => _blockSize;

    /// <summary>
    /// Gets the number of logical blocks in this table.
    /// </summary>
    public int NumLogicalBlocks => _physicalBlockIds.Count;

    /// <summary>
    /// Gets the total token capacity.
    /// </summary>
    public int Capacity => _physicalBlockIds.Count * _blockSize;

    /// <summary>
    /// Gets the physical block IDs as a read-only list.
    /// </summary>
    public IReadOnlyList<int> PhysicalBlockIds => _physicalBlockIds;

    /// <summary>
    /// Creates a new block table for a sequence.
    /// </summary>
    /// <param name="sequenceId">The sequence ID.</param>
    /// <param name="blockSize">Number of tokens per block.</param>
    public BlockTable(long sequenceId, int blockSize)
    {
        _sequenceId = sequenceId;
        _blockSize = blockSize;
        _physicalBlockIds = new List<int>();
    }

    /// <summary>
    /// Creates a new block table with pre-allocated blocks.
    /// </summary>
    public BlockTable(long sequenceId, int blockSize, IEnumerable<int> physicalBlockIds)
        : this(sequenceId, blockSize)
    {
        _physicalBlockIds.AddRange(physicalBlockIds);
    }

    /// <summary>
    /// Gets the physical block ID for a logical block index.
    /// </summary>
    /// <param name="logicalIndex">The logical block index.</param>
    /// <returns>The physical block ID.</returns>
    public int GetPhysicalBlock(int logicalIndex)
    {
        if (logicalIndex < 0 || logicalIndex >= _physicalBlockIds.Count)
            throw new ArgumentOutOfRangeException(nameof(logicalIndex),
                $"Logical index {logicalIndex} out of range [0, {_physicalBlockIds.Count})");

        return _physicalBlockIds[logicalIndex];
    }

    /// <summary>
    /// Gets the physical block ID and offset for a token position.
    /// </summary>
    /// <param name="tokenPosition">The token position in the sequence.</param>
    /// <returns>Tuple of (physical block ID, offset within block).</returns>
    public (int blockId, int offset) GetBlockAndOffset(int tokenPosition)
    {
        int logicalBlock = tokenPosition / _blockSize;
        int offset = tokenPosition % _blockSize;

        if (logicalBlock >= _physicalBlockIds.Count)
            throw new ArgumentOutOfRangeException(nameof(tokenPosition),
                $"Token position {tokenPosition} exceeds capacity {Capacity}");

        return (_physicalBlockIds[logicalBlock], offset);
    }

    /// <summary>
    /// Appends a new physical block to the table.
    /// </summary>
    /// <param name="physicalBlockId">The physical block ID to append.</param>
    public void AppendBlock(int physicalBlockId)
    {
        _physicalBlockIds.Add(physicalBlockId);
    }

    /// <summary>
    /// Appends multiple physical blocks to the table.
    /// </summary>
    public void AppendBlocks(IEnumerable<int> physicalBlockIds)
    {
        _physicalBlockIds.AddRange(physicalBlockIds);
    }

    /// <summary>
    /// Replaces a physical block ID at the specified logical index.
    /// </summary>
    /// <param name="logicalIndex">The logical block index.</param>
    /// <param name="newPhysicalBlockId">The new physical block ID.</param>
    /// <returns>The old physical block ID.</returns>
    public int ReplaceBlock(int logicalIndex, int newPhysicalBlockId)
    {
        if (logicalIndex < 0 || logicalIndex >= _physicalBlockIds.Count)
            throw new ArgumentOutOfRangeException(nameof(logicalIndex));

        int oldId = _physicalBlockIds[logicalIndex];
        _physicalBlockIds[logicalIndex] = newPhysicalBlockId;
        return oldId;
    }

    /// <summary>
    /// Removes the last block from the table.
    /// </summary>
    /// <returns>The removed physical block ID, or -1 if table is empty.</returns>
    public int RemoveLastBlock()
    {
        if (_physicalBlockIds.Count == 0)
            return -1;

        int lastBlock = _physicalBlockIds[^1];
        _physicalBlockIds.RemoveAt(_physicalBlockIds.Count - 1);
        return lastBlock;
    }

    /// <summary>
    /// Creates a copy of this block table (shallow copy - shares block IDs).
    /// </summary>
    /// <param name="newSequenceId">The new sequence ID for the copy.</param>
    /// <returns>A new block table with the same physical blocks.</returns>
    public BlockTable Copy(long newSequenceId)
    {
        return new BlockTable(newSequenceId, _blockSize, _physicalBlockIds);
    }

    /// <summary>
    /// Truncates the table to the specified number of logical blocks.
    /// </summary>
    /// <param name="numBlocks">The number of blocks to keep.</param>
    /// <returns>List of removed physical block IDs.</returns>
    public List<int> TruncateTo(int numBlocks)
    {
        var removed = new List<int>();

        while (_physicalBlockIds.Count > numBlocks)
        {
            removed.Add(_physicalBlockIds[^1]);
            _physicalBlockIds.RemoveAt(_physicalBlockIds.Count - 1);
        }

        return removed;
    }

    /// <summary>
    /// Clears all blocks from the table.
    /// </summary>
    /// <returns>List of all physical block IDs that were in the table.</returns>
    public List<int> Clear()
    {
        var blocks = new List<int>(_physicalBlockIds);
        _physicalBlockIds.Clear();
        return blocks;
    }

    /// <summary>
    /// Checks if the table has capacity for additional tokens.
    /// </summary>
    /// <param name="currentLength">Current token count.</param>
    /// <returns>True if more tokens can be added without new blocks.</returns>
    public bool HasCapacityFor(int currentLength)
    {
        return currentLength < Capacity;
    }

    /// <summary>
    /// Calculates how many blocks are needed for a given number of tokens.
    /// </summary>
    public int BlocksNeededFor(int numTokens)
    {
        return (numTokens + _blockSize - 1) / _blockSize;
    }

    /// <summary>
    /// Calculates how many additional blocks are needed for more tokens.
    /// </summary>
    public int AdditionalBlocksNeeded(int currentTokens, int additionalTokens)
    {
        int currentBlocks = BlocksNeededFor(currentTokens);
        int totalBlocks = BlocksNeededFor(currentTokens + additionalTokens);
        return Math.Max(0, totalBlocks - _physicalBlockIds.Count);
    }

    /// <summary>
    /// Gets the physical block IDs as an array (useful for GPU transfer).
    /// </summary>
    public int[] ToArray() => _physicalBlockIds.ToArray();

    /// <summary>
    /// Returns a string representation of the block table.
    /// </summary>
    public override string ToString()
    {
        return $"BlockTable[Seq={_sequenceId}, Blocks={NumLogicalBlocks}, Capacity={Capacity} tokens]";
    }
}

/// <summary>
/// Manages block tables for multiple sequences.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal class BlockTableManager<T>
{
    private readonly BlockManager<T> _blockManager;
    private readonly Dictionary<long, BlockTable> _blockTables;
    private readonly object _lock = new();

    /// <summary>
    /// Gets the underlying block manager.
    /// </summary>
    public BlockManager<T> BlockManager => _blockManager;

    /// <summary>
    /// Gets the number of active block tables.
    /// </summary>
    public int ActiveTableCount
    {
        get { lock (_lock) return _blockTables.Count; }
    }

    /// <summary>
    /// Creates a new block table manager.
    /// </summary>
    public BlockTableManager(BlockManager<T> blockManager)
    {
        _blockManager = blockManager ?? throw new ArgumentNullException(nameof(blockManager));
        _blockTables = new Dictionary<long, BlockTable>();
    }

    /// <summary>
    /// Creates a new block table for a sequence.
    /// </summary>
    /// <param name="sequenceId">The sequence ID.</param>
    /// <param name="initialBlocks">Number of initial blocks to allocate.</param>
    /// <returns>The created block table, or null if allocation failed.</returns>
    public BlockTable? CreateBlockTable(long sequenceId, int initialBlocks = 0)
    {
        lock (_lock)
        {
            if (_blockTables.ContainsKey(sequenceId))
                throw new InvalidOperationException($"Block table already exists for sequence {sequenceId}");

            var table = new BlockTable(sequenceId, _blockManager.Config.BlockSize);

            // Allocate initial blocks if requested
            if (initialBlocks > 0)
            {
                var blocks = _blockManager.AllocateBlocks(initialBlocks);
                if (blocks == null)
                    return null;

                table.AppendBlocks(blocks);
            }

            _blockTables[sequenceId] = table;
            return table;
        }
    }

    /// <summary>
    /// Gets the block table for a sequence.
    /// </summary>
    public BlockTable? GetBlockTable(long sequenceId)
    {
        lock (_lock)
        {
            return _blockTables.TryGetValue(sequenceId, out var table) ? table : null;
        }
    }

    /// <summary>
    /// Ensures a sequence has enough blocks for the specified token count.
    /// </summary>
    /// <param name="sequenceId">The sequence ID.</param>
    /// <param name="numTokens">Number of tokens needed.</param>
    /// <returns>True if successful, false if allocation failed.</returns>
    public bool EnsureCapacity(long sequenceId, int numTokens)
    {
        lock (_lock)
        {
            if (!_blockTables.TryGetValue(sequenceId, out var table))
                return false;

            int blocksNeeded = table.BlocksNeededFor(numTokens);
            int additionalBlocks = blocksNeeded - table.NumLogicalBlocks;

            if (additionalBlocks <= 0)
                return true;

            var newBlocks = _blockManager.AllocateBlocks(additionalBlocks);
            if (newBlocks == null)
                return false;

            table.AppendBlocks(newBlocks);
            return true;
        }
    }

    /// <summary>
    /// Frees a block table and returns its blocks to the pool.
    /// </summary>
    public void FreeBlockTable(long sequenceId)
    {
        lock (_lock)
        {
            if (!_blockTables.TryGetValue(sequenceId, out var table))
                return;

            _blockManager.FreeBlocks(table.PhysicalBlockIds);
            _blockTables.Remove(sequenceId);
        }
    }

    /// <summary>
    /// Forks a block table for beam search (creates copy with shared blocks).
    /// </summary>
    /// <param name="sourceSequenceId">The source sequence ID.</param>
    /// <param name="newSequenceId">The new sequence ID.</param>
    /// <returns>The forked block table, or null if source doesn't exist.</returns>
    public BlockTable? ForkBlockTable(long sourceSequenceId, long newSequenceId)
    {
        lock (_lock)
        {
            if (!_blockTables.TryGetValue(sourceSequenceId, out var sourceTable))
                return null;

            // Increment reference counts for all shared blocks
            foreach (int blockId in sourceTable.PhysicalBlockIds)
            {
                _blockManager.AddReference(blockId);
            }

            // Create copy with shared blocks
            var forkedTable = sourceTable.Copy(newSequenceId);
            _blockTables[newSequenceId] = forkedTable;

            return forkedTable;
        }
    }

    /// <summary>
    /// Performs copy-on-write for a block in a sequence's table.
    /// </summary>
    /// <param name="sequenceId">The sequence ID.</param>
    /// <param name="logicalBlockIndex">The logical block index to copy.</param>
    /// <param name="copyData">Action to copy data from old to new block.</param>
    /// <returns>True if successful.</returns>
    public bool CopyOnWrite(long sequenceId, int logicalBlockIndex, Action<int, int>? copyData = null)
    {
        lock (_lock)
        {
            if (!_blockTables.TryGetValue(sequenceId, out var table))
                return false;

            int oldBlockId = table.GetPhysicalBlock(logicalBlockIndex);
            int newBlockId = _blockManager.CopyOnWrite(oldBlockId, copyData);

            if (newBlockId < 0)
                return false;

            if (newBlockId != oldBlockId)
            {
                table.ReplaceBlock(logicalBlockIndex, newBlockId);
            }

            return true;
        }
    }

    /// <summary>
    /// Gets the physical block IDs for a sequence (for GPU transfer).
    /// </summary>
    public int[]? GetBlockTableArray(long sequenceId)
    {
        lock (_lock)
        {
            return _blockTables.TryGetValue(sequenceId, out var table) ? table.ToArray() : null;
        }
    }

    /// <summary>
    /// Gets all active sequence IDs.
    /// </summary>
    public long[] GetActiveSequenceIds()
    {
        lock (_lock)
        {
            return _blockTables.Keys.ToArray();
        }
    }

    /// <summary>
    /// Clears all block tables and frees all blocks.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            foreach (var table in _blockTables.Values)
            {
                _blockManager.FreeBlocks(table.PhysicalBlockIds);
            }
            _blockTables.Clear();
        }
    }
}
