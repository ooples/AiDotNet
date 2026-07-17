namespace AiDotNet.Serving.Engine;

/// <summary>
/// A copy-on-write instruction the block manager emits when prefix-shared sequences diverge: the runner must
/// duplicate the key/value contents of <see cref="Source"/> into <see cref="Destination"/> (across all layers)
/// before the writing sequence appends into what was a shared block.
/// </summary>
public readonly struct BlockCopy
{
    /// <summary>Creates a copy-on-write instruction.</summary>
    public BlockCopy(int source, int destination)
    {
        Source = source;
        Destination = destination;
    }

    /// <summary>Physical block id to copy KV from.</summary>
    public int Source { get; }

    /// <summary>Physical block id to copy KV into (freshly allocated, refcount 1).</summary>
    public int Destination { get; }
}

/// <summary>
/// Paged KV-cache block manager: the memory allocator at the heart of high-throughput serving. It partitions
/// the KV cache into fixed-size <b>blocks</b> (each holding <see cref="BlockSize"/> token slots) and hands each
/// sequence a <b>block table</b> (its ordered list of physical blocks). Blocks are reference counted so several
/// sequences can share a common prompt prefix (copy-on-write), and are recycled to a free list when released.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> a language model's "memory" of the tokens it has read (the KV cache) is large,
/// and naive serving gives each request one big contiguous chunk — which wastes memory and limits how many
/// requests fit at once. Paging (from vLLM's PagedAttention) instead cuts the cache into small fixed blocks,
/// like pages of RAM, and gives each sequence a list of block numbers. Blocks can be shared between requests
/// that start with the same prompt, and freed blocks go back into a pool for reuse. This class is the
/// bookkeeper for that pool: who owns which blocks, when to copy a shared block before writing (copy-on-write),
/// and when a block is free again.</para>
/// <para>This type is deliberately decoupled from the model and from <see cref="Sequence"/>: it reasons purely
/// about sequence ids and token counts, so it is fully unit-testable on its own. Its invariants — no physical
/// block is ever handed out twice, and free blocks + referenced blocks always sum to the pool size — are the
/// correctness backbone of the engine.</para>
/// </remarks>
public sealed class BlockManager
{
    private readonly int _blockSize;
    private readonly int _totalBlocks;
    private readonly int[] _refCounts;
    private readonly Stack<int> _freeBlocks;
    private readonly Dictionary<string, List<int>> _blockTables;
    private readonly Dictionary<string, int> _lengths;

    /// <summary>Creates a block manager over a pool of <paramref name="totalBlocks"/> blocks of the given size.</summary>
    /// <param name="totalBlocks">Number of physical KV blocks in the pool (sizes total KV memory).</param>
    /// <param name="blockSize">Token slots per block (must match the runner's block size).</param>
    public BlockManager(int totalBlocks, int blockSize)
    {
        if (totalBlocks < 1) throw new ArgumentOutOfRangeException(nameof(totalBlocks), "Pool must have at least one block.");
        if (blockSize < 1) throw new ArgumentOutOfRangeException(nameof(blockSize), "Block size must be at least one token.");

        _blockSize = blockSize;
        _totalBlocks = totalBlocks;
        _refCounts = new int[totalBlocks];
        _freeBlocks = new Stack<int>(totalBlocks);
        // Push in descending order so ids are popped ascending (0,1,2,...) — nicer for tests/debugging.
        for (int i = totalBlocks - 1; i >= 0; i--) _freeBlocks.Push(i);
        _blockTables = new Dictionary<string, List<int>>();
        _lengths = new Dictionary<string, int>();
    }

    /// <summary>Token slots per block.</summary>
    public int BlockSize => _blockSize;

    /// <summary>Total blocks in the pool.</summary>
    public int TotalBlocks => _totalBlocks;

    /// <summary>Blocks currently free (available for allocation).</summary>
    public int NumFreeBlocks => _freeBlocks.Count;

    /// <summary>Blocks currently referenced by at least one sequence.</summary>
    public int NumUsedBlocks => _totalBlocks - _freeBlocks.Count;

    /// <summary>Fraction (0..1) of the pool currently in use.</summary>
    public double Usage => _totalBlocks == 0 ? 0.0 : (double)NumUsedBlocks / _totalBlocks;

    /// <summary>Number of sequences currently holding a block table.</summary>
    public int NumSequences => _blockTables.Count;

    /// <summary>True if the given sequence currently has an allocation.</summary>
    public bool Contains(string sequenceId) => _blockTables.ContainsKey(sequenceId);

    /// <summary>Number of blocks needed to hold <paramref name="numTokens"/> tokens.</summary>
    public int BlocksForTokens(int numTokens) => (numTokens + _blockSize - 1) / _blockSize;

    /// <summary>True if the pool has enough free blocks to allocate a fresh sequence of <paramref name="numTokens"/> tokens.</summary>
    public bool CanAllocate(int numTokens)
    {
        if (numTokens < 1) throw new ArgumentOutOfRangeException(nameof(numTokens));
        return BlocksForTokens(numTokens) <= _freeBlocks.Count;
    }

    /// <summary>
    /// Allocates blocks for a new sequence's initial tokens (its prompt / prefill). The sequence must not
    /// already be allocated. Throws if the pool cannot satisfy the request (callers must check
    /// <see cref="CanAllocate"/> first).
    /// </summary>
    /// <returns>The sequence's block table (physical block ids in logical order).</returns>
    public IReadOnlyList<int> Allocate(string sequenceId, int numTokens)
    {
        if (string.IsNullOrEmpty(sequenceId)) throw new ArgumentException("Sequence id required.", nameof(sequenceId));
        if (numTokens < 1) throw new ArgumentOutOfRangeException(nameof(numTokens));
        if (_blockTables.ContainsKey(sequenceId))
            throw new InvalidOperationException($"Sequence '{sequenceId}' is already allocated.");

        int needed = BlocksForTokens(numTokens);
        if (needed > _freeBlocks.Count)
            throw new InvalidOperationException(
                $"Out of KV blocks: need {needed}, have {_freeBlocks.Count}. Check CanAllocate first.");

        var table = new List<int>(needed);
        for (int i = 0; i < needed; i++)
        {
            int block = _freeBlocks.Pop();
            _refCounts[block] = 1;
            table.Add(block);
        }
        _blockTables[sequenceId] = table;
        _lengths[sequenceId] = numTokens;
        return table;
    }

    /// <summary>
    /// True if the sequence can grow by <paramref name="numNewTokens"/> tokens: enough free blocks exist for any
    /// new blocks required plus a copy-on-write duplicate if its last block is shared.
    /// </summary>
    public bool CanAppend(string sequenceId, int numNewTokens = 1)
    {
        if (numNewTokens < 1) throw new ArgumentOutOfRangeException(nameof(numNewTokens));
        var table = RequireTable(sequenceId);
        int length = _lengths[sequenceId];

        int newBlocksNeeded = BlocksForTokens(length + numNewTokens) - table.Count;
        int cowNeeded = LastBlockNeedsCow(table, length) ? 1 : 0;
        return newBlocksNeeded + cowNeeded <= _freeBlocks.Count;
    }

    /// <summary>
    /// Grows a sequence by <paramref name="numNewTokens"/> tokens (normally 1, per decode step), allocating new
    /// blocks as needed and performing copy-on-write if its last block is shared with another sequence. Callers
    /// must check <see cref="CanAppend"/> first. Returns any copy-on-write instructions the runner must execute
    /// before writing the new token's KV (empty when no copy was needed).
    /// </summary>
    public IReadOnlyList<BlockCopy> Append(string sequenceId, int numNewTokens = 1)
    {
        if (numNewTokens < 1) throw new ArgumentOutOfRangeException(nameof(numNewTokens));
        var table = RequireTable(sequenceId);
        int length = _lengths[sequenceId];

        int newBlocksNeeded = BlocksForTokens(length + numNewTokens) - table.Count;
        bool cow = LastBlockNeedsCow(table, length);
        if (newBlocksNeeded + (cow ? 1 : 0) > _freeBlocks.Count)
            throw new InvalidOperationException(
                $"Out of KV blocks appending to '{sequenceId}'. Check CanAppend first.");

        List<BlockCopy>? copies = null;

        // Copy-on-write: the last block still has room and is shared (prefix-shared via Fork). We must not
        // write another sequence's token into a block others read, so duplicate it and rewire this sequence.
        if (cow)
        {
            int lastIndex = table.Count - 1;
            int oldBlock = table[lastIndex];
            int newBlock = _freeBlocks.Pop();
            _refCounts[newBlock] = 1;
            _refCounts[oldBlock]--;
            table[lastIndex] = newBlock;
            copies = new List<BlockCopy>(1) { new BlockCopy(oldBlock, newBlock) };
        }

        for (int i = 0; i < newBlocksNeeded; i++)
        {
            int block = _freeBlocks.Pop();
            _refCounts[block] = 1;
            table.Add(block);
        }

        _lengths[sequenceId] = length + numNewTokens;
        return copies ?? (IReadOnlyList<BlockCopy>)Array.Empty<BlockCopy>();
    }

    /// <summary>
    /// Creates a child sequence that shares the parent's blocks (copy-on-write prefix sharing). Both sequences
    /// read the same prefix KV until one of them writes into the shared tail, at which point <see cref="Append"/>
    /// duplicates only the block being written. The child must not already exist.
    /// </summary>
    /// <returns>The child's block table (initially identical physical blocks to the parent).</returns>
    public IReadOnlyList<int> Fork(string parentSequenceId, string childSequenceId)
    {
        var parentTable = RequireTable(parentSequenceId);
        if (string.IsNullOrEmpty(childSequenceId)) throw new ArgumentException("Child id required.", nameof(childSequenceId));
        if (_blockTables.ContainsKey(childSequenceId))
            throw new InvalidOperationException($"Sequence '{childSequenceId}' already exists.");

        var childTable = new List<int>(parentTable);
        foreach (int block in childTable) _refCounts[block]++;
        _blockTables[childSequenceId] = childTable;
        _lengths[childSequenceId] = _lengths[parentSequenceId];
        return childTable;
    }

    /// <summary>
    /// Releases a sequence's blocks: each block's reference count is decremented, and blocks reaching zero
    /// references return to the free pool. Safe to call once per sequence; unknown ids are ignored.
    /// </summary>
    public void Free(string sequenceId)
    {
        if (!_blockTables.TryGetValue(sequenceId, out var table)) return;
        foreach (int block in table)
        {
            if (--_refCounts[block] == 0)
                _freeBlocks.Push(block);
            else if (_refCounts[block] < 0)
                throw new InvalidOperationException($"Reference count underflow on block {block}.");
        }
        _blockTables.Remove(sequenceId);
        _lengths.Remove(sequenceId);
    }

    /// <summary>Returns the sequence's current block table (throws if unknown).</summary>
    public IReadOnlyList<int> GetBlockTable(string sequenceId) => RequireTable(sequenceId);

    /// <summary>Number of token slots filled in the sequence's final block (0 &lt; value ≤ BlockSize), or 0 if unallocated.</summary>
    public int FilledSlotsInLastBlock(string sequenceId)
    {
        if (!_lengths.TryGetValue(sequenceId, out int length) || length == 0) return 0;
        int rem = length % _blockSize;
        return rem == 0 ? _blockSize : rem;
    }

    /// <summary>Current logical token count of the sequence (throws if unknown).</summary>
    public int GetLength(string sequenceId) => _lengths.TryGetValue(sequenceId, out int l)
        ? l
        : throw new KeyNotFoundException($"Unknown sequence '{sequenceId}'.");

    private List<int> RequireTable(string sequenceId)
        => _blockTables.TryGetValue(sequenceId, out var table)
            ? table
            : throw new KeyNotFoundException($"Unknown sequence '{sequenceId}'.");

    // The last block must be copied-on-write iff it still has a free slot to be written AND it is shared with
    // another sequence. A full last block is never written into (a new block is appended instead), and an
    // unshared block can be written in place.
    private bool LastBlockNeedsCow(List<int> table, int length)
    {
        if (table.Count == 0) return false;
        bool lastHasRoom = length % _blockSize != 0;
        return lastHasRoom && _refCounts[table[table.Count - 1]] > 1;
    }
}
