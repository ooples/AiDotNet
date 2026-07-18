using System;

namespace AiDotNet.Inference.PagedAttention;

/// <summary>
/// A <see cref="PagedKVCache{T}"/> facade over N per-rank paged caches for tensor-parallel serving. The
/// continuous-batching scheduler treats it as one cache; its allocate/extend/free/truncate operations fan out
/// to every rank's cache in lockstep, so each tensor-parallel rank has the same sequence slots for its own
/// head-group's KV. Per-rank attention reads/writes its own concrete cache directly (not through this facade).
/// </summary>
/// <remarks>
/// <para>
/// The ranks are sized identically (each holds the KV for <c>numHeads / worldSize</c> heads), so length and
/// block-table queries are answered from rank 0. Allocation is all-or-nothing: if any rank cannot allocate,
/// the slots already taken on the earlier ranks are freed and allocation fails, keeping the ranks consistent.
/// </para>
/// <para><b>For Beginners:</b> Tensor parallelism spreads a model across several GPUs, each holding part of the
/// attention. Each GPU still needs its own key/value memory for the tokens it processes. This class keeps all
/// those per-GPU memories in step so the serving engine can manage them as if there were just one.
/// </para>
/// </remarks>
internal sealed class CompositePagedKVCache<T> : PagedKVCache<T>
{
    private readonly PagedKVCache<T>[] _ranks;

    /// <summary>Creates a composite over the given per-rank caches (at least one).</summary>
    public CompositePagedKVCache(PagedKVCache<T>[] ranks)
        : base(MinimalConfig())
    {
        if (ranks is null) throw new ArgumentNullException(nameof(ranks));
        if (ranks.Length == 0) throw new ArgumentException("At least one rank cache is required.", nameof(ranks));
        foreach (var r in ranks)
            if (r is null) throw new ArgumentException("Rank caches must be non-null.", nameof(ranks));
        _ranks = ranks;
    }

    /// <summary>The per-rank caches this composite fans to (rank order).</summary>
    public PagedKVCache<T>[] Ranks => _ranks;

    // The base storage is unused (all access is delegated); keep it tiny.
    private static PagedKVCacheConfig MinimalConfig() => new()
    {
        BlockSize = 1,
        NumBlocks = 1,
        NumLayers = 1,
        NumHeads = 1,
        HeadDimension = 1
    };

    public override bool AllocateSequence(long sequenceId, int initialTokens)
    {
        for (int i = 0; i < _ranks.Length; i++)
        {
            if (!_ranks[i].AllocateSequence(sequenceId, initialTokens))
            {
                // Roll back the ranks that already allocated so the ranks stay consistent.
                for (int j = 0; j < i; j++) _ranks[j].FreeSequence(sequenceId);
                return false;
            }
        }
        return true;
    }

    public override bool ExtendSequence(long sequenceId, int additionalTokens)
    {
        bool ok = true;
        foreach (var r in _ranks) ok &= r.ExtendSequence(sequenceId, additionalTokens);
        return ok;
    }

    public override void FreeSequence(long sequenceId)
    {
        foreach (var r in _ranks) r.FreeSequence(sequenceId);
    }

    public override bool TruncateSequence(long sequenceId, int newLength)
    {
        bool ok = true;
        foreach (var r in _ranks) ok &= r.TruncateSequence(sequenceId, newLength);
        return ok;
    }

    public override bool ForkSequence(long sourceSequenceId, long newSequenceId)
    {
        for (int i = 0; i < _ranks.Length; i++)
        {
            if (!_ranks[i].ForkSequence(sourceSequenceId, newSequenceId))
            {
                for (int j = 0; j < i; j++) _ranks[j].FreeSequence(newSequenceId);
                return false;
            }
        }
        return true;
    }

    public override bool HasCapacityFor(long sequenceId, int additionalTokens)
    {
        foreach (var r in _ranks)
            if (!r.HasCapacityFor(sequenceId, additionalTokens)) return false;
        return true;
    }

    // Ranks are symmetric (identical sizing / lockstep ops), so length/block-table/count queries use rank 0.
    public override int GetSequenceLength(long sequenceId) => _ranks[0].GetSequenceLength(sequenceId);

    public override int[]? GetBlockTable(long sequenceId) => _ranks[0].GetBlockTable(sequenceId);

    public override int ActiveSequenceCount => _ranks[0].ActiveSequenceCount;

    public override PagedKVCacheStats GetStats() => _ranks[0].GetStats();

    public override void Dispose()
    {
        foreach (var r in _ranks) r.Dispose();
        base.Dispose();
    }
}
