namespace AiDotNet.Inference;

/// <summary>
/// Per-call inference context carried through the forward pass so that paged/cached attention
/// layers can route to a specific KV-cache sequence at a specific position WITHOUT mutating shared
/// layer state. This is what makes one optimized model instance + one shared <c>PagedKVCache</c>
/// safe to drive for multiple concurrent generation sequences: the sequence id and position travel
/// with the call instead of living in mutable layer fields.
/// </summary>
/// <remarks>
/// <para>
/// A single network forward over <c>seqLen</c> tokens uses ONE context: every context-aware layer
/// reads <see cref="Position"/> as the start position for this forward and appends its KV at
/// <c>Position .. Position + seqLen</c>. Layers do NOT advance <see cref="Position"/> (they all
/// share the same start within a forward); the owning generation session advances it by the number
/// of tokens after the whole forward completes.
/// </para>
/// </remarks>
internal sealed class InferenceForwardContext
{
    /// <summary>
    /// The KV-cache sequence id this forward belongs to (isolates concurrent sequences).
    /// </summary>
    public long SequenceId { get; }

    /// <summary>
    /// Start position (number of tokens already cached for this sequence) for this forward.
    /// Advanced by the owning session between forwards, never by individual layers.
    /// </summary>
    public int Position { get; set; }

    /// <summary>
    /// When non-null, this forward is BATCHED: batch row <c>b</c> of a <c>[batch, seqLen, dim]</c> input
    /// belongs to sequence <c>SequenceIds[b]</c> at start position <c>Positions[b]</c>. This drives one
    /// batched decode step across many independent sequences over the shared paged KV cache (the
    /// continuous-batching throughput win). Null => single-sequence mode using
    /// <see cref="SequenceId"/>/<see cref="Position"/> (and batch size 1).
    /// </summary>
    public long[]? SequenceIds { get; }

    /// <summary>Per-row start positions, parallel to <see cref="SequenceIds"/>. Null in single-sequence mode.</summary>
    public int[]? Positions { get; }

    /// <summary>True when this context addresses one sequence per batch row (batched decode).</summary>
    public bool IsBatched => SequenceIds is not null;

    /// <summary>
    /// Creates a context for the given sequence at the given start position (single-sequence mode).
    /// </summary>
    public InferenceForwardContext(long sequenceId, int position = 0)
    {
        SequenceId = sequenceId;
        Position = position;
    }

    /// <summary>
    /// Creates a BATCHED context: batch row <c>b</c> maps to sequence <paramref name="sequenceIds"/>[b] at
    /// <paramref name="positions"/>[b]. The <see cref="SequenceId"/>/<see cref="Position"/> scalars mirror
    /// row 0 for back-compatibility with single-sequence code paths.
    /// </summary>
    public InferenceForwardContext(long[] sequenceIds, int[] positions)
    {
        Guard.NotNull(sequenceIds);
        Guard.NotNull(positions);
        if (sequenceIds.Length == 0 || sequenceIds.Length != positions.Length)
        {
            throw new ArgumentException(
                $"sequenceIds ({sequenceIds.Length}) and positions ({positions.Length}) must be non-empty and equal length.");
        }
        SequenceIds = sequenceIds;
        Positions = positions;
        SequenceId = sequenceIds[0];
        Position = positions[0];
    }
}

/// <summary>
/// Implemented by attention layers that can run a per-sequence, position-addressed forward driven by
/// an <see cref="InferenceForwardContext"/> (rather than mutable <c>SequenceId</c>/position fields),
/// enabling concurrent multi-sequence decode over a shared KV cache.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal interface IContextAwareInferenceLayer<T>
{
    /// <summary>
    /// Runs the layer's inference forward for the sequence and position in <paramref name="ctx"/>,
    /// appending to the corresponding KV-cache slot. Must not mutate shared layer state.
    /// </summary>
    Tensor<T> ForwardWithContext(Tensor<T> input, InferenceForwardContext ctx);
}
