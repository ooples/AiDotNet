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
    /// Creates a context for the given sequence at the given start position.
    /// </summary>
    public InferenceForwardContext(long sequenceId, int position = 0)
    {
        SequenceId = sequenceId;
        Position = position;
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
