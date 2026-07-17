using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// Describes the physical location of a sequence's key/value cache: the ordered list of KV-cache blocks it
/// occupies (its <b>block table</b>) plus how many token slots of the final block are actually filled. The
/// engine hands one of these per sequence to the fast-path runner so attention can gather KV from the paged
/// pool instead of a contiguous per-sequence buffer.
/// </summary>
public readonly struct SequenceKvLayout
{
    /// <summary>Creates a KV layout descriptor for one sequence.</summary>
    public SequenceKvLayout(string sequenceId, IReadOnlyList<int> blockTable, int filledTokens)
    {
        SequenceId = sequenceId;
        BlockTable = blockTable;
        FilledTokens = filledTokens;
    }

    /// <summary>The sequence whose KV this describes.</summary>
    public string SequenceId { get; }

    /// <summary>Physical KV-cache block ids, in logical order, backing this sequence.</summary>
    public IReadOnlyList<int> BlockTable { get; }

    /// <summary>Total number of token slots already populated across the block table.</summary>
    public int FilledTokens { get; }
}

/// <summary>
/// Optional <b>fast-path</b> capability a model advertises to the serving engine: paged-KV, batched,
/// incremental decoding. This is layered on top of <see cref="AiDotNet.Interfaces.IFullModel{T, TInput, TOutput}"/>
/// (which remains the universal execution path); a model that implements it lets the engine avoid recomputing
/// attention over the whole prefix every step, which is what makes throughput competitive with / better than
/// vLLM and TGI.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> a plain model only knows how to "read the whole prompt and answer" every time
/// (<c>Predict</c>). That is correct but slow for serving, because generating token #500 would re-read tokens
/// 1–499. A model that implements this interface instead keeps a <i>KV cache</i> — its memory of the tokens it
/// already read — in a shared block pool the engine manages, so each new token only does O(1) new work. The
/// engine calls <see cref="Prefill"/> once to load the prompt, then <see cref="DecodeStep"/> repeatedly, one
/// token per sequence, over many sequences at once (batched).</para>
/// <para>The engine owns KV memory (the block pool and block tables); the runner only reads/writes KV at the
/// block locations the engine specifies via <see cref="SequenceKvLayout"/>, and honors the copy-on-write block
/// copies the engine requests before a write into shared (prefix-shared) blocks.</para>
/// </remarks>
/// <typeparam name="T">The model's numeric type.</typeparam>
public interface ICausalLmRunner<T>
{
    /// <summary>Vocabulary size (the width of the returned logits rows).</summary>
    int VocabularySize { get; }

    /// <summary>Number of transformer layers whose KV must be cached (block pool is sized per layer).</summary>
    int NumLayers { get; }

    /// <summary>Number of key/value heads (GQA/MQA aware) used to size a KV block.</summary>
    int NumKvHeads { get; }

    /// <summary>Per-head dimension used to size a KV block.</summary>
    int HeadDim { get; }

    /// <summary>Token capacity of a single KV-cache block (must match the engine's block manager block size).</summary>
    int BlockSize { get; }

    /// <summary>
    /// Runs the prompt (prefill) for a batch of sequences, writing their KV into the specified block layouts
    /// and returning the last-position logits for each sequence (shape: [batch, vocab]) used to sample the
    /// first generated token. Supports chunked prefill: <paramref name="tokenCounts"/> may be a prefix of each
    /// sequence's prompt.
    /// </summary>
    /// <param name="tokenIdsPerSequence">The prompt token ids to process this call, per sequence.</param>
    /// <param name="layouts">Where each sequence's KV blocks live.</param>
    /// <param name="tokenCounts">How many tokens of each sequence to process this call (chunked prefill).</param>
    Tensor<T> Prefill(
        IReadOnlyList<IReadOnlyList<int>> tokenIdsPerSequence,
        IReadOnlyList<SequenceKvLayout> layouts,
        IReadOnlyList<int> tokenCounts);

    /// <summary>
    /// Advances a batch of sequences by exactly one token each: reads cached KV via the block layouts, appends
    /// the new token's KV, and returns next-token logits (shape: [batch, vocab]).
    /// </summary>
    /// <param name="lastTokenIds">The most recently produced token id for each sequence.</param>
    /// <param name="layouts">Where each sequence's KV blocks live (last block is the write target).</param>
    Tensor<T> DecodeStep(
        IReadOnlyList<int> lastTokenIds,
        IReadOnlyList<SequenceKvLayout> layouts);

    /// <summary>
    /// Executes engine-requested copy-on-write block copies before a write into shared blocks. Each copy
    /// duplicates the KV contents of a source block into a freshly allocated destination block across all
    /// layers. Called by the engine when prefix-shared sequences diverge.
    /// </summary>
    void CopyBlocks(IReadOnlyList<BlockCopy> copies);
}
