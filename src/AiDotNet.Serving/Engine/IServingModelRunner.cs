using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// One sequence's execution descriptor for a single engine step: the tokens the model must process, how far its
/// KV cache has already advanced, and where its KV blocks live. The engine builds one of these per scheduled
/// sequence and hands the batch to an <see cref="IServingModelRunner{T}"/>.
/// </summary>
/// <typeparam name="T">The model's numeric type.</typeparam>
public readonly struct SequenceExecution<T>
{
    /// <summary>Creates an execution descriptor.</summary>
    public SequenceExecution(
        string sequenceId,
        IReadOnlyList<int> allTokenIds,
        int numComputedTokens,
        IReadOnlyList<int> blockTable,
        IReadOnlyList<BlockCopy> blockCopies,
        bool isPrefill)
    {
        SequenceId = sequenceId;
        AllTokenIds = allTokenIds;
        NumComputedTokens = numComputedTokens;
        BlockTable = blockTable;
        BlockCopies = blockCopies;
        IsPrefill = isPrefill;
    }

    /// <summary>The sequence being executed.</summary>
    public string SequenceId { get; }

    /// <summary>Full token ids (prompt + generated) currently in the sequence.</summary>
    public IReadOnlyList<int> AllTokenIds { get; }

    /// <summary>
    /// How many leading tokens already have cached KV. A paged runner processes only tokens
    /// [<see cref="NumComputedTokens"/>, count); a recompute runner ignores this and reprocesses from the start.
    /// </summary>
    public int NumComputedTokens { get; }

    /// <summary>Physical KV-cache block ids backing this sequence (in logical order).</summary>
    public IReadOnlyList<int> BlockTable { get; }

    /// <summary>Copy-on-write block copies the runner must perform before writing this sequence's new KV.</summary>
    public IReadOnlyList<BlockCopy> BlockCopies { get; }

    /// <summary>True if this is a prefill step (processing the prompt), false for a single-token decode step.</summary>
    public bool IsPrefill { get; }
}

/// <summary>
/// The engine's internal model driver: given the batch of sequences scheduled this step, produce each one's
/// next-token logits. This is the single seam the <see cref="ContinuousBatchingEngine{T}"/> loop calls, so the
/// scheduler is model-agnostic and testable. Two adapters satisfy it: a paged fast-path adapter over a model's
/// <see cref="ICausalLmRunner{T}"/> capability, and a correctness fallback that recomputes via
/// <see cref="AiDotNet.Interfaces.IFullModel{T, TInput, TOutput}"/>.
/// </summary>
/// <typeparam name="T">The model's numeric type.</typeparam>
public interface IServingModelRunner<T>
{
    /// <summary>Vocabulary size (length of each returned logits vector).</summary>
    int VocabularySize { get; }

    /// <summary>
    /// Executes the scheduled batch and returns per-sequence next-token logits, in the same order as
    /// <paramref name="batch"/>. Each returned vector has length <see cref="VocabularySize"/>.
    /// </summary>
    IReadOnlyList<Vector<T>> Execute(IReadOnlyList<SequenceExecution<T>> batch);
}
