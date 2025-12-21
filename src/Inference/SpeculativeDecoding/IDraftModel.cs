using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Interface for draft models used in speculative decoding.
/// </summary>
/// <remarks>
/// <para>
/// Draft models are small, fast models that generate candidate tokens
/// for verification by the larger target model. They should be lightweight
/// enough to generate multiple tokens with minimal latency.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations.</typeparam>
internal interface IDraftModel<T>
{
    /// <summary>
    /// Gets the maximum number of tokens this draft model can generate in one call.
    /// </summary>
    int MaxDraftTokens { get; }

    /// <summary>
    /// Generates draft tokens autoregressively.
    /// </summary>
    /// <param name="inputTokens">The input token sequence.</param>
    /// <param name="numDraftTokens">Number of draft tokens to generate.</param>
    /// <param name="temperature">Sampling temperature.</param>
    /// <returns>Draft generation result with tokens and probabilities.</returns>
    DraftResult<T> GenerateDraft(
        Vector<int> inputTokens,
        int numDraftTokens,
        T temperature);

    /// <summary>
    /// Gets the vocabulary size of this model.
    /// </summary>
    int VocabSize { get; }

    /// <summary>
    /// Resets any internal state (e.g., KV cache).
    /// </summary>
    void Reset();
}
