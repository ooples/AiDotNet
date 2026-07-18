using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Interface for draft models used in speculative decoding. Implement this to plug a custom draft model into
/// AiDotNet's serving engine (e.g. a small distilled model, an EAGLE/Medusa head, or a domain-specific
/// heuristic) so the target model verifies your drafts instead of the built-in N-gram prompt-lookup draft.
/// </summary>
/// <remarks>
/// <para>
/// Draft models are small, fast models that generate candidate tokens
/// for verification by the larger target model. They should be lightweight
/// enough to generate multiple tokens with minimal latency.
/// </para>
/// <para><b>For Beginners:</b> Speculative decoding speeds up generation by having a small, fast "draft"
/// model guess the next few tokens, which the big model then checks all at once. This interface is that draft
/// model's contract: given the tokens so far, return a handful of guessed next tokens (and their
/// probabilities). AiDotNet ships an N-gram draft by default; implement this interface if you want to supply
/// your own faster/smarter guesser.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations.</typeparam>
public interface IDraftModel<T>
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
