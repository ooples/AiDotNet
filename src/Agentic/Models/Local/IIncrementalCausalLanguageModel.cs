using AiDotNet.LinearAlgebra;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// An <see cref="ICausalLanguageModel{T}"/> that supports incremental (KV-cached) decoding: it processes the
/// prompt once, caches the per-position state, and then advances one token at a time without recomputing the
/// whole sequence. This is the fast path for autoregressive generation.
/// </summary>
/// <typeparam name="T">The tensor element type.</typeparam>
/// <remarks>
/// <para>
/// The base <see cref="ICausalLanguageModel{T}.NextTokenLogits"/> re-feeds the full context each step, which
/// is correct but O(n²) over a generation. A model that maintains a key/value cache implements this interface
/// so <see cref="LocalEngineChatClient{T}"/> can drive it incrementally: <see cref="StartSequence"/> primes
/// the cache with the prompt and returns the first next-token logits, then each <see cref="AppendToken"/>
/// feeds a single new token and returns the following logits. <see cref="ResetCache"/> clears state between
/// independent generations.
/// </para>
/// <para><b>For Beginners:</b> Without caching, predicting each new word re-reads the entire conversation —
/// slower and slower as it grows. With caching, the model remembers the work it already did and only looks at
/// the one new word each time. This interface is how a model advertises "I can do the fast, remember-as-you-go
/// version"; the engine uses it automatically when available and falls back to the simple way otherwise.
/// </para>
/// </remarks>
public interface IIncrementalCausalLanguageModel<T> : ICausalLanguageModel<T>
{
    /// <summary>
    /// Clears any cached decoding state, so the next <see cref="StartSequence"/> begins fresh.
    /// </summary>
    void ResetCache();

    /// <summary>
    /// Processes the prompt, populating the cache, and returns the logits for the first generated token.
    /// </summary>
    /// <param name="promptTokenIds">The prompt token ids. Must be non-empty.</param>
    /// <returns>The next-token logits following the prompt (length = <see cref="ICausalLanguageModel{T}.VocabularySize"/>).</returns>
    Vector<T> StartSequence(IReadOnlyList<int> promptTokenIds);

    /// <summary>
    /// Appends a single token to the cached context and returns the logits for the token after it.
    /// </summary>
    /// <param name="tokenId">The token id to append (typically the one just generated).</param>
    /// <returns>The next-token logits (length = <see cref="ICausalLanguageModel{T}.VocabularySize"/>).</returns>
    Vector<T> AppendToken(int tokenId);
}
