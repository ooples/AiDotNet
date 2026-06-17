using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Models;

/// <summary>
/// Optional capability for servable models that support autoregressive text generation.
/// </summary>
/// <remarks>
/// <para>
/// Token-level generation requires a model whose forward pass maps a sequence of token IDs
/// to per-position vocabulary logits (the shape transformer language models produce). Models
/// served as tensor-to-tensor (<see cref="IServableModel{T}"/> backed by a neural network)
/// expose this capability; vector/matrix prediction models do not.
/// </para>
/// <para><b>For Beginners:</b> Ordinary serving runs one prediction and returns a vector.
/// Text generation is different: the model is called repeatedly, each call producing the
/// logits for the next token, which is then appended and fed back in. This interface exposes
/// that single "tokens in, logits out" forward pass so the continuous-batching engine can
/// drive the generation loop.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used by the model.</typeparam>
internal interface IServableGenerativeModel<T>
{
    /// <summary>
    /// Gets whether this model supports token-level generation. When false,
    /// <see cref="Forward"/> must not be called.
    /// </summary>
    bool SupportsGeneration { get; }

    /// <summary>
    /// Runs a single token-level forward pass.
    /// </summary>
    /// <param name="inputTokenIds">
    /// Token IDs shaped <c>[batch = 1, sequenceLength]</c> (each element is a token ID cast to <typeparamref name="T"/>).
    /// </param>
    /// <returns>
    /// Logits shaped <c>[1, sequenceLength, vocabularySize]</c> or <c>[1, vocabularySize]</c>.
    /// </returns>
    Tensor<T> Forward(Tensor<T> inputTokenIds);

    /// <summary>
    /// Gets whether this model supports KV-cached incremental generation via per-request sessions
    /// (<see cref="BeginGeneration"/>). When false, callers fall back to stateless full-context
    /// <see cref="Forward"/> decoding.
    /// </summary>
    bool SupportsIncrementalGeneration { get; }

    /// <summary>
    /// Begins a new generation session backed by its own KV-cache sequence, isolated from other
    /// concurrent sessions sharing the same model. Only valid when
    /// <see cref="SupportsIncrementalGeneration"/> is true.
    /// </summary>
    /// <returns>A disposable session; dispose it to free the KV-cache sequence.</returns>
    IGenerationSession<T> BeginGeneration();

    /// <summary>
    /// Begins a generation session for a specific prompt, reusing cached KV for the longest
    /// registered prompt prefix (RadixAttention-style sharing via copy-on-write fork). The returned
    /// session's <see cref="IGenerationSession{T}.CachedPromptTokens"/> reports how many leading
    /// prompt tokens are already in the KV cache, so the caller forwards only the remaining suffix.
    /// </summary>
    /// <param name="promptTokens">The full prompt token IDs.</param>
    IGenerationSession<T> BeginGeneration(System.Collections.Generic.IReadOnlyList<int> promptTokens);

    /// <summary>
    /// Whether the model accepts a multi-token forward (<c>[1, n]</c>) so the prompt can be prefilled
    /// in a single pass (per-position logits) rather than one token at a time. Determined by a probe;
    /// false for fixed single-token-step models.
    /// </summary>
    bool SupportsBatchedPrefill { get; }
}

/// <summary>
/// A per-request KV-cached generation session over a shared model + paged KV cache. Each session
/// owns a distinct cache sequence id, so concurrent sessions decode independently. Forwarding only
/// the new tokens advances the session's position and KV cache (true incremental decode).
/// </summary>
/// <typeparam name="T">The numeric type used by the model.</typeparam>
public interface IGenerationSession<T> : IDisposable
{
    /// <summary>
    /// Runs an incremental forward over only the new tokens (the prompt on the first call, then one
    /// token per decode step), appending to this session's KV cache.
    /// </summary>
    /// <param name="newTokenIds">New token IDs shaped <c>[1, newLength]</c>.</param>
    /// <returns>Logits for the new tokens (<c>[1, newLength, vocab]</c> or a last-position form).</returns>
    Tensor<T> Forward(Tensor<T> newTokenIds);

    /// <summary>
    /// Number of leading prompt tokens already present in this session's KV cache from a prefix-share
    /// fork (0 for a fresh session). The caller forwards only <c>prompt[CachedPromptTokens..]</c>.
    /// </summary>
    int CachedPromptTokens { get; }

    /// <summary>
    /// Registers this session's current KV state (which must hold exactly the prefilled prompt) as a
    /// reusable prefix base, so later requests whose prompt extends this one can fork from it. Call
    /// once, immediately after prefill and before decoding. No-op when prefix sharing is unavailable.
    /// </summary>
    /// <param name="promptTokens">The prompt token IDs just prefilled into this session.</param>
    void RegisterPromptPrefix(System.Collections.Generic.IReadOnlyList<int> promptTokens);
}
