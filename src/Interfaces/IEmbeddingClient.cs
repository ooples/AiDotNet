using AiDotNet.Agentic.Embeddings;

namespace AiDotNet.Interfaces;

/// <summary>Turns text into embedding vectors, reporting transport failure as a value rather than an exception.</summary>
/// <remarks>
/// <para>
/// This is the seam every embedding-backed feature talks to, so the core library gains no provider dependency: the
/// shipped implementations are an OpenAI-compatible HTTP client, a caching decorator, and a deterministic in-memory
/// client that contacts nothing at all. A batch call is the primitive rather than a single-text call because every
/// provider charges and rate-limits per request, and a novelty decision usually needs a candidate plus several
/// neighbours at once.
/// </para>
/// <para>
/// Implementations must be safe for concurrent use, must preserve input order in
/// <see cref="EmbeddingBatch.Vectors"/>, and must return <see cref="EmbeddingBatch.Failure"/> for a transport or
/// provider error instead of throwing, so a caller can decide whether an unavailable provider means "accept" or
/// "reject". Argument mistakes — a <c>null</c> list, a <c>null</c> entry, an empty batch — remain exceptions,
/// because those are programming errors rather than run-time conditions. An implementation must never write an API
/// key to disk, a log, or a returned failure reason.
/// </para>
/// <para><b>For Beginners:</b> An embedding model reads a piece of text and returns a list of numbers that captures
/// what the text is about; two texts about the same thing get similar lists. This interface is how the library asks
/// for those numbers without caring which provider answers. If you have no provider, use the deterministic client
/// and everything still runs, just without real semantic meaning.</para>
/// </remarks>
public interface IEmbeddingClient
{
    /// <summary>Gets the identifier of the embedding model this client requests.</summary>
    string ModelId { get; }

    /// <summary>Embeds one batch of texts, preserving input order.</summary>
    /// <param name="texts">The texts to embed; at least one, none of them <c>null</c>.</param>
    /// <param name="cancellationToken">A token that cancels the request.</param>
    /// <returns>
    /// A batch holding one vector per input, or <see cref="EmbeddingBatch.Failure"/> when the provider could not be
    /// reached or answered unusably.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="texts"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="texts"/> is empty or contains a <c>null</c> entry.</exception>
    ValueTask<EmbeddingBatch> EmbedAsync(
        IReadOnlyList<string> texts,
        CancellationToken cancellationToken = default);
}
