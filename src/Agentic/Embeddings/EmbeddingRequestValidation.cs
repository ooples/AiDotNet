using AiDotNet.Validation;

namespace AiDotNet.Agentic.Embeddings;

/// <summary>The single argument check every <c>IEmbeddingClient</c> implementation applies to a batch.</summary>
/// <remarks>
/// Centralized so that the HTTP client, the caching decorator, and the deterministic client cannot drift apart on
/// what counts as a caller mistake, which matters because the contract distinguishes an argument error (throw) from
/// a provider failure (a failed batch).
/// </remarks>
internal static class EmbeddingRequestValidation
{
    internal static void Validate(IReadOnlyList<string> texts)
    {
        Guard.NotNull(texts);
        if (texts.Count == 0)
        {
            throw new ArgumentException("At least one text is required.", nameof(texts));
        }

        for (int index = 0; index < texts.Count; index++)
        {
            if (texts[index] is null)
            {
                throw new ArgumentException("A text to embed cannot be null.", nameof(texts));
            }
        }
    }
}
