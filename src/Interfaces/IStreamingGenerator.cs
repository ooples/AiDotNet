using System.Collections.Generic;
using System.Threading;

namespace AiDotNet.Interfaces;

/// <summary>
/// A generator that can stream its output incrementally (token/chunk deltas) as it is produced,
/// in addition to the buffered <see cref="IGenerator{T}.Generate(string)"/>.
/// </summary>
/// <remarks>
/// <para>
/// Streaming lets a RAG answer render as it is generated rather than after the whole response is
/// complete, which is the standard low-latency UX for chat/RAG systems.
/// </para>
/// <para><b>For Beginners:</b> instead of waiting for the whole answer, you get it piece by piece —
/// like watching text appear as an assistant "types" — by iterating the returned async stream.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
public interface IStreamingGenerator<T> : IGenerator<T>
{
    /// <summary>
    /// Generates a response to <paramref name="prompt"/>, yielding incremental text deltas as they arrive.
    /// </summary>
    /// <param name="prompt">The input prompt (typically already augmented with retrieved context).</param>
    /// <param name="cancellationToken">Cancels the in-flight generation.</param>
    /// <returns>An async stream of text fragments whose concatenation is the full response.</returns>
    IAsyncEnumerable<string> GenerateStreamAsync(string prompt, CancellationToken cancellationToken = default);
}
