namespace AiDotNet.Enums;

/// <summary>
/// How a multi-vector retriever combines the query similarities of one document's vectors into a single score.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A document can be stored as several vectors (for example a summary followed by
/// its sections). This setting decides how their individual match scores become one document score.
/// </para>
/// </remarks>
public enum MultiVectorAggregation
{
    /// <summary>The best-matching vector's score (recommended for precise queries).</summary>
    Max,

    /// <summary>The average score of the matched vectors (recommended for general queries).</summary>
    Mean,

    /// <summary>
    /// A weighted average where vector N (from the "docId_vector_N" id) has weight 1 / (N + 1),
    /// so earlier vectors such as summaries count more than later details.
    /// </summary>
    Weighted,
}
