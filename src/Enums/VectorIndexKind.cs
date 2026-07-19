namespace AiDotNet.Enums;

/// <summary>
/// Selects the in-memory vector-search index used to back a RAG document store.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A vector index is the data structure that makes "find the most
/// similar vectors" fast. Different indexes trade accuracy for speed:
/// <list type="bullet">
/// <item><description><b>Flat</b> - exact brute-force search; most accurate, slowest on large sets.</description></item>
/// <item><description><b>HNSW</b> - graph-based approximate search; excellent speed/recall balance.</description></item>
/// <item><description><b>IVF</b> - inverted-file clustering; fast on large collections.</description></item>
/// <item><description><b>LSH</b> - locality-sensitive hashing; very fast, lower recall.</description></item>
/// </list>
/// </para>
/// </remarks>
public enum VectorIndexKind
{
    /// <summary>Exact brute-force flat index (most accurate).</summary>
    Flat = 0,

    /// <summary>Hierarchical Navigable Small World graph index (fast, high recall).</summary>
    HNSW = 1,

    /// <summary>Inverted File index with clustering (scales to large collections).</summary>
    IVF = 2,

    /// <summary>Locality-Sensitive Hashing index (very fast, approximate).</summary>
    LSH = 3,
}
