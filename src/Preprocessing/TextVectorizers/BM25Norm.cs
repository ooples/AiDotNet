namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Specifies the normalization method for BM25 vectors.
/// </summary>
public enum BM25Norm
{
    /// <summary>
    /// No normalization (recommended for BM25).
    /// </summary>
    None,

    /// <summary>
    /// L1 normalization (sum of absolute values = 1).
    /// </summary>
    L1,

    /// <summary>
    /// L2 normalization (Euclidean length = 1).
    /// </summary>
    L2
}
