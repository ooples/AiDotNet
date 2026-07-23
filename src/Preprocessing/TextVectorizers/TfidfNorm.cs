namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Specifies the normalization method for TF-IDF vectors.
/// </summary>
public enum TfidfNorm
{
    /// <summary>
    /// No normalization.
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
