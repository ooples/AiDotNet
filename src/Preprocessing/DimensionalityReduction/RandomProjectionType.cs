namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Specifies the type of random projection.
/// </summary>
public enum RandomProjectionType
{
    /// <summary>
    /// Gaussian random projection using N(0, 1/k) entries.
    /// </summary>
    Gaussian,

    /// <summary>
    /// Sparse random projection using {-1, 0, +1} entries.
    /// </summary>
    Sparse
}
