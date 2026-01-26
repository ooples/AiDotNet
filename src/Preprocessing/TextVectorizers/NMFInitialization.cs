namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Specifies the initialization method for NMF.
/// </summary>
public enum NMFInitialization
{
    /// <summary>
    /// Random initialization with small positive values.
    /// </summary>
    Random,

    /// <summary>
    /// Non-negative Double SVD initialization (more stable but slower).
    /// </summary>
    NNDSVD
}
