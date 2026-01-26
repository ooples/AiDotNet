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
    /// NNDSVD-inspired initialization using scaled random values based on input statistics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a simplified approximation that uses the mean of the input matrix
    /// to scale the random initialization, providing better starting values than
    /// pure random initialization. It does not implement the full NNDSVD algorithm
    /// (which requires SVD decomposition), but often achieves similar benefits for
    /// convergence stability.
    /// </para>
    /// </remarks>
    NNDSVD
}
