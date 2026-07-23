namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Specifies the affinity type for Laplacian Eigenmaps.
/// </summary>
public enum LaplacianAffinityType
{
    /// <summary>
    /// Binary connectivity based on k-nearest neighbors.
    /// </summary>
    NearestNeighbors,

    /// <summary>
    /// RBF (Gaussian) kernel weights on k-nearest neighbor graph.
    /// </summary>
    RBFNeighbors,

    /// <summary>
    /// Full RBF (Gaussian) kernel on all pairs.
    /// </summary>
    RBF
}
