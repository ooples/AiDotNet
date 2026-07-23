namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Specifies the distance metric for MDS.
/// </summary>
public enum MDSMetric
{
    /// <summary>
    /// Euclidean (L2) distance.
    /// </summary>
    Euclidean,

    /// <summary>
    /// Squared Euclidean distance.
    /// </summary>
    SquaredEuclidean,

    /// <summary>
    /// Manhattan (L1) distance.
    /// </summary>
    Manhattan
}
