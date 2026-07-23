namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Specifies the type of MDS algorithm.
/// </summary>
public enum MDSType
{
    /// <summary>
    /// Classical MDS using eigendecomposition.
    /// </summary>
    Classical,

    /// <summary>
    /// Non-metric MDS using SMACOF algorithm.
    /// </summary>
    NonMetric
}
