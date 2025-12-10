namespace AiDotNet.Enums;

/// <summary>
/// Aggregation function type for GraphSAGE.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These are different ways to combine information from neighbors.
///
/// - **Mean**: Average all neighbor features (balanced, smooth)
/// - **MaxPool**: Take the maximum value from neighbors (emphasizes outliers)
/// - **Sum**: Add up all neighbor features (sensitive to number of neighbors)
/// </para>
/// </remarks>
public enum SAGEAggregatorType
{
    /// <summary>
    /// Mean aggregation: averages neighbor features.
    /// </summary>
    Mean,

    /// <summary>
    /// Max pooling aggregation: takes maximum of neighbor features.
    /// </summary>
    MaxPool,

    /// <summary>
    /// Sum aggregation: sums neighbor features.
    /// </summary>
    Sum
}
