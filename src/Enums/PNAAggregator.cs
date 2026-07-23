namespace AiDotNet.Enums;

/// <summary>
/// Aggregation function types for Principal Neighbourhood Aggregation (PNA).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These are different ways to combine information from neighbor nodes:
///
/// - **Mean**: Average all neighbor features (balanced, smooth)
/// - **Max**: Take the maximum value (emphasizes strong signals)
/// - **Min**: Take the minimum value (emphasizes weak signals)
/// - **Sum**: Add up all features (sensitive to number of neighbors)
/// - **StdDev**: Standard deviation (captures variance in neighborhood)
/// </para>
/// </remarks>
public enum PNAAggregator
{
    /// <summary>Mean aggregation - averages neighbor features.</summary>
    Mean,

    /// <summary>Max aggregation - takes maximum of neighbor features.</summary>
    Max,

    /// <summary>Min aggregation - takes minimum of neighbor features.</summary>
    Min,

    /// <summary>Sum aggregation - sums neighbor features.</summary>
    Sum,

    /// <summary>Standard deviation aggregation - computes std of neighbor features.</summary>
    StdDev
}
