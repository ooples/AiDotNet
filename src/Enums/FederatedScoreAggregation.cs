namespace AiDotNet.Enums;

/// <summary>
/// How a federated feature selector combines the per-client feature scores into one global score.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Each client scores the features on its own data. The aggregation decides how the
/// clients' opinions are merged: an average, an average weighted by how much data each client has, or a robust
/// choice (median, trimmed mean) that ignores clients whose scores look extreme.
/// </para>
/// </remarks>
public enum FederatedScoreAggregation
{
    /// <summary>Average weighted by each client's number of samples.</summary>
    WeightedAverage,

    /// <summary>The median of the client scores (the mean of the two middle values for an even client count).</summary>
    Median,

    /// <summary>The mean after dropping the max(1, clients / 5) lowest and highest client scores.</summary>
    TrimmedMean,

    /// <summary>The plain average of the client scores.</summary>
    Mean
}
