using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options for Online/Streaming K-Means.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Online K-Means processes data points one at a time or in small batches,
/// making it suitable for streaming data or datasets too large to fit in memory.
/// </para>
/// <para><b>For Beginners:</b> Regular clustering loads ALL data at once.
/// That's a problem when:
/// - Data is too big to fit in memory
/// - Data arrives continuously (streaming)
/// - You need real-time updates
///
/// Online K-Means solves this by:
/// - Processing one point (or small batch) at a time
/// - Incrementally updating cluster centers
/// - Never needing to see all data at once
///
/// The learning rate controls how much new data affects existing clusters.
/// </para>
/// </remarks>
public class OnlineKMeansOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Initializes OnlineKMeansOptions with appropriate defaults.
    /// </summary>
    public OnlineKMeansOptions()
    {
        MaxIterations = 1;
    }

    /// <summary>
    /// Gets or sets the number of clusters.
    /// </summary>
    /// <value>Number of clusters. Default is 3.</value>
    public int NumClusters { get; set; } = 3;

    /// <summary>
    /// Gets or sets the learning rate for center updates.
    /// </summary>
    /// <value>Learning rate (0-1). Default is 0.01.</value>
    /// <remarks>
    /// <para>
    /// Controls how much each new point affects cluster centers:
    /// - Lower: Stable but slow to adapt
    /// - Higher: Adapts quickly but may be unstable
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether to decay the learning rate over time.
    /// </summary>
    /// <value>True to decay learning rate. Default is true.</value>
    public bool DecayLearningRate { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum learning rate when decaying.
    /// </summary>
    /// <value>Minimum learning rate. Default is 0.0001.</value>
    public double MinLearningRate { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the distance metric.
    /// </summary>
    /// <value>Distance metric, or null for Euclidean.</value>
    public new IDistanceMetric<T>? DistanceMetric { get; set; }
}
