namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options specific to KMeans clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// KMeans is a centroid-based clustering algorithm that partitions n observations
/// into k clusters where each observation belongs to the cluster with the nearest centroid.
/// </para>
/// <para><b>For Beginners:</b> KMeans works by:
/// 1. Choosing k initial center points
/// 2. Assigning each data point to its nearest center
/// 3. Recalculating centers as the mean of assigned points
/// 4. Repeating steps 2-3 until centers stabilize
///
/// The key settings are:
/// - Number of clusters (k): How many groups to find
/// - Initialization method: How to choose starting centers
/// - Number of runs: How many times to try different starting points
/// </para>
/// </remarks>
public class KMeansOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Gets or sets the number of clusters to find.
    /// </summary>
    /// <value>The number of clusters. Must be positive.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the "k" in KMeans - how many groups
    /// you want the algorithm to find. If you're not sure, try different values
    /// and compare results using the elbow method or silhouette score.
    /// </para>
    /// </remarks>
    public int NumClusters { get; set; } = 8;

    /// <summary>
    /// Gets or sets the initialization method for cluster centers.
    /// </summary>
    /// <value>The initialization method. Default is KMeansPlusPlus.</value>
    public KMeansInitMethod InitMethod { get; set; } = KMeansInitMethod.KMeansPlusPlus;

    /// <summary>
    /// Gets or sets custom initial cluster centers.
    /// </summary>
    /// <value>Matrix where each row is a cluster center, or null to use InitMethod.</value>
    /// <remarks>
    /// <para>
    /// When set, this overrides the InitMethod setting. The matrix must have
    /// NumClusters rows and the same number of columns as the input data.
    /// </para>
    /// </remarks>
    public Matrix<T>? InitialCenters { get; set; }

    /// <summary>
    /// Gets or sets the algorithm variant to use.
    /// </summary>
    /// <value>The algorithm variant. Default is Lloyd.</value>
    public KMeansAlgorithm Algorithm { get; set; } = KMeansAlgorithm.Lloyd;

    /// <summary>
    /// Gets or sets whether to precompute distances.
    /// </summary>
    /// <value>
    /// When true, computes full distance matrix (faster but uses O(n*k) memory).
    /// When false, computes distances on-the-fly (slower but memory efficient).
    /// Default is Auto.
    /// </value>
    public PrecomputeOption PrecomputeDistances { get; set; } = PrecomputeOption.Auto;

    /// <summary>
    /// Gets or sets whether to copy input data for safety.
    /// </summary>
    /// <value>When true, creates a copy of input data. Default is true.</value>
    public bool CopyX { get; set; } = true;
}

/// <summary>
/// Methods for initializing KMeans cluster centers.
/// </summary>
public enum KMeansInitMethod
{
    /// <summary>
    /// Random selection from data points.
    /// Fast but may produce suboptimal results.
    /// </summary>
    Random,

    /// <summary>
    /// KMeans++ smart initialization.
    /// Spreads initial centers apart for better convergence.
    /// This is the recommended default.
    /// </summary>
    KMeansPlusPlus,

    /// <summary>
    /// Custom initial centers provided in InitialCenters.
    /// </summary>
    Custom
}

/// <summary>
/// KMeans algorithm variants.
/// </summary>
public enum KMeansAlgorithm
{
    /// <summary>
    /// Standard Lloyd's algorithm (expectation-maximization).
    /// Most common, works well for most cases.
    /// </summary>
    Lloyd,

    /// <summary>
    /// Elkan's algorithm using triangle inequality.
    /// Faster for well-clustered data with Euclidean distance.
    /// Uses more memory for distance bounds.
    /// </summary>
    Elkan
}

/// <summary>
/// Options for precomputing distances.
/// </summary>
public enum PrecomputeOption
{
    /// <summary>
    /// Automatically decide based on data size.
    /// </summary>
    Auto,

    /// <summary>
    /// Always precompute (faster, more memory).
    /// </summary>
    True,

    /// <summary>
    /// Never precompute (slower, less memory).
    /// </summary>
    False
}
