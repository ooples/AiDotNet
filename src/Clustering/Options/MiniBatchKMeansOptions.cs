namespace AiDotNet.Clustering.Options;

/// <summary>
/// Configuration options specific to MiniBatch K-Means clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MiniBatch K-Means is a variant of KMeans that uses mini-batches (random samples)
/// to reduce computation time for large datasets while producing similar results.
/// </para>
/// <para><b>For Beginners:</b> MiniBatch K-Means is like regular K-Means but faster.
///
/// Instead of using all data points in every step, it uses a random sample (mini-batch).
/// This makes it:
/// - Much faster for large datasets (millions of points)
/// - Uses less memory
/// - Produces slightly less optimal clusters (but usually very close)
///
/// Use MiniBatch K-Means when:
/// - Your dataset has more than ~10,000 points
/// - Speed is more important than perfect clustering
/// - You're doing online/streaming clustering
/// </para>
/// </remarks>
public class MiniBatchKMeansOptions<T> : ClusteringOptions<T>
{
    /// <summary>
    /// Gets or sets the number of clusters to find.
    /// </summary>
    /// <value>The number of clusters. Must be positive.</value>
    public int NumClusters { get; set; } = 8;

    /// <summary>
    /// Gets or sets the size of the mini-batches.
    /// </summary>
    /// <value>The batch size. Default is 1024.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls the trade-off between
    /// speed and quality:
    /// - Smaller batches: Faster iterations but more noise
    /// - Larger batches: Better quality but slower
    ///
    /// 1024 is a good default for most cases.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the initialization method for cluster centers.
    /// </summary>
    /// <value>The initialization method. Default is KMeansPlusPlus.</value>
    public KMeansInitMethod InitMethod { get; set; } = KMeansInitMethod.KMeansPlusPlus;

    /// <summary>
    /// Gets or sets custom initial cluster centers.
    /// </summary>
    /// <value>Matrix where each row is a cluster center, or null to use InitMethod.</value>
    public Matrix<T>? InitialCenters { get; set; }

    /// <summary>
    /// Gets or sets the number of iterations with no improvement to wait before stopping.
    /// </summary>
    /// <value>Number of iterations with no improvement. Default is 10.</value>
    /// <remarks>
    /// <para>
    /// Early stopping helps prevent overfitting and saves computation time.
    /// Set to 0 to disable early stopping.
    /// </para>
    /// </remarks>
    public int MaxNoImprovement { get; set; } = 10;

    /// <summary>
    /// Gets or sets the fraction of centers that must be reassigned in each iteration.
    /// </summary>
    /// <value>Minimum reassignment fraction. Default is 0.01 (1%).</value>
    /// <remarks>
    /// <para>
    /// If fewer than this fraction of points are reassigned to a different cluster,
    /// the algorithm considers it converged.
    /// </para>
    /// </remarks>
    public double ReassignmentRatio { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether to reassign empty clusters.
    /// </summary>
    /// <value>When true, empty clusters are reassigned to the farthest point. Default is true.</value>
    public bool ReassignEmptyClusters { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of random samples used to initialize centers.
    /// </summary>
    /// <value>Number of samples for initialization. Default is 3 times batch size.</value>
    public int? InitSize { get; set; }
}
