namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for K-Nearest Neighbors classifiers.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm that classifies
/// samples based on the majority class among their k nearest neighbors in the feature space.
/// </para>
/// <para><b>For Beginners:</b> KNN is like asking your neighbors for advice!
///
/// When you need to classify a new sample:
/// 1. Find the k training samples closest to it
/// 2. Look at what classes those neighbors belong to
/// 3. Predict the most common class among those neighbors
///
/// Example: To predict if a movie is "Action" or "Comedy":
/// - Find 5 similar movies (based on runtime, budget, etc.)
/// - If 4 are Action and 1 is Comedy, predict "Action"
///
/// Key settings:
/// - K (N_Neighbors): How many neighbors to consider (default: 5)
/// - Metric: How to measure distance (Euclidean, Manhattan, etc.)
/// </para>
/// </remarks>
public class KNeighborsOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of neighbors to use for classification.
    /// </summary>
    /// <value>
    /// The number of neighbors (k). Default is 5.
    /// </value>
    /// <remarks>
    /// <para>
    /// Smaller values of k make the model more sensitive to noise but can capture
    /// local patterns. Larger values provide smoother decision boundaries but may
    /// miss local patterns.
    /// </para>
    /// <para><b>For Beginners:</b> K is the number of neighbors to ask for their opinion.
    ///
    /// - K = 1: Only look at the single closest neighbor (very sensitive to noise)
    /// - K = 5: Look at 5 closest neighbors (good balance)
    /// - K = 20: Look at 20 neighbors (smoother but may ignore local patterns)
    ///
    /// Odd values are often preferred to avoid ties in binary classification.
    /// </para>
    /// </remarks>
    public int NNeighbors { get; set; } = 5;

    /// <summary>
    /// Gets or sets the distance metric used to find nearest neighbors.
    /// </summary>
    /// <value>
    /// The distance metric. Default is Euclidean.
    /// </value>
    /// <remarks>
    /// <para>
    /// The choice of metric affects which points are considered "nearest."
    /// Euclidean distance works well for continuous features with similar scales.
    /// Manhattan distance can be better for high-dimensional data.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how we measure "closeness."
    ///
    /// - Euclidean: Straight-line distance (like a bird flying)
    /// - Manhattan: Distance along axes (like walking city blocks)
    /// - Minkowski: Generalization of both (with parameter p)
    ///
    /// Euclidean is the most common choice for most problems.
    /// </para>
    /// </remarks>
    public DistanceMetric Metric { get; set; } = DistanceMetric.Euclidean;

    /// <summary>
    /// Gets or sets the weight function used in prediction.
    /// </summary>
    /// <value>
    /// The weighting scheme. Default is Uniform.
    /// </value>
    /// <remarks>
    /// <para>
    /// Uniform weighting treats all neighbors equally. Distance weighting gives
    /// closer neighbors more influence on the prediction.
    /// </para>
    /// <para><b>For Beginners:</b> Should all neighbors have equal say?
    ///
    /// - Uniform: Every neighbor's vote counts equally
    /// - Distance: Closer neighbors count more (weight = 1/distance)
    ///
    /// Distance weighting often works better because the closest neighbors
    /// are usually more relevant.
    /// </para>
    /// </remarks>
    public WeightingScheme Weights { get; set; } = WeightingScheme.Uniform;

    /// <summary>
    /// Gets or sets the power parameter for the Minkowski metric.
    /// </summary>
    /// <value>
    /// The Minkowski power parameter. Default is 2 (Euclidean).
    /// </value>
    /// <remarks>
    /// <para>
    /// Only used when Metric is Minkowski.
    /// p = 1 is equivalent to Manhattan distance.
    /// p = 2 is equivalent to Euclidean distance.
    /// </para>
    /// </remarks>
    public double P { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the algorithm used to compute nearest neighbors.
    /// </summary>
    /// <value>
    /// The search algorithm. Default is Auto.
    /// </value>
    /// <remarks>
    /// <para>
    /// Auto chooses the best algorithm based on data characteristics.
    /// BruteForce computes all pairwise distances (O(n*d) per query).
    /// KDTree uses a tree structure for faster queries in low dimensions.
    /// BallTree is better for high-dimensional data.
    /// </para>
    /// <para><b>For Beginners:</b> How to find neighbors efficiently.
    ///
    /// - Auto: Let the algorithm choose (recommended)
    /// - BruteForce: Compare to every training sample (slow but accurate)
    /// - KDTree: Use a tree structure (fast for small dimensions)
    /// - BallTree: Better for many dimensions
    /// </para>
    /// </remarks>
    public KNNAlgorithm Algorithm { get; set; } = KNNAlgorithm.Auto;

    /// <summary>
    /// Gets or sets the leaf size for tree-based algorithms.
    /// </summary>
    /// <value>
    /// The leaf size for KDTree or BallTree. Default is 30.
    /// </value>
    /// <remarks>
    /// <para>
    /// This affects the speed of tree construction and query, as well as
    /// memory requirements. Larger values create shallower trees.
    /// </para>
    /// </remarks>
    public int LeafSize { get; set; } = 30;
}

/// <summary>
/// Distance metrics for measuring similarity between samples.
/// </summary>
public enum DistanceMetric
{
    /// <summary>
    /// Euclidean (L2) distance - straight-line distance.
    /// </summary>
    Euclidean = 0,

    /// <summary>
    /// Manhattan (L1) distance - sum of absolute differences.
    /// </summary>
    Manhattan = 1,

    /// <summary>
    /// Minkowski distance - generalized form with parameter p.
    /// </summary>
    Minkowski = 2,

    /// <summary>
    /// Chebyshev (L-infinity) distance - maximum absolute difference.
    /// </summary>
    Chebyshev = 3,

    /// <summary>
    /// Cosine distance - angle between feature vectors.
    /// </summary>
    Cosine = 4
}

/// <summary>
/// Weighting schemes for neighbor voting.
/// </summary>
public enum WeightingScheme
{
    /// <summary>
    /// All neighbors contribute equally.
    /// </summary>
    Uniform = 0,

    /// <summary>
    /// Closer neighbors contribute more (weight = 1/distance).
    /// </summary>
    Distance = 1
}

/// <summary>
/// Algorithms for finding nearest neighbors.
/// </summary>
public enum KNNAlgorithm
{
    /// <summary>
    /// Automatically choose the best algorithm.
    /// </summary>
    Auto = 0,

    /// <summary>
    /// Compute all pairwise distances (O(n*d) per query).
    /// </summary>
    BruteForce = 1,

    /// <summary>
    /// K-D tree for faster queries in low dimensions.
    /// </summary>
    KDTree = 2,

    /// <summary>
    /// Ball tree for higher-dimensional data.
    /// </summary>
    BallTree = 3
}
