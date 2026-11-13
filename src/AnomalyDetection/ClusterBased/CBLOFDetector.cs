using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.ClusterBased;

/// <summary>
/// Detects anomalies using Cluster-Based Local Outlier Factor (CBLOF).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> CBLOF combines clustering with local outlier detection.
/// It first clusters the data, then scores each point based on its distance to
/// its cluster center, weighted by the cluster size.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Cluster data into large and small clusters
/// 2. For large cluster points: score = cluster_size * distance_to_center
/// 3. For small cluster points: score = cluster_size * distance_to_nearest_large_cluster
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When anomalies are expected in small clusters or far from large clusters
/// - Faster than LOF for large datasets
/// - When cluster structure is meaningful
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - K (clusters): 8
/// - Alpha (large cluster threshold): 0.9 (90% of data)
/// - Beta (size threshold): 5 points
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: He, Z., et al. (2003). "Discovering Cluster-Based Local Outliers." Pattern Recognition Letters.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Clustering)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Discovering Cluster-Based Local Outliers", "https://doi.org/10.1016/S0167-8655(03)00003-5", Year = 2003, Authors = "Zengyou He, Xiaofei Xu, Shengchun Deng")]
public partial class CBLOFDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _nClusters;
    private readonly double _alpha;
    private readonly int _beta;
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _centroids;
    private int[]? _clusterSizes;
    private bool[]? _isLargeCluster;

    /// <summary>
    /// Gets the number of clusters.
    /// </summary>
    public int NClusters => _nClusters;

    /// <summary>
    /// Gets the alpha parameter (proportion for large clusters).
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the beta parameter (minimum size for large clusters).
    /// </summary>
    public int Beta => _beta;

    /// <summary>
    /// Creates a new CBLOF anomaly detector.
    /// </summary>
    /// <param name="nClusters">Number of clusters. Default is 8.</param>
    /// <param name="alpha">
    /// Fraction of total points that large clusters should contain.
    /// Default is 0.9 (90%). Clusters are marked as large if they collectively
    /// contain at least alpha * total_points.
    /// </param>
    /// <param name="beta">
    /// Minimum number of points for a cluster to be considered large.
    /// Default is 5.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public CBLOFDetector(int nClusters = 8, double alpha = 0.9, int beta = 5,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nClusters < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nClusters),
                "NClusters must be at least 1.");
        }

        if (alpha <= 0 || alpha > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha),
                "Alpha must be between 0 (exclusive) and 1 (inclusive). Recommended is 0.9.");
        }

        if (beta < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(beta),
                "Beta must be at least 1. Recommended is 5.");
        }

        _nClusters = nClusters;
        _alpha = alpha;
        _beta = beta;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows < _nClusters)
        {
            throw new ArgumentException(
                $"Number of samples ({X.Rows}) must be at least nClusters ({_nClusters}).",
                nameof(X));
        }

        // Run K-Means clustering
        var (centroids, assignments) = RunKMeans(X);
        _centroids = centroids;

        // Compute cluster sizes
        _clusterSizes = new int[_nClusters];
        for (int i = 0; i < X.Rows; i++)
        {
            _clusterSizes[assignments[i]]++;
        }

        // Determine large and small clusters
        _isLargeCluster = DetermineLargeClusters(X.Rows);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        var centroids = _centroids;
        var isLargeCluster = _isLargeCluster;
        var clusterSizes = _clusterSizes;
        if (centroids == null || isLargeCluster == null || clusterSizes == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var engine = Engine;
        int k = _nClusters;
        var large = new int[k];
        for (int c = 0; c < k; c++)
        {
            large[c] = isLargeCluster[c] ? 1 : 0;
        }

        var largeTensor = Tensor<int>.FromVector(Vector<int>.Wrap(large), new[] { k }).Cast<T>();
        var sizes = Tensor<int>.FromVector(Vector<int>.Wrap(clusterSizes), new[] { k }).Cast<T>();

        // Nearest cluster (first on ties) and the distance to it
        var distances = AnomalyTensorOps.PairwiseDistances(AnomalyTensorOps.ToTensor(X), AnomalyTensorOps.ToTensor(centroids));
        var nearestCluster = AnomalyTensorOps.FirstArgMinRows(distances);
        var nearestDist = engine.TensorAmin(distances, new[] { 1 }, false);

        // Distance to the nearest large cluster
        var smallColumns = engine.TensorEqScalar(AnomalyTensorOps.BroadcastRow(largeTensor, X.Rows), NumOps.Zero);
        var minLargeDist = engine.TensorAmin(engine.TensorMaskedFill(distances, smallColumns, NumOps.MaxValue), new[] { 1 }, false);

        // Large cluster: CBLOF = size * distance_to_centroid;
        // small cluster: CBLOF = size * distance_to_nearest_large_cluster
        return engine.TensorMultiply(
            engine.TensorGather(sizes, nearestCluster, 0),
            engine.TensorWhere(engine.TensorGather(largeTensor, nearestCluster, 0), nearestDist, minLargeDist)).ToVector();
    }

    private bool[] DetermineLargeClusters(int totalPoints)
    {
        var isLarge = new bool[_nClusters];
        var clusterSizes = _clusterSizes ?? throw new InvalidOperationException("_clusterSizes has not been initialized.");

        // Sort clusters by size (descending)
        var sortedClusters = Enumerable.Range(0, _nClusters)
            .OrderByDescending(c => clusterSizes[c])
            .ToList();

        // Mark clusters as large until we reach alpha * total_points
        int cumulativeSize = 0;
        int threshold = (int)(_alpha * totalPoints);

        foreach (int c in sortedClusters)
        {
            if (clusterSizes[c] >= _beta && cumulativeSize < threshold)
            {
                isLarge[c] = true;
                cumulativeSize += clusterSizes[c];
            }
        }

        // Ensure at least one large cluster
        if (!isLarge.Any(l => l) && sortedClusters.Count > 0)
        {
            isLarge[sortedClusters[0]] = true;
        }

        return isLarge;
    }

    private (Matrix<T> centroids, int[] assignments) RunKMeans(Matrix<T> X)
    {
        var random = RandomHelper.CreateSeededRandom(_randomSeed);
        int n = X.Rows;
        int k = _nClusters;
        var engine = Engine;
        var points = AnomalyTensorOps.ToTensor(X);

        // Initialize centroids randomly
        var initial = new int[k];
        var selectedIndices = new HashSet<int>();

        for (int c = 0; c < k; c++)
        {
            int idx;
            do
            {
                idx = random.Next(n);
            } while (selectedIndices.Contains(idx));

            selectedIndices.Add(idx);
            initial[c] = idx;
        }

        var centroids = AnomalyTensorOps.SelectRows(points, initial);
        var assignments = new int[n];
        int maxIterations = 100;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Assignment step (first nearest on ties)
            var nearest = AnomalyTensorOps.FirstArgMinRows(AnomalyTensorOps.PairwiseDistances(points, centroids)).ToArray();
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                if (assignments[i] != nearest[i])
                {
                    assignments[i] = nearest[i];
                    changed = true;
                }
            }

            if (!changed) break;

            // Update step: cluster means, empty clusters re-initialized from a random point
            centroids = AnomalyTensorOps.KMeansUpdate(points, assignments, k, random);
        }

        return (centroids.ToMatrix(), assignments);
    }
}
