using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using LDCOF (Local Density Cluster-Based Outlier Factor).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LDCOF combines clustering with density-based outlier detection.
/// It first clusters the data, then computes outlier scores based on how a point's density
/// compares to its cluster's density. Points in sparse regions of dense clusters are flagged.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Cluster the data using k-means
/// 2. Compute local density for each point
/// 3. Compare point density to cluster average density
/// 4. Large deviations indicate outliers
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When data has natural cluster structure
/// - For detecting local outliers within clusters
/// - When global methods miss cluster-specific anomalies
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Number of clusters: 8
/// - Number of neighbors (k): 10
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Amer, M., Goldstein, M. (2012). "Nearest-Neighbor and Clustering based
/// Anomaly Detection Algorithms for RapidMiner." Workshop on Open Source Data Mining Software.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Clustering)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Enhancing the Effectiveness of Clustering-Based Outlier Detection", "https://goldstein.center/publications.html", Year = 2012, Authors = "Mennatallah Amer, Markus Goldstein")]
public partial class LDCOFDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _numClusters;
    private readonly int _numNeighbors;
    [Buffer]
    private Matrix<T>? _trainingData;
    private Matrix<T>? _clusterCenters;
    private int[]? _clusterAssignments;
    private Vector<T>? _clusterDensities;

    /// <summary>
    /// Gets the number of clusters.
    /// </summary>
    public int NumClusters => _numClusters;

    /// <summary>
    /// Gets the number of neighbors for density estimation.
    /// </summary>
    public int NumNeighbors => _numNeighbors;

    /// <summary>
    /// Creates a new LDCOF anomaly detector.
    /// </summary>
    /// <param name="numClusters">Number of clusters. Default is 8.</param>
    /// <param name="numNeighbors">Number of neighbors for density. Default is 10.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LDCOFDetector(int numClusters = 8, int numNeighbors = 10,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (numClusters < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numClusters),
                "NumClusters must be at least 1. Recommended is 8.");
        }

        if (numNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numNeighbors),
                "NumNeighbors must be at least 1. Recommended is 10.");
        }

        _numClusters = numClusters;
        _numNeighbors = numNeighbors;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;

        _trainingData = X;

        // Run k-means clustering
        int effectiveClusters = Math.Min(_numClusters, n);
        (_clusterCenters, _clusterAssignments) = KMeansClustering(X, effectiveClusters);

        // Compute cluster densities
        _clusterDensities = ComputeClusterDensities();

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private (Matrix<T> centers, int[] assignments) KMeansClustering(Matrix<T> data, int k)
    {
        int n = data.Rows;
        var random = RandomHelper.CreateSeededRandom(_randomSeed);
        var engine = Engine;
        var points = AnomalyTensorOps.ToTensor(data);

        // Initialize centers randomly
        var indices = Enumerable.Range(0, n).OrderBy(_ => random.NextDouble()).Take(k).ToArray();
        var centers = AnomalyTensorOps.SelectRows(points, indices);

        var assignments = new int[n];
        int maxIterations = 100;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Assign points to nearest center (first on ties)
            var nearest = AnomalyTensorOps.FirstArgMinRows(AnomalyTensorOps.PairwiseDistances(points, centers)).ToArray();
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

            // Update centers: cluster means, empty clusters re-initialized from a random point
            var counts = new int[k];
            foreach (int c in assignments)
            {
                counts[c]++;
            }

            var replacement = new int[k];
            var empty = new int[k];
            for (int c = 0; c < k; c++)
            {
                if (counts[c] == 0)
                {
                    replacement[c] = random.Next(n);
                    empty[c] = 1;
                }
            }

            var membership = ClusterMembership(assignments, k);
            var means = AnomalyTensorOps.DivideColumn(engine.TensorMatMul(membership, points),
                engine.TensorMax(engine.ReduceSum(membership, new[] { 1 }, false), NumOps.One));
            var emptyMask = AnomalyTensorOps.BroadcastColumn(
                Tensor<int>.FromVector(Vector<int>.Wrap(empty), new[] { k }).Cast<T>(), data.Columns);
            centers = engine.TensorWhere(emptyMask, AnomalyTensorOps.SelectRows(points, replacement), means);
        }

        return (centers.ToMatrix(), assignments);
    }

    // [k, n] 0/1 membership of point i in cluster c.
    private Tensor<T> ClusterMembership(int[] assignments, int k)
    {
        var engine = Engine;
        int n = assignments.Length;
        var labels = Tensor<int>.FromVector(Vector<int>.Wrap(assignments), new[] { n }).Cast<T>();
        return engine.TensorEquals(
            AnomalyTensorOps.BroadcastColumn(engine.TensorArange<T>(0, k, 1), n),
            AnomalyTensorOps.BroadcastRow(labels, k));
    }

    private Vector<T> ComputeClusterDensities()
    {
        var clusterCenters = _clusterCenters;
        var trainingData = _trainingData;
        var clusterAssignments = _clusterAssignments;
        if (clusterCenters == null || trainingData == null || clusterAssignments == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var engine = Engine;
        int k = clusterCenters.Rows;
        int n = trainingData.Rows;
        var membership = ClusterMembership(clusterAssignments, k);
        var clusterSizes = engine.ReduceSum(membership, new[] { 1 }, false);

        // k-distance of every point within its own cluster (itself excluded by index): the
        // min(numNeighbors, size - 1)-th smallest distance, 0 for a single-point cluster
        var sizes = new int[k];
        foreach (int c in clusterAssignments)
        {
            sizes[c]++;
        }

        var kthPosition = new int[n];
        var hasNeighbors = new int[n];
        for (int i = 0; i < n; i++)
        {
            int effectiveK = Math.Min(_numNeighbors, sizes[clusterAssignments[i]] - 1);
            kthPosition[i] = Math.Max(effectiveK - 1, 0);
            hasNeighbors[i] = effectiveK > 0 ? 1 : 0;
        }

        var points = AnomalyTensorOps.ToTensor(trainingData);
        var sameCluster = engine.TensorMatMul(engine.TensorTranspose(membership), membership);
        var other = engine.TensorEqScalar(engine.TensorSubtract(sameCluster, engine.TensorEye<T>(n)), NumOps.Zero);
        var withinCluster = engine.TensorMaskedFill(AnomalyTensorOps.PairwiseDistances(points, points), other, NumOps.MaxValue);
        var (sorted, _) = engine.TensorSort(withinCluster, 1, false);
        var kDistances = engine.Reshape(engine.TensorTakeAlongDim(sorted,
            Tensor<int>.FromVector(Vector<int>.Wrap(kthPosition), new[] { n, 1 }), 1), new[] { n });
        kDistances = engine.TensorMultiply(kDistances,
            Tensor<int>.FromVector(Vector<int>.Wrap(hasNeighbors), new[] { n }).Cast<T>());

        // Average k-distance per cluster (inverse is density); an empty cluster has density 1
        var totalKDist = engine.Reshape(
            engine.TensorMatMul(membership, engine.Reshape(kDistances, new[] { n, 1 })), new[] { k });
        var nonEmpty = engine.TensorGreaterThan(clusterSizes, NumOps.Zero);
        var avgKDist = engine.TensorDivide(totalKDist, engine.TensorMax(clusterSizes, NumOps.One));
        var density = engine.TensorReciprocal(engine.TensorAddScalar(avgKDist, NumOps.FromDouble(1e-10)));
        return engine.TensorWhere(nonEmpty, density, Tensor<T>.CreateOnes(new[] { k })).ToVector();
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

        var clusterCenters = _clusterCenters;
        var trainingData = _trainingData;
        var clusterDensities = _clusterDensities;
        if (clusterCenters == null || trainingData == null || clusterDensities == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var engine = Engine;
        int nTraining = trainingData.Rows;
        int effectiveK = Math.Min(_numNeighbors, nTraining - 1);
        T eps = NumOps.FromDouble(1e-10);
        var points = AnomalyTensorOps.ToTensor(X);

        // Nearest cluster (first on ties)
        var nearestCluster = AnomalyTensorOps.FirstArgMinRows(
            AnomalyTensorOps.PairwiseDistances(points, AnomalyTensorOps.ToTensor(clusterCenters)));

        // Local density 1 / (k-distance + eps), the k-distance being the effectiveK-th smallest distance to the
        // training points (the largest when effectiveK is 0)
        var (sorted, _) = engine.TensorSort(AnomalyTensorOps.PairwiseDistances(points, AnomalyTensorOps.ToTensor(trainingData)), 1, false);
        var kDist = engine.TensorSliceAxis(sorted, 1, effectiveK > 0 ? effectiveK - 1 : nTraining - 1);
        var localDensity = engine.TensorReciprocal(engine.TensorAddScalar(kDist, eps));

        // LDCOF score: ratio of cluster density to local density
        return engine.TensorDivide(
            engine.TensorGather(clusterDensities.AsTensor(), nearestCluster, 0),
            engine.TensorAddScalar(localDensity, eps)).ToVector();
    }
}
