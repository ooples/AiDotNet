using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.ClusterBased;

/// <summary>
/// Detects anomalies using K-Means clustering distance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> K-Means clustering groups data into k clusters based on distance
/// to cluster centers (centroids). Points far from their nearest centroid are considered
/// anomalies because they don't fit well into any cluster.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Partition data into k clusters using K-Means
/// 2. For each point, find distance to nearest centroid
/// 3. Points with large distances are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When data naturally forms spherical clusters
/// - When you have a rough idea of the number of clusters
/// - Fast and scalable to large datasets
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - K (clusters): 8 (or estimated from data)
/// - Max iterations: 100
/// - Contamination: 0.1 (10%)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Clustering)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Some Methods for Classification and Analysis of Multivariate Observations", "https://doi.org/10.1007/978-3-540-74958-5_28")]
public partial class KMeansDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    private readonly int _maxIterations;
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _centroids;
    private int[]? _clusterSizes;
    private int _totalSamples;

    /// <summary>
    /// Gets the number of clusters.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Gets the maximum number of iterations.
    /// </summary>
    public int MaxIterations => _maxIterations;

    /// <summary>
    /// Creates a new K-Means anomaly detector.
    /// </summary>
    /// <param name="k">Number of clusters. Default is 8.</param>
    /// <param name="maxIterations">Maximum iterations for K-Means. Default is 100.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public KMeansDetector(int k = 8, int maxIterations = 100, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (k < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(k),
                "K must be at least 1. Recommended value is 8 or use the elbow method.");
        }

        if (maxIterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxIterations),
                "MaxIterations must be at least 1.");
        }

        _k = k;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows < _k)
        {
            throw new ArgumentException(
                $"Number of samples ({X.Rows}) must be at least k ({_k}).",
                nameof(X));
        }

        // Run K-Means clustering and compute cluster sizes
        _totalSamples = X.Rows;
        _centroids = RunKMeans(X, out _clusterSizes);

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
        if (centroids == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var engine = Engine;
        int k = centroids.Rows;

        // Distance to the nearest centroid (first on ties) and that cluster
        var distances = AnomalyTensorOps.PairwiseDistances(AnomalyTensorOps.ToTensor(X), AnomalyTensorOps.ToTensor(centroids));
        var nearestCluster = engine.TensorArgMin(distances, 1);
        var minDist = engine.TensorAmin(distances, new[] { 1 }, false);

        // Weight the distance by inverse cluster size fraction.
        // Points in small clusters (e.g., a singleton outlier that became its own centroid)
        // get amplified scores. A point with distance 0 in a singleton cluster gets
        // a high score because being in a tiny cluster is itself anomalous.
        var sizes = _clusterSizes != null
            ? Tensor<int>.FromVector(Vector<int>.Wrap(_clusterSizes), new[] { k }).Cast<T>()
            : Tensor<T>.CreateZeros(new[] { k });
        var fractions = engine.TensorDivideScalar(sizes, NumOps.FromDouble(_totalSamples));
        var clusterFraction = engine.TensorWhere(engine.TensorGreaterThan(sizes, NumOps.Zero), fractions,
            Tensor<T>.CreateOnes(new[] { k }));

        // Score = distance + penalty for small clusters
        // The penalty term ensures singleton cluster members get high scores even with distance 0
        var smallClusterPenalty = engine.ScalarMinusTensor(NumOps.One, engine.TensorGather(clusterFraction, nearestCluster, 0));
        return engine.TensorAdd(minDist, smallClusterPenalty).ToVector();
    }

    private Matrix<T> RunKMeans(Matrix<T> X, out int[] clusterSizes)
    {
        var random = RandomHelper.CreateSeededRandom(_randomSeed);
        int n = X.Rows;
        int k = _k;
        var engine = Engine;
        var points = AnomalyTensorOps.ToTensor(X);

        // Initialize centroids using K-Means++
        var centroids = InitializeCentroids(points, random);

        // Assignment array
        var assignments = new int[n];

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Assignment step: assign each point to nearest centroid (first on ties)
            var nearest = engine.TensorArgMin(AnomalyTensorOps.PairwiseDistances(points, centroids), 1).ToArray();
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

            // Update step: recalculate centroids, re-initializing empty clusters from a random point
            var counts = new int[k];
            foreach (int cluster in assignments)
            {
                counts[cluster]++;
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

            var labels = Tensor<int>.FromVector(Vector<int>.Wrap(assignments), new[] { n }).Cast<T>();
            var membership = engine.TensorEquals(
                AnomalyTensorOps.BroadcastColumn(engine.TensorArange<T>(0, k, 1), n),
                AnomalyTensorOps.BroadcastRow(labels, k));
            var means = AnomalyTensorOps.DivideColumn(engine.TensorMatMul(membership, points),
                engine.TensorMax(engine.ReduceSum(membership, new[] { 1 }, false), NumOps.One));
            var emptyMask = AnomalyTensorOps.BroadcastColumn(
                Tensor<int>.FromVector(Vector<int>.Wrap(empty), new[] { k }).Cast<T>(), X.Columns);
            centroids = engine.TensorWhere(emptyMask, AnomalyTensorOps.SelectRows(points, replacement), means);
        }

        // Compute final cluster sizes for anomaly scoring
        clusterSizes = new int[k];
        foreach (int cluster in assignments)
        {
            clusterSizes[cluster]++;
        }

        return centroids.ToMatrix();
    }

    private Tensor<T> InitializeCentroids(Tensor<T> points, Random random)
    {
        // K-Means++ initialization
        var engine = Engine;
        int n = points.Shape[0];
        int k = _k;

        // Random draws in master's order: the first centroid index, then one uniform draw per further centroid
        int firstIdx = random.Next(n);
        var draws = new double[Math.Max(k - 1, 0)];
        for (int c = 1; c < k; c++)
        {
            draws[c - 1] = random.NextDouble();
        }

        var uniform = Tensor<double>.FromVector(Vector<double>.Wrap(draws), new[] { draws.Length }).Cast<T>();
        var chosen = new Tensor<T>[k];
        chosen[0] = engine.TensorSliceAxis(points, 0, firstIdx);

        // Distance of every point to its nearest chosen centroid, updated as centroids are added
        var minDist = engine.Fill(n, NumOps.MaxValue).AsTensor();
        for (int c = 1; c < k; c++)
        {
            minDist = engine.TensorMin(minDist, engine.TensorSqrt(
                AnomalyTensorOps.SquaredDistancesToPoint(points, chosen[c - 1])));

            // Next centroid with probability proportional to distance squared: the first point whose
            // cumulative weight reaches uniform * total (the first point when none does)
            var cumulative = engine.TensorCumSum(engine.TensorSquare(minDist), 0);
            var total = engine.TensorSlice(cumulative, new[] { n - 1 }, new[] { 1 });
            var threshold = engine.TensorBroadcastTo(
                engine.TensorMultiply(engine.TensorSlice(uniform, new[] { c - 1 }, new[] { 1 }), total), new[] { n });
            var reached = engine.TensorAdd(
                engine.TensorGreaterThan(cumulative, threshold), engine.TensorEquals(cumulative, threshold));
            int selectedIdx = engine.TensorArgMax(reached, 0).ToArray()[0];
            chosen[c] = engine.TensorSliceAxis(points, 0, selectedIdx);
        }

        return engine.TensorStack(chosen, 0);
    }
}
