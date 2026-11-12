using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.ClusterBased;

/// <summary>
/// Detects anomalies using DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DBSCAN groups together points that are closely packed and
/// marks points in low-density regions as outliers (noise).
/// </para>
/// <para>
/// The algorithm works by:
/// 1. For each point, find all neighbors within epsilon distance
/// 2. If a point has at least minPts neighbors, it's a core point
/// 3. Expand clusters from core points
/// 4. Points not belonging to any cluster are noise (anomalies)
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When anomalies are in low-density regions
/// - When clusters have arbitrary shapes
/// - No need to specify number of clusters in advance
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Epsilon: estimated from data (k-distance method)
/// - MinPts: 2 * dimensions
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Ester, M., et al. (1996). "A Density-Based Algorithm for Discovering Clusters." KDD.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Clustering)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise", "https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf", Year = 1996, Authors = "Martin Ester, Hans-Peter Kriegel, Joerg Sander, Xiaowei Xu")]
public partial class DBSCANDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double? _epsilon;
    private readonly int? _minPts;
    private T _fittedEpsilon;
    private int _fittedMinPts;
    private int[]? _clusterLabels;
    [Buffer]
    private Matrix<T>? _trainingData;

    /// <summary>
    /// Gets the epsilon (neighborhood radius) parameter.
    /// </summary>
    public T Epsilon => _fittedEpsilon;

    /// <summary>
    /// Gets the minimum points parameter.
    /// </summary>
    public int MinPts => _fittedMinPts;

    /// <summary>
    /// Creates a new DBSCAN anomaly detector.
    /// </summary>
    /// <param name="epsilon">Neighborhood radius. If null, estimated automatically from the data.</param>
    /// <param name="minPts">Minimum number of points to form a dense region. If null, uses 2 * dimensions.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public DBSCANDetector(double? epsilon = null, int? minPts = null, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (epsilon.HasValue && epsilon.Value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon),
                "Epsilon must be positive if specified.");
        }

        if (minPts.HasValue && minPts.Value < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(minPts),
                "MinPts must be at least 1 if specified.");
        }

        _epsilon = epsilon;
        _minPts = minPts;
        _fittedEpsilon = NumOps.Zero;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _trainingData = X;

        // Estimate parameters if not provided
        _fittedMinPts = _minPts ?? Math.Max(2, 2 * X.Columns);
        _fittedEpsilon = _epsilon.HasValue
            ? NumOps.FromDouble(_epsilon.Value)
            : EstimateEpsilon(X);

        // Run DBSCAN clustering
        _clusterLabels = RunDBSCAN(X);

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

        var trainingData = _trainingData ?? throw new InvalidOperationException("Model not properly fitted.");
        var engine = Engine;

        // Neighbour counts within epsilon (a NaN distance is never "greater", so it counts, as before),
        // over bounded blocks of query rows so only a [block, nTrain] distance matrix is live
        var training = AnomalyTensorOps.ToTensor(trainingData);
        T epsilon = _fittedEpsilon;
        var outside = AnomalyTensorOps.MapRowBlocks(AnomalyTensorOps.ToTensor(X), trainingData.Rows,
            block => engine.ReduceSum(engine.TensorGreaterThan(AnomalyTensorOps.PairwiseDistances(block, training), epsilon),
                new[] { 1 }, false));
        var neighborCount = engine.ScalarMinusTensor(NumOps.FromDouble(trainingData.Rows), outside);

        // Core point (count >= minPts): 1 / (count + 1). Border point: 1 - count / minPts, which is 1 for
        // a noise point (count = 0).
        T minPts = NumOps.FromDouble(_fittedMinPts);
        var core = engine.TensorReciprocal(engine.TensorAddScalar(neighborCount, NumOps.One));
        var border = engine.ScalarMinusTensor(NumOps.One, engine.TensorDivideScalar(neighborCount, minPts));
        return engine.TensorWhere(engine.TensorLessThan(neighborCount, minPts), border, core).ToVector();
    }

    private int[] RunDBSCAN(Matrix<T> X)
    {
        int n = X.Rows;
        var trainingData = _trainingData ?? throw new InvalidOperationException("Model not properly fitted.");
        var points = AnomalyTensorOps.ToTensor(X);
        var training = ReferenceEquals(X, trainingData) ? points : AnomalyTensorOps.ToTensor(trainingData);
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = -1; // -1 = unvisited
        }

        int clusterId = 0;

        for (int i = 0; i < n; i++)
        {
            if (labels[i] != -1) continue;

            var neighbors = GetNeighborsInRadius(points, training, i);

            if (neighbors.Count < _fittedMinPts)
            {
                labels[i] = 0; // Noise
            }
            else
            {
                clusterId++;
                ExpandCluster(points, training, i, neighbors, labels, clusterId);
            }
        }

        return labels;
    }

    private void ExpandCluster(Tensor<T> points, Tensor<T> training, int pointIdx, List<int> neighbors, int[] labels,
        int clusterId)
    {
        labels[pointIdx] = clusterId;

        var queue = new Queue<int>(neighbors);

        while (queue.Count > 0)
        {
            int current = queue.Dequeue();

            if (labels[current] == 0)
            {
                labels[current] = clusterId;
            }

            if (labels[current] != -1) continue;

            labels[current] = clusterId;

            var currentNeighbors = GetNeighborsInRadius(points, training, current);

            if (currentNeighbors.Count >= _fittedMinPts)
            {
                foreach (int neighbor in currentNeighbors)
                {
                    if (labels[neighbor] <= 0)
                    {
                        queue.Enqueue(neighbor);
                    }
                }
            }
        }
    }

    private T EstimateEpsilon(Matrix<T> X)
    {
        // K-distance method: find the elbow in sorted k-distances
        int k = _fittedMinPts;
        int n = X.Rows;
        if (n - 1 < k)
        {
            // No point has k other points, so there is no k-distance to take a percentile of.
            throw new ArgumentOutOfRangeException(nameof(X),
                $"Estimating epsilon needs more than minPts ({k}) samples; got {n}.");
        }

        var engine = Engine;

        // Distance of every point to its k-th nearest other point (self pushed to the end of each row), in
        // bounded row blocks so only a [block, n] distance matrix is live; row r of the block starting at
        // first is point first + r, so its self entry is the diagonal j - r = first.
        var points = AnomalyTensorOps.ToTensor(X);
        int d = X.Columns;
        int rowsPerBlock = Math.Max(1, Math.Min(n, EpsilonBlockElements / n));
        var kParts = new List<Tensor<T>>();
        for (int first = 0; first < n; first += rowsPerBlock)
        {
            int count = Math.Min(rowsPerBlock, n - first);
            var distances = AnomalyTensorOps.PairwiseDistances(
                engine.TensorSlice(points, new[] { first, 0 }, new[] { count, d }), points);
            var ones = Tensor<T>.CreateOnes(new[] { count, n });
            var self = engine.TensorEqScalar(
                engine.TensorSubtract(engine.TensorTril(ones, first), engine.TensorTril(ones, first - 1)), NumOps.One);
            var (sortedRows, _) = engine.TensorSort(engine.TensorMaskedFill(distances, self, NumOps.MaxValue), 1, false);
            kParts.Add(engine.TensorSliceAxis(sortedRows, 1, k - 1));
        }

        var kDistances = kParts.Count == 1 ? kParts[0] : engine.TensorConcatenate(kParts.ToArray(), 0);

        // Use the 90th percentile k-distance as epsilon estimate
        var (sortedKDistances, _) = engine.TensorSort(kDistances, 0, false);
        int elbowIdx = (int)(n * 0.9);
        return sortedKDistances[Math.Min(elbowIdx, n - 1)];
    }

    // Upper bound on the elements of one [rows, n] block of the k-distance computation.
    private const int EpsilonBlockElements = 1 << 22;

    private List<int> GetNeighborsInRadius(Tensor<T> points, Tensor<T> training, int pointIdx)
    {
        // Training points whose distance is not greater than epsilon, in index order.
        var engine = Engine;
        var point = engine.TensorSliceAxis(points, 0, pointIdx);
        var dist = engine.TensorSqrt(AnomalyTensorOps.SquaredDistancesToPoint(training, point));
        var within = engine.ScalarMinusTensor(NumOps.One, engine.TensorGreaterThan(dist, _fittedEpsilon));
        return new List<int>(AnomalyTensorOps.NonzeroIndices(within));
    }
}
