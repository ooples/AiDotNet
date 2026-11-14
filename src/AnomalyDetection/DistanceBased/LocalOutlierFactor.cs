using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Implements the Local Outlier Factor (LOF) algorithm for density-based anomaly detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LOF detects anomalies by comparing the local density of a point
/// to the local densities of its neighbors. Points in low-density regions (relative to
/// their neighbors) are considered anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Finding the k-nearest neighbors for each point
/// 2. Computing the local reachability density (LRD) for each point
/// 3. Comparing each point's LRD to its neighbors' LRDs to get the LOF score
/// 4. Points with LOF score significantly greater than 1 are anomalies
/// </para>
/// <para>
/// <b>When to use:</b> LOF is particularly effective for:
/// - Detecting local anomalies (points that are anomalies relative to their neighborhood)
/// - Data with varying densities across different regions
/// - When you want interpretable density-based detection
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Number of neighbors (k): 20 (good balance between local and global detection)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Breunig, M. M., Kriegel, H. P., Ng, R. T., and Sander, J. (2000).
/// "LOF: Identifying Density-Based Local Outliers." ACM SIGMOD Record.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("LOF: Identifying Density-Based Local Outliers", "https://doi.org/10.1145/335191.335388", Year = 2000, Authors = "Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, Joerg Sander")]
public partial class LocalOutlierFactor<T> : AnomalyDetectorBase<T>
{
    private readonly int _numNeighbors;
    [Buffer]
    private Matrix<T>? _trainingData;
    [Buffer]
    private Vector<T>[]? _kDistances;
    private int[][]? _neighborIndices;
    [Buffer]
    private Vector<T>? _lrd;

    /// <summary>
    /// Gets the number of neighbors used for LOF computation.
    /// </summary>
    public int NumNeighbors => _numNeighbors;

    /// <summary>
    /// Creates a new Local Outlier Factor detector.
    /// </summary>
    /// <param name="numNeighbors">
    /// The number of neighbors to use for computing LOF. Default is 20 (industry standard).
    /// Larger values provide more stable results but may miss local anomalies.
    /// Smaller values are more sensitive to local density variations.
    /// Common values: 5-50 depending on use case.
    /// </param>
    /// <param name="contamination">
    /// The expected proportion of anomalies in the data. Default is 0.1 (10%).
    /// Used to set the decision threshold after fitting.
    /// </param>
    /// <param name="randomSeed">
    /// Random seed for reproducibility. Default is 42.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The number of neighbors (k) is the main parameter to tune:
    /// - Use smaller k (5-10) if you expect very localized anomalies
    /// - Use larger k (20-50) for more global anomaly detection
    /// - If unsure, start with k=20 and adjust based on results
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when numNeighbors is less than 1.
    /// </exception>
    public LocalOutlierFactor(
        int numNeighbors = 20,
        double contamination = 0.1,
        int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (numNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numNeighbors),
                "Number of neighbors must be at least 1. Recommended value is 20.");
        }

        _numNeighbors = numNeighbors;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows <= _numNeighbors)
        {
            throw new ArgumentException(
                $"Number of samples ({X.Rows}) must be greater than number of neighbors ({_numNeighbors}).",
                nameof(X));
        }

        _trainingData = X;
        int n = X.Rows;
        int k = _numNeighbors;

        // Find k-nearest neighbors and k-distances for each point
        var data = AnomalyTensorOps.ToTensor(X);
        var distances = AnomalyTensorOps.NearestNeighbors(data, data, k, true, out var neighbors);
        _neighborIndices = AnomalyTensorOps.ToJagged(neighbors);
        _kDistances = AnomalyTensorOps.RowVectors(distances);

        // Compute Local Reachability Density (LRD) for each training point
        _lrd = ComputeLRD(distances, neighbors, Engine.TensorSliceAxis(distances, 1, k - 1)).ToVector();

        // Calculate LOF scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X, true);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X, false);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X, bool isTrainingData)
    {
        ValidateInput(X);

        var kDistances = _kDistances;
        var neighborIndices = _neighborIndices;
        var lrd = _lrd;
        var trainingData = _trainingData;
        if (kDistances == null || neighborIndices == null || lrd == null || trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var engine = Engine;
        int k = _numNeighbors;
        var lrdTensor = lrd.AsTensor();

        // Use precomputed neighbourhoods for training data, otherwise find them in the training data
        Tensor<T> pointLrd;
        Tensor<int> neighbors;
        if (isTrainingData)
        {
            neighbors = AnomalyTensorOps.FromJagged(neighborIndices, k);
            pointLrd = lrdTensor;
        }
        else
        {
            var distances = AnomalyTensorOps.NearestNeighbors(
                AnomalyTensorOps.ToTensor(X), AnomalyTensorOps.ToTensor(trainingData), k, false, out neighbors);
            var trainingKDistances = engine.TensorSliceAxis(AnomalyTensorOps.StackRows(kDistances), 1, k - 1);
            pointLrd = ComputeLRD(distances, neighbors, trainingKDistances);
        }

        // LOF = average of lrd(neighbor) / lrd(point); the ratios count only when lrd(point) > 0.
        // LOF > 1 indicates anomaly, higher = more anomalous
        var ratios = AnomalyTensorOps.DivideColumn(AnomalyTensorOps.GatherByIndex(lrdTensor, neighbors), pointLrd);
        var lofSum = engine.TensorWhere(
            engine.TensorGreaterThan(pointLrd, NumOps.Zero),
            engine.ReduceSum(ratios, new[] { 1 }, false),
            Tensor<T>.CreateZeros(new[] { X.Rows }));
        return engine.TensorDivideScalar(lofSum, NumOps.FromDouble(k)).ToVector();
    }

    // Local reachability density of points from their k neighbour distances ([m, k]) and indices:
    // k / sum_j max(k-distance(neighbor_j), d(point, neighbor_j)), or MaxValue when that sum is 0.
    private Tensor<T> ComputeLRD(Tensor<T> distances, Tensor<int> neighbors, Tensor<T> referenceKDistances)
    {
        var engine = Engine;
        int m = distances.Shape[0];
        var reachDist = engine.TensorMax(AnomalyTensorOps.GatherByIndex(referenceKDistances, neighbors), distances);
        var reachabilitySum = engine.ReduceSum(reachDist, new[] { 1 }, false);
        var lrd = engine.TensorDivide(
            engine.Fill(m, NumOps.FromDouble(_numNeighbors)).AsTensor(), reachabilitySum);
        return engine.TensorWhere(engine.TensorGreaterThan(reachabilitySum, NumOps.Zero), lrd, engine.Fill(m, NumOps.MaxValue).AsTensor());
    }
}
