using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.Partitioning;

/// <summary>
/// Mini-Batch K-Means clustering algorithm for large-scale clustering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Mini-Batch K-Means uses small random batches of data to update cluster centers,
/// making it much faster than standard K-Means for large datasets while producing
/// similar results.
/// </para>
/// <para>
/// Algorithm steps:
/// 1. Initialize k cluster centers
/// 2. For each iteration:
///    a. Sample a mini-batch of points
///    b. Assign each sample to its nearest center
///    c. Update centers using a gradient descent-like step
/// 3. Repeat until convergence or max iterations
/// </para>
/// <para><b>For Beginners:</b> Think of Mini-Batch K-Means as "K-Means on a diet."
///
/// Instead of looking at all your data in each step (which is slow for big data),
/// it only looks at a random sample. This is much faster but gives nearly the
/// same result.
///
/// Speed comparison:
/// - 1 million points: Standard K-Means ~minutes, Mini-Batch ~seconds
/// - 10 million points: Standard K-Means ~hours, Mini-Batch ~minutes
///
/// The trade-off is slightly less optimal clustering, but usually the
/// difference is very small (a few percent in inertia).
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var newData = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for mini-batch K-Means clustering
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new MiniBatchKMeans&lt;double&gt;(new MiniBatchKMeansOptions&lt;double&gt;()));
///
/// var result = builder.Build(dataMatrix, labels);
/// var predictions = result.Predict(newData);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Web-Scale K-Means Clustering", "https://doi.org/10.1145/1772690.1772862", Year = 2010, Authors = "David Sculley")]
public partial class MiniBatchKMeans<T> : ClusteringBase<T>
{
    private readonly MiniBatchKMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Random _random;
    private int _numIterations;
    private int[] _centerCounts;

    /// <summary>
    /// Initializes a new MiniBatchKMeans instance with the specified options.
    /// </summary>
    /// <param name="options">The MiniBatchKMeans configuration options.</param>
    public MiniBatchKMeans(MiniBatchKMeansOptions<T>? options = null)
        : base(options ??= new MiniBatchKMeansOptions<T>())
    {
        _options = options;
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSeededRandom(42);
        _centerCounts = new int[_options.NumClusters];

        if (_options.DistanceMetric is null)
        {
            _options.DistanceMetric = new EuclideanDistance<T>();
        }

        // Set NumClusters from options
        NumClusters = _options.NumClusters;
    }

    /// <summary>
    /// Gets the number of iterations from the last fit.
    /// </summary>
    public int NumIterations => _numIterations;

    /// <inheritdoc />

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (MiniBatchKMeans<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidateInputData(x);

        Matrix<T>? bestCenters = null;
        Vector<T>? bestLabels = null;
        T bestInertia = NumOps.MaxValue;

        // Run multiple initializations and keep the best (the first one with the lowest inertia)
        int numInits = _options.InitMethod == KMeansInitMethod.Custom ? 1 : _options.NumInitializations;
        var runs = new List<(Matrix<T> Centers, Vector<T> Labels, T Inertia, int Iterations)>(numInits);
        for (int init = 0; init < numInits; init++)
        {
            runs.Add(FitSingle(x));
        }

        if (runs.Count > 0)
        {
            int best = ClusteringTensorOps.ArgMin(Vector<T>.FromEnumerable(runs.Select(run => run.Inertia)));
            bestCenters = runs[best].Centers;
            bestLabels = runs[best].Labels;
            bestInertia = runs[best].Inertia;
            _numIterations = runs[best].Iterations;
        }

        ClusterCenters = bestCenters;
        Labels = bestLabels;
        Inertia = bestInertia;
        MergeDegenerateClusters(x);

        // After merge, _centerCounts must match the post-merge cluster count.
        // NumClusters and ClusterCenters.Rows may now be smaller than _options.NumClusters.
        _centerCounts = Labels is not null && NumClusters > 0
            ? ClusteringTensorOps.ToIntArray(ClusteringTensorOps.ClusterCounts(Labels, NumClusters))
            : new int[NumClusters];

        IsTrained = true;
    }

    /// <summary>
    /// Performs a partial fit on a batch of data (for online/streaming learning).
    /// </summary>
    /// <param name="x">The batch of data to fit.</param>
    public void PartialFit(Matrix<T> x)
    {
        if (ClusterCenters is null)
        {
            // First call: initialize centers
            int initSize = _options.InitSize ?? Math.Min(3 * _options.BatchSize, x.Rows);
            var initData = SampleBatch(x, initSize);
            ClusterCenters = InitializeCenters(initData);
            _centerCounts = new int[_options.NumClusters];
        }

        // Update centers with this batch
        UpdateCentersWithBatch(x);
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        // Return stored labels for in-sample prediction (preserves merge results)
        if (Labels is not null && ReferenceEquals(x, TrainingDataRef))
            return new Vector<T>(Labels);

        if (ClusterCenters is null)
            throw new InvalidOperationException("MiniBatchKMeans: ClusterCenters not initialized. Call Train() first.");

        return AssignLabels(x, ClusterCenters);
    }

    /// <inheritdoc />
    public override Matrix<T> Transform(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        if (ClusterCenters is null)
            throw new InvalidOperationException("MiniBatchKMeans: ClusterCenters not initialized. Call Train() first.");

        return ComputeDistancesToCenters(x, ClusterCenters);
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? throw new InvalidOperationException("Training failed to produce cluster labels.");
    }

    private (Matrix<T> Centers, Vector<T> Labels, T Inertia, int Iterations) FitSingle(Matrix<T> x)
    {
        // Initialize centers
        int initSize = _options.InitSize ?? Math.Min(3 * _options.BatchSize, x.Rows);
        var initData = SampleBatch(x, initSize);
        var centers = InitializeCenters(initData);
        _centerCounts = new int[_options.NumClusters];

        T prevInertia = NumOps.MaxValue;
        int noImprovementCount = 0;
        int iterations = 0;

        for (iterations = 0; iterations < _options.MaxIterations; iterations++)
        {
            // Sample a mini-batch
            var batch = SampleBatch(x, _options.BatchSize);

            // Update centers with this batch
            UpdateCentersWithBatch(batch, centers);

            // Compute inertia on a sample (for efficiency)
            var sampleForInertia = SampleBatch(x, Math.Min(1000, x.Rows));
            var sampleLabels = AssignLabels(sampleForInertia, centers);
            T inertia = ComputeInertia(sampleForInertia, sampleLabels, centers);

            // Check for improvement
            if (ImprovedLessThan(prevInertia, inertia, _options.Tolerance))
            {
                noImprovementCount++;
                if (_options.MaxNoImprovement > 0 && noImprovementCount >= _options.MaxNoImprovement)
                {
                    break;
                }
            }
            else
            {
                noImprovementCount = 0;
            }

            prevInertia = inertia;
        }

        // Compute final labels and inertia on all data
        var labels = AssignLabels(x, centers);
        var finalInertia = ComputeInertia(x, labels, centers);

        return (centers, labels, finalInertia, iterations);
    }

    /// <summary>True when the inertia dropped by less than the tolerance (or rose).</summary>
    private bool ImprovedLessThan(T previous, T current, double tolerance)
        => NumOps.ToDouble(previous) - NumOps.ToDouble(current) < tolerance;

    private void UpdateCentersWithBatch(Matrix<T> batch, Matrix<T>? centers = null)
    {
        centers ??= ClusterCenters ?? throw new InvalidOperationException("Cluster centers not initialized.");
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Assign each point in the batch to its nearest center
        var batchDistances = distanceMetric.ComputePairwise(batch, centers);
        var nearestCenters = ClusteringTensorOps.ArgMinRows(batchDistances);
        var distances = ClusteringTensorOps.MinRows(batchDistances);

        // Streaming average: processing the batch points one by one with eta = 1 / count is the running mean
        // c <- (N * c + sum of new points) / (N + m) for a center with N earlier and m new points.
        var batchLabels = ClusteringTensorOps.FromIntArray<T>(nearestCenters);
        var sums = ClusteringTensorOps.ClusterSums(batch, batchLabels, centers.Rows, out var batchCounts);
        var priorCounts = ClusteringTensorOps.FromIntArray<T>(_centerCounts.Length == centers.Rows
            ? _centerCounts
            : _centerCounts.Take(centers.Rows).ToArray());
        var totals = Engine.Add(priorCounts, batchCounts);
        var weighted = Engine.TensorAdd(
            Engine.TensorMultiply(Tensor<T>.FromMatrix(centers),
                Engine.TensorBroadcastTo(Tensor<T>.FromVector(priorCounts, new[] { centers.Rows, 1 }), new[] { centers.Rows, centers.Columns })),
            Tensor<T>.FromMatrix(sums)).ToMatrix();
        var updated = ClusteringTensorOps.RowsWherePositive(batchCounts, ClusteringTensorOps.DivideRows(weighted, totals), centers);
        centers.SetSubMatrix(0, 0, updated);

        var added = ClusteringTensorOps.ToIntArray(batchCounts);
        for (int k = 0; k < added.Length; k++)
        {
            _centerCounts[k] += added[k];
        }

        // Handle empty clusters
        if (_options.ReassignEmptyClusters)
        {
            ReassignEmptyClusters(batch, centers, nearestCenters, distances);
        }
    }

    private void ReassignEmptyClusters(Matrix<T> batch, Matrix<T> centers, int[] assignments, Vector<T> distances)
    {
        // The point farthest from its assigned center (first one on ties; point 0 when every distance is zero)
        int farthestIdx = distances.Length > 0 ? ClusteringTensorOps.ArgMax(distances) : 0;
        for (int k = 0; k < centers.Rows && k < _centerCounts.Length; k++)
        {
            if (_centerCounts[k] == 0)
            {
                // Reassign this center to the farthest point
                Engine.SetRow(centers, k, batch.GetRow(farthestIdx));
                _centerCounts[k] = 1;
            }
        }
    }

    private Matrix<T> SampleBatch(Matrix<T> x, int batchSize)
    {
        batchSize = Math.Min(batchSize, x.Rows);

        // Fisher-Yates shuffle for random sampling without replacement
        var indices = new int[x.Rows];
        for (int i = 0; i < x.Rows; i++)
        {
            indices[i] = i;
        }

        for (int i = 0; i < batchSize; i++)
        {
            int j = _random.Next(i, x.Rows);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        return ClusteringTensorOps.SelectRows(x, indices.Take(batchSize).ToArray());
    }

    private Matrix<T> InitializeCenters(Matrix<T> x)
    {
        return _options.InitMethod switch
        {
            KMeansInitMethod.Custom when _options.InitialCenters is not null => _options.InitialCenters,
            KMeansInitMethod.KMeansPlusPlus => InitializeKMeansPlusPlus(x),
            _ => InitializeRandom(x)
        };
    }

    private Matrix<T> InitializeRandom(Matrix<T> x)
    {
        var selectedIndices = new HashSet<int>();
        var order = new int[_options.NumClusters];

        for (int k = 0; k < _options.NumClusters; k++)
        {
            int idx;
            do
            {
                idx = _random.Next(x.Rows);
            } while (selectedIndices.Contains(idx));

            selectedIndices.Add(idx);
            order[k] = idx;
        }

        return ClusteringTensorOps.SelectRows(x, order);
    }

    private Matrix<T> InitializeKMeansPlusPlus(Matrix<T> x)
    {
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var chosen = new int[_options.NumClusters];

        // Choose first center randomly
        chosen[0] = _random.Next(x.Rows);

        // Choose remaining centers with probability proportional to the squared distance to the nearest chosen one
        Vector<T>? minDistances = null;
        for (int k = 1; k < _options.NumClusters; k++)
        {
            var distances = ClusteringTensorOps.DistancesToPoint(distanceMetric, x, x.GetRow(chosen[k - 1]));
            var distSquared = Engine.Multiply(distances, distances);
            minDistances = minDistances is null ? distSquared : Engine.Min(minDistances, distSquared);
            chosen[k] = ClusteringTensorOps.SampleProportional(minDistances, _random);
        }

        return ClusteringTensorOps.SelectRows(x, chosen);
    }

    private Vector<T> AssignLabels(Matrix<T> x, Matrix<T> centers)
    {
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        return ClusteringTensorOps.FromIntArray<T>(ClusteringTensorOps.NearestCenters(distanceMetric, x, centers));
    }

    private new T ComputeInertia(Matrix<T> x, Vector<T> labels, Matrix<T> centers)
    {
        // Squared distance of every sample to its assigned center (labels outside the centers are skipped)
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var distances = ClusteringTensorOps.AssignedDistances(distanceMetric, x, labels, centers);
        return Engine.DotProduct(distances, distances);
    }

    private Matrix<T> ComputeDistancesToCenters(Matrix<T> x, Matrix<T> centers)
    {
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        return distanceMetric.ComputePairwise(x, centers);
    }

    private void ValidateInputData(Matrix<T> x)
    {
        if (x.Rows < _options.NumClusters)
        {
            throw new ArgumentException(
                $"Number of samples ({x.Rows}) must be at least the number of clusters ({_options.NumClusters}).");
        }
    }

    private void ValidatePredictInput(Matrix<T> x)
    {
        if (ClusterCenters is not null && x.Columns != ClusterCenters.Columns)
        {
            throw new ArgumentException(
                $"Input columns ({x.Columns}) must match training data columns ({ClusterCenters.Columns}).");
        }
    }
}
