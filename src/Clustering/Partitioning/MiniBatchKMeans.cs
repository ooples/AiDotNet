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
public class MiniBatchKMeans<T> : ClusteringBase<T>
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
        : base(options ?? new MiniBatchKMeansOptions<T>())
    {
        _options = options ?? new MiniBatchKMeansOptions<T>();
        _random = _options.RandomState.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomState.Value)
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
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new MiniBatchKMeans<T>(new MiniBatchKMeansOptions<T>
        {
            NumClusters = _options.NumClusters,
            MaxIterations = _options.MaxIterations,
            Tolerance = _options.Tolerance,
            RandomState = _options.RandomState,
            NumInitializations = _options.NumInitializations,
            BatchSize = _options.BatchSize,
            InitMethod = _options.InitMethod,
            MaxNoImprovement = _options.MaxNoImprovement,
            DistanceMetric = _options.DistanceMetric
        });
    }

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
        T bestInertia = NumOps.FromDouble(double.MaxValue);

        int numInits = _options.InitMethod == KMeansInitMethod.Custom ? 1 : _options.NumInitializations;

        for (int init = 0; init < numInits; init++)
        {
            var (centers, labels, inertia, iterations) = FitSingle(x);

            if (NumOps.ToDouble(inertia) < NumOps.ToDouble(bestInertia))
            {
                bestCenters = centers;
                bestLabels = labels;
                bestInertia = inertia;
                _numIterations = iterations;
            }
        }

        ClusterCenters = bestCenters;
        Labels = bestLabels;
        Inertia = bestInertia;
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

        return AssignLabels(x, ClusterCenters!);
    }

    /// <inheritdoc />
    public override Matrix<T> Transform(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        return ComputeDistancesToCenters(x, ClusterCenters!);
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels!;
    }

    private (Matrix<T> Centers, Vector<T> Labels, T Inertia, int Iterations) FitSingle(Matrix<T> x)
    {
        // Initialize centers
        int initSize = _options.InitSize ?? Math.Min(3 * _options.BatchSize, x.Rows);
        var initData = SampleBatch(x, initSize);
        var centers = InitializeCenters(initData);
        _centerCounts = new int[_options.NumClusters];

        T prevInertia = NumOps.FromDouble(double.MaxValue);
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
            double inertiaChange = NumOps.ToDouble(prevInertia) - NumOps.ToDouble(inertia);
            if (inertiaChange < _options.Tolerance)
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

    private void UpdateCentersWithBatch(Matrix<T> batch, Matrix<T>? centers = null)
    {
        centers ??= ClusterCenters!;
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Assign each point in the batch to its nearest center
        var nearestCenters = new int[batch.Rows];
        var distances = new T[batch.Rows];

        for (int i = 0; i < batch.Rows; i++)
        {
            var point = GetRow(batch, i);
            double minDist = double.MaxValue;
            int nearest = 0;

            for (int k = 0; k < _options.NumClusters; k++)
            {
                var center = GetRow(centers, k);
                T dist = distanceMetric.Compute(point, center);
                double distDouble = NumOps.ToDouble(dist);

                if (distDouble < minDist)
                {
                    minDist = distDouble;
                    nearest = k;
                }
            }

            nearestCenters[i] = nearest;
            distances[i] = NumOps.FromDouble(minDist);
        }

        // Update centers using streaming average
        for (int i = 0; i < batch.Rows; i++)
        {
            int k = nearestCenters[i];
            _centerCounts[k]++;
            double eta = 1.0 / _centerCounts[k]; // Learning rate decreases over time

            for (int j = 0; j < batch.Columns; j++)
            {
                // center[k] = center[k] + eta * (point[i] - center[k])
                T diff = NumOps.Subtract(batch[i, j], centers[k, j]);
                T update = NumOps.Multiply(NumOps.FromDouble(eta), diff);
                centers[k, j] = NumOps.Add(centers[k, j], update);
            }
        }

        // Handle empty clusters
        if (_options.ReassignEmptyClusters)
        {
            ReassignEmptyClusters(batch, centers, nearestCenters, distances);
        }
    }

    private void ReassignEmptyClusters(Matrix<T> batch, Matrix<T> centers, int[] assignments, T[] distances)
    {
        for (int k = 0; k < _options.NumClusters; k++)
        {
            if (_centerCounts[k] == 0)
            {
                // Find the point with maximum distance to its assigned center
                int farthestIdx = 0;
                double maxDist = 0;

                for (int i = 0; i < batch.Rows; i++)
                {
                    double dist = NumOps.ToDouble(distances[i]);
                    if (dist > maxDist)
                    {
                        maxDist = dist;
                        farthestIdx = i;
                    }
                }

                // Reassign this center to the farthest point
                for (int j = 0; j < batch.Columns; j++)
                {
                    centers[k, j] = batch[farthestIdx, j];
                }
                _centerCounts[k] = 1;
            }
        }
    }

    private Matrix<T> SampleBatch(Matrix<T> x, int batchSize)
    {
        batchSize = Math.Min(batchSize, x.Rows);
        var batch = new Matrix<T>(batchSize, x.Columns);

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

            for (int col = 0; col < x.Columns; col++)
            {
                batch[i, col] = x[indices[i], col];
            }
        }

        return batch;
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
        var centers = new Matrix<T>(_options.NumClusters, x.Columns);
        var selectedIndices = new HashSet<int>();

        for (int k = 0; k < _options.NumClusters; k++)
        {
            int idx;
            do
            {
                idx = _random.Next(x.Rows);
            } while (selectedIndices.Contains(idx));

            selectedIndices.Add(idx);

            for (int j = 0; j < x.Columns; j++)
            {
                centers[k, j] = x[idx, j];
            }
        }

        return centers;
    }

    private Matrix<T> InitializeKMeansPlusPlus(Matrix<T> x)
    {
        var centers = new Matrix<T>(_options.NumClusters, x.Columns);
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        int firstIdx = _random.Next(x.Rows);
        for (int j = 0; j < x.Columns; j++)
        {
            centers[0, j] = x[firstIdx, j];
        }

        var minDistances = new double[x.Rows];
        for (int i = 0; i < x.Rows; i++)
        {
            minDistances[i] = double.MaxValue;
        }

        for (int k = 1; k < _options.NumClusters; k++)
        {
            var lastCenter = GetRow(centers, k - 1);
            double totalWeight = 0;

            for (int i = 0; i < x.Rows; i++)
            {
                var point = GetRow(x, i);
                T dist = distanceMetric.Compute(point, lastCenter);
                double distSquared = Math.Pow(NumOps.ToDouble(dist), 2);

                if (distSquared < minDistances[i])
                {
                    minDistances[i] = distSquared;
                }

                totalWeight += minDistances[i];
            }

            double threshold = _random.NextDouble() * totalWeight;
            double cumulative = 0;
            int selectedIdx = 0;

            for (int i = 0; i < x.Rows; i++)
            {
                cumulative += minDistances[i];
                if (cumulative >= threshold)
                {
                    selectedIdx = i;
                    break;
                }
            }

            for (int j = 0; j < x.Columns; j++)
            {
                centers[k, j] = x[selectedIdx, j];
            }
        }

        return centers;
    }

    private Vector<T> AssignLabels(Matrix<T> x, Matrix<T> centers)
    {
        var labels = new Vector<T>(x.Rows);
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        for (int i = 0; i < x.Rows; i++)
        {
            var point = GetRow(x, i);
            double minDist = double.MaxValue;
            int nearestCluster = 0;

            for (int k = 0; k < _options.NumClusters; k++)
            {
                var center = GetRow(centers, k);
                T dist = distanceMetric.Compute(point, center);
                double distDouble = NumOps.ToDouble(dist);

                if (distDouble < minDist)
                {
                    minDist = distDouble;
                    nearestCluster = k;
                }
            }

            labels[i] = NumOps.FromDouble(nearestCluster);
        }

        return labels;
    }

    private new T ComputeInertia(Matrix<T> x, Vector<T> labels, Matrix<T> centers)
    {
        T totalInertia = NumOps.Zero;
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        for (int i = 0; i < x.Rows; i++)
        {
            int cluster = (int)NumOps.ToDouble(labels[i]);
            var point = GetRow(x, i);
            var center = GetRow(centers, cluster);
            T dist = distanceMetric.Compute(point, center);
            totalInertia = NumOps.Add(totalInertia, NumOps.Multiply(dist, dist));
        }

        return totalInertia;
    }

    private Matrix<T> ComputeDistancesToCenters(Matrix<T> x, Matrix<T> centers)
    {
        var distances = new Matrix<T>(x.Rows, _options.NumClusters);
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        for (int i = 0; i < x.Rows; i++)
        {
            var point = GetRow(x, i);
            for (int k = 0; k < _options.NumClusters; k++)
            {
                var center = GetRow(centers, k);
                distances[i, k] = distanceMetric.Compute(point, center);
            }
        }

        return distances;
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
