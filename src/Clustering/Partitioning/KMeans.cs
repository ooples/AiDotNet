using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.Partitioning;

/// <summary>
/// K-Means clustering algorithm implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// K-Means is one of the most widely used clustering algorithms. It partitions
/// n observations into k clusters by minimizing within-cluster variance (inertia).
/// </para>
/// <para>
/// Algorithm steps:
/// 1. Initialize k cluster centers (randomly or using k-means++)
/// 2. Assign each point to the nearest center
/// 3. Update each center as the mean of its assigned points
/// 4. Repeat steps 2-3 until convergence or max iterations
/// </para>
/// <para><b>For Beginners:</b> K-Means finds k groups in your data by:
/// - Starting with k "center" points
/// - Grouping data points by their closest center
/// - Moving each center to the middle of its group
/// - Repeating until centers stop moving
///
/// It works best when:
/// - You know how many clusters to look for
/// - Clusters are roughly spherical and similar in size
/// - Data doesn't have many outliers
/// </para>
/// </remarks>
public class KMeans<T> : ClusteringBase<T>
{
    private readonly KMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Random _random;
    private int _numIterations;

    /// <summary>
    /// Initializes a new KMeans instance with the specified options.
    /// </summary>
    /// <param name="options">The KMeans configuration options.</param>
    public KMeans(KMeansOptions<T>? options = null)
        : base(options ?? new KMeansOptions<T>())
    {
        _options = options ?? new KMeansOptions<T>();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Set distance metric if not specified
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
        return new KMeans<T>(new KMeansOptions<T>
        {
            NumClusters = _options.NumClusters,
            MaxIterations = _options.MaxIterations,
            Tolerance = _options.Tolerance,
            Seed = _options.Seed,
            NumInitializations = _options.NumInitializations,
            InitMethod = _options.InitMethod,
            DistanceMetric = _options.DistanceMetric
        });
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (KMeans<T>)CreateNewInstance();
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

        // Run multiple initializations and keep the best
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
        var centers = InitializeCenters(x);

        Vector<T> labels = new Vector<T>(x.Rows);
        T inertia = NumOps.Zero;
        int iterations = 0;

        for (iterations = 0; iterations < _options.MaxIterations; iterations++)
        {
            // Assignment step: assign each point to nearest center
            var newLabels = AssignLabels(x, centers);

            // Update step: compute new centers
            var (newCenters, newInertia) = UpdateCenters(x, newLabels, centers);

            // Check for convergence
            T centerShift = ComputeCenterShift(centers, newCenters);
            if (NumOps.ToDouble(centerShift) < _options.Tolerance)
            {
                centers = newCenters;
                labels = newLabels;
                inertia = newInertia;
                break;
            }

            centers = newCenters;
            labels = newLabels;
            inertia = newInertia;
        }

        return (centers, labels, inertia, iterations);
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

        // Choose first center randomly
        int firstIdx = _random.Next(x.Rows);
        for (int j = 0; j < x.Columns; j++)
        {
            centers[0, j] = x[firstIdx, j];
        }

        // Choose remaining centers with probability proportional to distance squared
        var minDistances = new double[x.Rows];
        for (int i = 0; i < x.Rows; i++)
        {
            minDistances[i] = double.MaxValue;
        }

        for (int k = 1; k < _options.NumClusters; k++)
        {
            // Update minimum distances to already selected centers
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

            // Select next center with probability proportional to distance squared
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

    private (Matrix<T> Centers, T Inertia) UpdateCenters(Matrix<T> x, Vector<T> labels, Matrix<T> oldCenters)
    {
        var newCenters = new Matrix<T>(_options.NumClusters, x.Columns);
        var counts = new int[_options.NumClusters];
        T totalInertia = NumOps.Zero;
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Sum points for each cluster
        for (int i = 0; i < x.Rows; i++)
        {
            int cluster = (int)NumOps.ToDouble(labels[i]);
            counts[cluster]++;

            for (int j = 0; j < x.Columns; j++)
            {
                newCenters[cluster, j] = NumOps.Add(newCenters[cluster, j], x[i, j]);
            }
        }

        // Compute mean and inertia
        for (int k = 0; k < _options.NumClusters; k++)
        {
            if (counts[k] > 0)
            {
                T countT = NumOps.FromDouble(counts[k]);
                for (int j = 0; j < x.Columns; j++)
                {
                    newCenters[k, j] = NumOps.Divide(newCenters[k, j], countT);
                }
            }
            else
            {
                // Empty cluster: reinitialize with random point or keep old center
                for (int j = 0; j < x.Columns; j++)
                {
                    newCenters[k, j] = oldCenters[k, j];
                }
            }
        }

        // Compute inertia (sum of squared distances to assigned centers)
        for (int i = 0; i < x.Rows; i++)
        {
            int cluster = (int)NumOps.ToDouble(labels[i]);
            var point = GetRow(x, i);
            var center = GetRow(newCenters, cluster);
            T dist = distanceMetric.Compute(point, center);
            totalInertia = NumOps.Add(totalInertia, NumOps.Multiply(dist, dist));
        }

        return (newCenters, totalInertia);
    }

    private T ComputeCenterShift(Matrix<T> oldCenters, Matrix<T> newCenters)
    {
        T maxShift = NumOps.Zero;
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        for (int k = 0; k < _options.NumClusters; k++)
        {
            var oldCenter = GetRow(oldCenters, k);
            var newCenter = GetRow(newCenters, k);
            T shift = distanceMetric.Compute(oldCenter, newCenter);

            if (NumOps.ToDouble(shift) > NumOps.ToDouble(maxShift))
            {
                maxShift = shift;
            }
        }

        return maxShift;
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

        if (_options.InitMethod == KMeansInitMethod.Custom && _options.InitialCenters is not null)
        {
            if (_options.InitialCenters.Rows != _options.NumClusters)
            {
                throw new ArgumentException(
                    $"Initial centers rows ({_options.InitialCenters.Rows}) must equal NumClusters ({_options.NumClusters}).");
            }
            if (_options.InitialCenters.Columns != x.Columns)
            {
                throw new ArgumentException(
                    $"Initial centers columns ({_options.InitialCenters.Columns}) must equal data columns ({x.Columns}).");
            }
        }

        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                T value = x[i, j];
                if (NumericalStabilityHelper.IsNaN(value) || NumericalStabilityHelper.IsInfinity(value))
                {
                    throw new ArgumentException("Input data contains NaN or Infinity values.", nameof(x));
                }
            }
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
