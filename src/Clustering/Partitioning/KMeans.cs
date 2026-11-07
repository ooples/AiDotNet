using AiDotNet.Attributes;
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
/// <para>
/// Convergence follows scikit-learn's Lloyd iteration: it stops when an assignment leaves every label unchanged, or
/// when the sum over clusters of each center's squared shift is at most
/// <see cref="Clustering.Options.ClusteringOptions{T}.Tolerance"/> times the mean per-feature variance of the data (so
/// the default 1e-4 means what it means in scikit-learn). Unless the labels were already unchanged, a final assignment
/// makes the labels those of the returned centers. <see cref="NumIterations"/> counts the update steps run.
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
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new KMeans&lt;double&gt;(new KMeansOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // one cluster index per row
/// var assignments = result.Predict(dataMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Some methods for classification and analysis of multivariate observations", "https://projecteuclid.org/proceedings/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Fifth-Berkeley-Symposium-on-Mathematical-Statistics-and/Chapter/0/bsmsp/1200512992", Year = 1967, Authors = "James MacQueen")]
public partial class KMeans<T> : ClusteringBase<T>
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
        : base(options ??= new KMeansOptions<T>())
    {
        _options = options;
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
        NumFeatures = x.Columns;

        Matrix<T>? bestCenters = null;
        Vector<T>? bestLabels = null;
        T bestInertia = NumOps.MaxValue;

        // Run multiple initializations and keep the best
        int numInits = _options.InitMethod == KMeansInitMethod.Custom ? 1 : _options.NumInitializations;

        // One double copy of the data and its row norms serves every initialization.
        var kernel = UsesEuclideanDistance ? CreateKernel(x) : null;

        for (int init = 0; init < numInits; init++)
        {
            var (centers, labels, inertia, iterations) = kernel is null ? FitSingle(x) : FitSingleEuclidean(x, kernel);

            if (NumOps.LessThan(inertia, bestInertia))
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
        MergeDegenerateClusters(x);
        IsTrained = true;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        if (ClusterCenters is null) return new Vector<T>(x.Rows);
        if (UsesEuclideanDistance)
        {
            var kernel = CreateKernel(x);
            var labels = new int[x.Rows];
            kernel.Assign(ToDoubles(ClusterCenters), ClusterCenters.Rows, labels, null, null);
            var result = new Vector<T>(x.Rows);
            for (int i = 0; i < labels.Length; i++) result[i] = NumOps.FromDouble(labels[i]);
            return result;
        }

        return AssignLabels(x, ClusterCenters);
    }

    /// <inheritdoc />
    public override Matrix<T> Transform(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        if (ClusterCenters is null) return new Matrix<T>(x.Rows, NumClusters);
        if (UsesEuclideanDistance)
        {
            int clusters = ClusterCenters.Rows;
            var distances = CreateKernel(x).Distances(ToDoubles(ClusterCenters), clusters);
            var result = new Matrix<T>(x.Rows, clusters);
            for (int i = 0; i < x.Rows; i++)
                for (int k = 0; k < clusters; k++)
                    result[i, k] = NumOps.FromDouble(distances[i * clusters + k]);
            return result;
        }

        return ComputeDistancesToCenters(x, ClusterCenters);
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
    }

    /// <summary>
    /// True when distances are plain Euclidean, which <see cref="EuclideanKMeansKernel"/> computes directly. A custom
    /// metric, including a subclass of <see cref="EuclideanDistance{T}"/>, keeps the generic path.
    /// </summary>
    private bool UsesEuclideanDistance =>
        _options.DistanceMetric is null || _options.DistanceMetric.GetType() == typeof(EuclideanDistance<T>);

    private EuclideanKMeansKernel CreateKernel(Matrix<T> x) => new(ToDoubles(x), x.Rows, x.Columns);

    private double[] ToDoubles(Matrix<T> matrix)
    {
        var values = new double[matrix.Rows * matrix.Columns];
        for (int i = 0; i < matrix.Rows; i++)
        {
            var row = matrix.GetRowReadOnlySpan(i);
            int offset = i * matrix.Columns;
            for (int j = 0; j < row.Length; j++) values[offset + j] = NumOps.ToDouble(row[j]);
        }

        return values;
    }

    /// <summary>
    /// <see cref="FitSingle"/> for Euclidean distance, on <paramref name="kernel"/>: the same initialization, the same
    /// Lloyd iterations, empty-cluster handling and convergence test.
    /// </summary>
    private (Matrix<T> Centers, Vector<T> Labels, T Inertia, int Iterations) FitSingleEuclidean(
        Matrix<T> x, EuclideanKMeansKernel kernel)
    {
        int clusters = _options.NumClusters;
        int columns = kernel.Columns;
        var centers = _options.InitMethod switch
        {
            KMeansInitMethod.Custom when _options.InitialCenters is not null => ToDoubles(_options.InitialCenters),
            KMeansInitMethod.KMeansPlusPlus => InitializeKMeansPlusPlusEuclidean(kernel, clusters),
            _ => ToDoubles(InitializeRandom(x))
        };

        var labels = new int[kernel.Rows];
        var previousLabels = new int[kernel.Rows];
        for (int i = 0; i < previousLabels.Length; i++) previousLabels[i] = -1;
        var sums = new double[clusters * columns];
        var counts = new int[clusters];
        double tolerance = AbsoluteTolerance(kernel.Data, kernel.Rows, columns);
        bool unchanged = false;
        int iterations = 0;
        while (iterations < _options.MaxIterations)
        {
            kernel.Assign(centers, clusters, labels, sums, counts);
            iterations++;

            var newCenters = new double[clusters * columns];
            double shift = 0;
            for (int k = 0; k < clusters; k++)
            {
                int offset = k * columns;
                if (counts[k] > 0)
                {
                    for (int j = 0; j < columns; j++) newCenters[offset + j] = sums[offset + j] / counts[k];
                }
                else
                {
                    // Empty cluster: keep its old center.
                    Array.Copy(centers, offset, newCenters, offset, columns);
                }

                shift += EuclideanKMeansKernel.SquaredDistance(centers, offset, newCenters, offset, columns);
            }

            centers = newCenters;
            if (labels.AsSpan().SequenceEqual(previousLabels))
            {
                unchanged = true;
                break;
            }

            if (shift <= tolerance) break;
            Array.Copy(labels, previousLabels, labels.Length);
        }

        // Label every row by the centers returned, unless the last assignment already did.
        if (iterations > 0 && !unchanged)
            kernel.Assign(centers, clusters, labels, null, null);

        double inertia = iterations > 0 ? kernel.Inertia(centers, labels) : 0;

        var centerMatrix = new Matrix<T>(clusters, columns);
        for (int k = 0; k < clusters; k++)
            for (int j = 0; j < columns; j++)
                centerMatrix[k, j] = NumOps.FromDouble(centers[k * columns + j]);
        var labelVector = new Vector<T>(labels.Length);
        for (int i = 0; i < labels.Length; i++) labelVector[i] = NumOps.FromDouble(labels[i]);
        return (centerMatrix, labelVector, NumOps.FromDouble(inertia), iterations);
    }

    /// <summary>
    /// k-means++ seeding on <paramref name="kernel"/>, drawing from the random stream exactly as
    /// <see cref="InitializeKMeansPlusPlus"/> does.
    /// </summary>
    private double[] InitializeKMeansPlusPlusEuclidean(EuclideanKMeansKernel kernel, int clusters)
    {
        int rows = kernel.Rows;
        int columns = kernel.Columns;
        var data = kernel.Data;
        var centers = new double[clusters * columns];

        int firstIdx = _random.Next(rows);
        Array.Copy(data, firstIdx * columns, centers, 0, columns);

        var minDistances = new double[rows];
        for (int i = 0; i < rows; i++) minDistances[i] = double.MaxValue;

        for (int k = 1; k < clusters; k++)
        {
            kernel.UpdateMinSquaredDistances(centers, k - 1, minDistances);
            double totalWeight = 0;
            for (int i = 0; i < rows; i++) totalWeight += minDistances[i];

            double threshold = _random.NextDouble() * totalWeight;
            double cumulative = 0;
            int selectedIdx = 0;
            for (int i = 0; i < rows; i++)
            {
                cumulative += minDistances[i];
                if (cumulative >= threshold)
                {
                    selectedIdx = i;
                    break;
                }
            }

            Array.Copy(data, selectedIdx * columns, centers, k * columns, columns);
        }

        return centers;
    }

    private (Matrix<T> Centers, Vector<T> Labels, T Inertia, int Iterations) FitSingle(Matrix<T> x)
    {
        var centers = InitializeCenters(x);
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        double tolerance = AbsoluteTolerance(ToDoubles(x), x.Rows, x.Columns);

        Vector<T> labels = new Vector<T>(x.Rows);
        Vector<T>? previousLabels = null;
        bool unchanged = false;
        int iterations = 0;
        while (iterations < _options.MaxIterations)
        {
            // Assignment step: assign each point to nearest center
            labels = AssignLabels(x, centers);
            iterations++;

            // Update step: compute new centers, and the sum of the centers' squared shifts
            var (newCenters, _) = UpdateCenters(x, labels, centers);
            double shift = 0;
            for (int k = 0; k < _options.NumClusters; k++)
            {
                double distance = NumOps.ToDouble(distanceMetric.Compute(GetRow(centers, k), GetRow(newCenters, k)));
                shift += distance * distance;
            }

            centers = newCenters;
            if (previousLabels is not null && SameLabels(labels, previousLabels))
            {
                unchanged = true;
                break;
            }

            if (shift <= tolerance) break;
            previousLabels = labels;
        }

        // Label every row by the centers returned, unless the last assignment already did.
        if (iterations > 0 && !unchanged)
            labels = AssignLabels(x, centers);

        T inertia = NumOps.Zero;
        if (iterations > 0)
        {
            for (int i = 0; i < x.Rows; i++)
            {
                T distance = distanceMetric.Compute(GetRow(x, i), GetRow(centers, (int)NumOps.ToDouble(labels[i])));
                inertia = NumOps.Add(inertia, NumOps.Multiply(distance, distance));
            }
        }

        return (centers, labels, inertia, iterations);
    }

    private bool SameLabels(Vector<T> a, Vector<T> b)
    {
        for (int i = 0; i < a.Length; i++)
            if (NumOps.Compare(a[i], b[i]) != 0) return false;
        return true;
    }

    /// <summary>
    /// The convergence threshold on the summed squared center shift: the tolerance times the mean over features of
    /// each feature's (population) variance, as scikit-learn's KMeans.
    /// </summary>
    private double AbsoluteTolerance(double[] rowMajor, int rows, int columns)
    {
        if (rows == 0 || columns == 0) return 0;

        double totalVariance = 0;
        for (int j = 0; j < columns; j++)
        {
            double mean = 0;
            for (int i = 0; i < rows; i++) mean += rowMajor[i * columns + j];
            mean /= rows;

            double variance = 0;
            for (int i = 0; i < rows; i++)
            {
                double d = rowMajor[i * columns + j] - mean;
                variance += d * d;
            }

            totalVariance += variance / rows;
        }

        return _options.Tolerance * (totalVariance / columns);
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

        // The centers are fixed for the whole pass; copying each one out again for every point allocated
        // rows x clusters vectors per iteration.
        var centerRows = new Vector<T>[centers.Rows];
        for (int k = 0; k < centers.Rows; k++)
            centerRows[k] = GetRow(centers, k);

        for (int i = 0; i < x.Rows; i++)
        {
            var point = GetRow(x, i);
            T minDist = NumOps.MaxValue;
            int nearestCluster = 0;

            for (int k = 0; k < centers.Rows; k++)
            {
                var center = centerRows[k];
                T dist = distanceMetric.Compute(point, center);

                if (NumOps.LessThan(dist, minDist))
                {
                    minDist = dist;
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

    private Matrix<T> ComputeDistancesToCenters(Matrix<T> x, Matrix<T> centers)
    {
        var distances = new Matrix<T>(x.Rows, _options.NumClusters);
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        for (int i = 0; i < x.Rows; i++)
        {
            var point = GetRow(x, i);
            for (int k = 0; k < centers.Rows; k++)
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
            foreach (T value in x.GetRowReadOnlySpan(i))
            {
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
