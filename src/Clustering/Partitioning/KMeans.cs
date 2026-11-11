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

        // Run multiple initializations and keep the best (the first one with the lowest inertia)
        int numInits = _options.InitMethod == KMeansInitMethod.Custom ? 1 : _options.NumInitializations;
        var runs = new List<(Matrix<T> Centers, Vector<T> Labels, T Inertia, int Iterations)>(numInits);
        for (int init = 0; init < numInits; init++)
        {
            runs.Add(FitSingle(x));
        }

        Matrix<T>? bestCenters = null;
        Vector<T>? bestLabels = null;
        T bestInertia = NumOps.MaxValue;
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
        IsTrained = true;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        if (ClusterCenters is null) return new Vector<T>(x.Rows);
        return AssignLabels(x, ClusterCenters);
    }

    /// <inheritdoc />
    public override Matrix<T> Transform(Matrix<T> x)
    {
        ValidateIsTrained();
        ValidatePredictInput(x);

        if (ClusterCenters is null) return new Matrix<T>(x.Rows, NumClusters);
        return ComputeDistancesToCenters(x, ClusterCenters);
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        return Labels ?? new Vector<T>(0);
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
            bool converged = HasConverged(centers, newCenters);
            centers = newCenters;
            labels = newLabels;
            inertia = newInertia;
            if (converged)
            {
                break;
            }
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
            // Update minimum distances to already selected centers
            var distances = ClusteringTensorOps.DistancesToPoint(distanceMetric, x, x.GetRow(chosen[k - 1]));
            var distSquared = Engine.Multiply(distances, distances);
            minDistances = minDistances is null ? distSquared : Engine.Min(minDistances, distSquared);

            // Select next center with probability proportional to distance squared
            chosen[k] = ClusteringTensorOps.SampleProportional(minDistances, _random);
        }

        return ClusteringTensorOps.SelectRows(x, chosen);
    }

    private Vector<T> AssignLabels(Matrix<T> x, Matrix<T> centers)
    {
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        return ClusteringTensorOps.FromIntArray<T>(ClusteringTensorOps.NearestCenters(distanceMetric, x, centers));
    }

    private (Matrix<T> Centers, T Inertia) UpdateCenters(Matrix<T> x, Vector<T> labels, Matrix<T> oldCenters)
    {
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();

        // Mean of each cluster; an empty cluster keeps its old center
        var means = ClusteringTensorOps.ClusterMeans(x, labels, _options.NumClusters, out var counts);
        var newCenters = ClusteringTensorOps.RowsWherePositive(counts, means, oldCenters);

        // Compute inertia (sum of squared distances to assigned centers)
        var distances = ClusteringTensorOps.AssignedDistances(distanceMetric, x, labels, newCenters);
        T totalInertia = Engine.DotProduct(distances, distances);

        return (newCenters, totalInertia);
    }

    private bool HasConverged(Matrix<T> oldCenters, Matrix<T> newCenters)
        => NumOps.ToDouble(ComputeCenterShift(oldCenters, newCenters)) < _options.Tolerance;

    private T ComputeCenterShift(Matrix<T> oldCenters, Matrix<T> newCenters)
    {
        // Largest distance any center moved (zero when there are no centers)
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        return ClusteringTensorOps.MaxOrZero(ClusteringTensorOps.RowwiseDistances(distanceMetric, oldCenters, newCenters));
    }

    private Matrix<T> ComputeDistancesToCenters(Matrix<T> x, Matrix<T> centers)
    {
        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var distances = distanceMetric.ComputePairwise(x, centers);
        if (centers.Rows >= _options.NumClusters)
        {
            return distances;
        }

        // Fewer centers than configured clusters (after merging): the remaining columns stay zero
        var padded = new Matrix<T>(x.Rows, _options.NumClusters);
        padded.SetSubMatrix(0, 0, distances);
        return padded;
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

        if (ClusteringTensorOps.ContainsNonFinite(x))
        {
            throw new ArgumentException("Input data contains NaN or Infinity values.", nameof(x));
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
