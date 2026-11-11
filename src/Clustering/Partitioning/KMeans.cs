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

        // The data, its row norms and the convergence threshold serve every initialization.
        var data = new KMeansData(x, UsesEuclideanDistance, AbsoluteTolerance(x));

        // Run multiple initializations and keep the best (the first one with the lowest inertia)
        int numInits = _options.InitMethod == KMeansInitMethod.Custom ? 1 : _options.NumInitializations;
        var runs = new List<(Matrix<T> Centers, Vector<T> Labels, T Inertia, int Iterations)>(numInits);
        for (int init = 0; init < numInits; init++)
        {
            runs.Add(FitSingle(data));
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
        return AssignLabels(new KMeansData(x, UsesEuclideanDistance, NumOps.Zero), ClusterCenters);
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

    /// <summary>
    /// True when distances are plain Euclidean, so the assignment can use the batched
    /// <c>||x||^2 - 2 x.c + ||c||^2</c> form. A custom metric, including a subclass of
    /// <see cref="EuclideanDistance{T}"/>, goes through the metric itself.
    /// </summary>
    private bool UsesEuclideanDistance =>
        _options.DistanceMetric is null || _options.DistanceMetric.GetType() == typeof(EuclideanDistance<T>);

    /// <summary>
    /// The data of one fit or prediction: the matrix, its rank-2 tensor, each row's squared norm (Euclidean only) and the
    /// absolute convergence threshold.
    /// </summary>
    private sealed class KMeansData
    {
        public KMeansData(Matrix<T> x, bool euclidean, T tolerance)
        {
            X = x;
            Values = ClusteringTensorOps.ToTensor(x);
            Euclidean = euclidean;
            Tolerance = tolerance;
            if (euclidean && x.Rows > 0)
            {
                var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
                RowNorms = engine.ReduceSum(engine.TensorSquare(Values), new[] { 1 }, true);
            }
        }

        public Matrix<T> X { get; }

        /// <summary>The data as a [rows, columns] tensor.</summary>
        public Tensor<T> Values { get; }

        public bool Euclidean { get; }

        public T Tolerance { get; }

        /// <summary>Each row's squared norm as [rows, 1]; null unless Euclidean.</summary>
        public Tensor<T>? RowNorms { get; }
    }

    private (Matrix<T> Centers, Vector<T> Labels, T Inertia, int Iterations) FitSingle(KMeansData data)
    {
        var x = data.X;
        var centers = InitializeCenters(data);

        Vector<T> labels = new Vector<T>(x.Rows);
        Vector<T>? previousLabels = null;
        bool unchanged = false;
        int iterations = 0;
        while (iterations < _options.MaxIterations)
        {
            // Assignment step: assign each point to nearest center
            labels = AssignLabels(data, centers);
            iterations++;

            // Update step: compute new centers, and whether the sum of the centers' squared shifts is within tolerance
            var newCenters = UpdateCenters(x, labels, centers);
            bool shiftWithinTolerance = HasConverged(data, centers, newCenters);

            centers = newCenters;
            if (previousLabels is not null && Engine.TensorEqual(labels.AsTensor(), previousLabels.AsTensor()))
            {
                unchanged = true;
                break;
            }

            if (shiftWithinTolerance) break;
            previousLabels = labels;
        }

        // Label every row by the centers returned, unless the last assignment already did.
        if (iterations > 0 && !unchanged)
            labels = AssignLabels(data, centers);

        T inertia = iterations > 0 ? ComputeInertia(data, labels, centers) : NumOps.Zero;
        return (centers, labels, inertia, iterations);
    }

    /// <summary>
    /// The convergence threshold on the summed squared center shift: the tolerance times the mean over features of
    /// each feature's (population) variance, as scikit-learn's KMeans.
    /// </summary>
    private T AbsoluteTolerance(Matrix<T> x)
    {
        if (x.Rows == 0 || x.Columns == 0) return NumOps.Zero;

        // Mean over all entries of the squared deviation from the column mean = mean of the per-feature variances.
        var deviations = ClusteringTensorOps.SubtractRow(x, ClusteringTensorOps.ColumnMeans(x));
        T meanVariance = Engine.TensorMean(Engine.TensorSquare(deviations));
        return NumOps.Multiply(NumOps.FromDouble(_options.Tolerance), meanVariance);
    }

    private Matrix<T> InitializeCenters(KMeansData data)
    {
        return _options.InitMethod switch
        {
            KMeansInitMethod.Custom when _options.InitialCenters is not null => _options.InitialCenters,
            KMeansInitMethod.KMeansPlusPlus => InitializeKMeansPlusPlus(data),
            _ => InitializeRandom(data.X)
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

    /// <summary>
    /// Candidates per k-means++ center: <see cref="KMeansOptions{T}.KMeansPlusPlusLocalTrials"/>, or scikit-learn's
    /// <c>2 + floor(ln k)</c>.
    /// </summary>
    private int LocalTrials(int clusters)
    {
        if (_options.KMeansPlusPlusLocalTrials is int trials)
        {
            if (trials < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(KMeansOptions<T>.KMeansPlusPlusLocalTrials), trials,
                    "k-means++ needs at least one candidate per center.");
            }

            return trials;
        }

        return 2 + (int)Math.Log(clusters);
    }

    /// <summary>
    /// Greedy k-means++ (Arthur and Vassilvitskii, 2007; scikit-learn's default): each new center is the best of
    /// <see cref="LocalTrials"/> rows drawn with probability proportional to the squared distance to the nearest chosen
    /// center, the one whose potential (the total of those squared distances once it is added) is lowest, the first drawn
    /// on a tie. One trial is classic k-means++, which takes the drawn row.
    /// </summary>
    private Matrix<T> InitializeKMeansPlusPlus(KMeansData data)
    {
        var x = data.X;
        int rows = x.Rows;
        var chosen = new int[_options.NumClusters];

        // Choose first center randomly
        chosen[0] = _random.Next(rows);

        int trials = LocalTrials(_options.NumClusters);
        Vector<T>? minDistances = null;
        for (int k = 1; k < _options.NumClusters; k++)
        {
            // The first center sets the minimum squared distances; each later pick lowers them below
            if (minDistances is null)
            {
                minDistances = Engine.Reshape(
                    SquaredDistancesTo(data, ClusteringTensorOps.SelectRows(x, new[] { chosen[k - 1] })), new[] { rows }).ToVector();
            }

            // Select next center with probability proportional to distance squared, keeping the best of the trials
            var candidates = ClusteringTensorOps.SampleProportionalMany(minDistances, _random, trials);
            if (trials == 1)
            {
                chosen[k] = candidates[0];
                var toChosen = SquaredDistancesTo(data, ClusteringTensorOps.SelectRows(x, candidates));
                minDistances = Engine.TensorMin(minDistances.AsTensor(), Engine.Reshape(toChosen, new[] { rows })).ToVector();
                continue;
            }

            // [rows, trials]: every row's minimum squared distance once each candidate is a center
            var lowered = Engine.TensorMin(
                Engine.TensorBroadcastTo(Tensor<T>.FromVector(minDistances, new[] { rows, 1 }), new[] { rows, trials }),
                SquaredDistancesTo(data, ClusteringTensorOps.SelectRows(x, candidates)));
            var potentials = Engine.ReduceSum(lowered, new[] { 0 }, false).ToVector();
            int best = ClusteringTensorOps.ArgMin(potentials);
            chosen[k] = candidates[best];
            minDistances = Engine.Reshape(Engine.TensorSlice(lowered, new[] { 0, best }, new[] { rows, 1 }), new[] { rows }).ToVector();
        }

        return ClusteringTensorOps.SelectRows(x, chosen);
    }

    /// <summary>
    /// Squared distance from every row of the data to every row of <paramref name="points"/>, as [rows, points]:
    /// computed directly for Euclidean distance, as the square of the configured metric otherwise.
    /// </summary>
    private Tensor<T> SquaredDistancesTo(KMeansData data, Matrix<T> points)
    {
        if (data.Euclidean)
        {
            return ClusteringTensorOps.ToTensor(ClusteringTensorOps.SquaredDistances(data.X, points));
        }

        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        return Engine.TensorSquare(ClusteringTensorOps.ToTensor(distanceMetric.ComputePairwise(data.X, points)));
    }

    /// <summary>
    /// Nearest-center label (the lowest index on a tie) for every row. Euclidean distance ranks the centers by
    /// <c>||x||^2 + ||c||^2 - 2 x.c</c>, one matrix product for all rows and centers; any other metric by its own
    /// pairwise distances.
    /// </summary>
    private Vector<T> AssignLabels(KMeansData data, Matrix<T> centers)
    {
        var x = data.X;
        if (!data.Euclidean || data.RowNorms is null || centers.Rows == 0)
        {
            var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
            return ClusteringTensorOps.FromIntArray<T>(ClusteringTensorOps.NearestCenters(distanceMetric, x, centers));
        }

        int rows = x.Rows;
        int clusters = centers.Rows;
        var ct = ClusteringTensorOps.ToTensor(centers);
        var dots = Engine.TensorMatMul(data.Values, Engine.TensorTranspose(ct));
        var centerNorms = Engine.Reshape(Engine.ReduceSum(Engine.TensorSquare(ct), new[] { 1 }, false), new[] { 1, clusters });
        var norms = Engine.TensorAdd(
            Engine.TensorBroadcastTo(data.RowNorms, new[] { rows, clusters }),
            Engine.TensorBroadcastTo(centerNorms, new[] { rows, clusters }));
        var scores = Engine.TensorSubtract(norms, Engine.TensorMultiplyScalar(dots, NumOps.FromDouble(2.0)));
        return ClusteringTensorOps.FromIntArray<T>(Engine.TensorArgMin(scores, 1).ToArray());
    }

    private Matrix<T> UpdateCenters(Matrix<T> x, Vector<T> labels, Matrix<T> oldCenters)
    {
        // Mean of each cluster; an empty cluster keeps its old center
        var means = ClusteringTensorOps.ClusterMeans(x, labels, _options.NumClusters, out var counts);
        return ClusteringTensorOps.RowsWherePositive(counts, means, oldCenters);
    }

    /// <summary>True when the sum over clusters of each center's squared shift is at most the absolute tolerance.</summary>
    private bool HasConverged(KMeansData data, Matrix<T> oldCenters, Matrix<T> newCenters)
        => NumOps.LessThanOrEquals(ComputeCenterShift(data, oldCenters, newCenters), data.Tolerance);

    /// <summary>The sum over clusters of each center's squared shift.</summary>
    private T ComputeCenterShift(KMeansData data, Matrix<T> oldCenters, Matrix<T> newCenters)
    {
        if (oldCenters.Rows == 0) return NumOps.Zero;

        if (data.Euclidean)
        {
            return Engine.TensorSum(ClusteringTensorOps.RowSquaredEuclidean(
                ClusteringTensorOps.ToTensor(oldCenters), ClusteringTensorOps.ToTensor(newCenters)));
        }

        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var shifts = ClusteringTensorOps.RowwiseDistances(distanceMetric, oldCenters, newCenters);
        return Engine.DotProduct(shifts, shifts);
    }

    /// <summary>Sum of squared distances from every row to the center its label names.</summary>
    private T ComputeInertia(KMeansData data, Vector<T> labels, Matrix<T> centers)
    {
        if (data.Euclidean)
        {
            return ClusteringTensorOps.Inertia(data.X, labels, centers);
        }

        var distanceMetric = _options.DistanceMetric ?? new EuclideanDistance<T>();
        var distances = ClusteringTensorOps.AssignedDistances(distanceMetric, data.X, labels, centers);
        return Engine.DotProduct(distances, distances);
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
