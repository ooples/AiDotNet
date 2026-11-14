using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.Streaming;

/// <summary>
/// Online K-Means clustering for streaming data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Online K-Means processes data points one at a time, updating cluster
/// centers incrementally. This is suitable for streaming data or when
/// memory is limited.
/// </para>
/// <para>
/// Algorithm:
/// 1. Initialize K cluster centers
/// 2. For each incoming point:
///    a. Find the nearest cluster center
///    b. Move that center slightly toward the point
///    c. Update cluster assignment
/// 3. Optionally decay learning rate over time
/// </para>
/// <para><b>For Beginners:</b> Online K-Means learns continuously.
///
/// Think of it like learning a new skill:
/// - Each example you see teaches you a little bit
/// - You don't need to remember every example
/// - Your understanding improves over time
///
/// The learning rate is like "how much you learn from each example":
/// - High: Quick learner, but might forget old lessons
/// - Low: Slow learner, but very stable knowledge
///
/// Use cases:
/// - Real-time sensor data
/// - Social media streams
/// - Financial market data
/// - Any data that arrives continuously
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var dataMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 8.0, 8.0 }, { 1.0, 0.6 }, { 9.0, 11.0 } });
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new OnlineKMeans&lt;double&gt;(new OnlineKMeansOptions&lt;double&gt;()))
///     .Build(dataMatrix);
///
/// // the cluster each training row was assigned to (Predict places rows the model has not seen)
/// var labels = result.GetClusterLabels();
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Mini-Batch K-Means Clustering", "https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf")]
public partial class OnlineKMeans<T> : ClusteringBase<T>
{
    private readonly OnlineKMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Matrix<T> _centers = new Matrix<T>(0, 0);
    private int[] _clusterCounts = Array.Empty<int>();
    private long _totalPointsSeen;

    /// <summary>
    /// Initializes a new Online K-Means instance.
    /// </summary>
    /// <param name="options">The Online K-Means options.</param>
    public OnlineKMeans(OnlineKMeansOptions<T>? options = null)
        : base(options ??= new OnlineKMeansOptions<T>())
    {
        _options = options;
    }

    /// <summary>
    /// Gets the total number of points seen during training.
    /// </summary>
    public long TotalPointsSeen => _totalPointsSeen;

    /// <summary>
    /// Gets the current learning rate.
    /// </summary>
    public double CurrentLearningRate { get; private set; }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newInstance = (OnlineKMeans<T>)CreateNewInstance();
        newInstance.SetParameters(parameters);
        return newInstance;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int d = x.Columns;
        int k = _options.NumClusters;
        NumFeatures = d;
        NumClusters = k;

        var rand = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize centers if not already initialized
        if (_centers.Rows == 0)
        {
            InitializeCenters(x, k, d, rand);
        }

        var labels = new int[n];
        CurrentLearningRate = _options.LearningRate;

        // Multi-pass: when MaxIterations > 1, iterate through the data multiple times
        // for better convergence. This is standard for batch online KMeans.
        int numPasses = Math.Max(1, _options.MaxIterations);

        for (int pass = 0; pass < numPasses; pass++)
        {
            // Process each point: every update changes the centers the next point is compared with
            for (int i = 0; i < n; i++)
            {
                var point = x.GetRow(i);

                // Find nearest center
                int nearest = FindNearestCenter(point, k);
                labels[i] = nearest;

                // Update center
                UpdateCenter(point, nearest);

                _totalPointsSeen++;
                _clusterCounts[nearest]++;

                // Decay learning rate
                if (_options.DecayLearningRate)
                {
                    CurrentLearningRate = Math.Max(
                        _options.MinLearningRate,
                        _options.LearningRate / (1 + _totalPointsSeen * 0.0001)
                    );
                }
            }
        }

        // Set results — clone internal centers to base class property (avoid aliasing)
        ClusterCenters = _centers.Clone();
        Labels = ClusteringTensorOps.FromIntArray<T>(labels);

        MergeDegenerateClusters(x);
        IsTrained = true;
    }

    private void InitializeCenters(Matrix<T> x, int k, int d, Random rand)
    {
        _clusterCounts = new int[k];

        // K-Means++ initialization: select well-separated initial centers
        // First center: random data point
        var chosen = new int[k];
        chosen[0] = rand.Next(x.Rows);

        // Subsequent centers: choose proportional to squared distance from nearest existing center
        Vector<T>? distances = null;
        for (int c = 1; c < k; c++)
        {
            var toLast = ClusteringTensorOps.SquaredDistancesToPoint(x.GetRow(chosen[c - 1]), x);
            distances = distances is null ? toLast : Engine.Min(distances, toLast);
            chosen[c] = ClusteringTensorOps.SampleProportional(distances, rand);
        }

        _centers = k > 0 ? ClusteringTensorOps.SelectRows(x, chosen) : new Matrix<T>(0, d);
    }

    private int FindNearestCenter(Vector<T> point, int k)
    {
        var centers = _centers.Rows == k ? _centers : _centers.GetSubMatrix(0, 0, k, _centers.Columns);
        return ClusteringTensorOps.ArgMin(ClusteringTensorOps.SquaredDistancesToPoint(point, centers));
    }

    private void UpdateCenter(Vector<T> point, int clusterIdx)
    {
        // center += lr * (point - center)
        T lr = NumOps.FromDouble(CurrentLearningRate);
        var center = _centers.GetRow(clusterIdx);
        Engine.SetRow(_centers, clusterIdx, Engine.Add(center, Engine.Multiply(Engine.Subtract(point, center), lr)));
    }

    /// <summary>
    /// Processes a single data point (true online/streaming mode).
    /// </summary>
    /// <param name="point">The data point to process.</param>
    /// <returns>The cluster assignment for this point.</returns>
    public int PartialFit(Vector<T> point)
    {
        if (_centers.Rows == 0)
        {
            throw new InvalidOperationException("Model must be initialized before calling PartialFit. " +
                "Call Train with initial data first, or use InitializeWithRandom.");
        }

        int nearest = FindNearestCenter(point, NumClusters);
        UpdateCenter(point, nearest);

        _totalPointsSeen++;
        _clusterCounts[nearest]++;

        if (_options.DecayLearningRate)
        {
            CurrentLearningRate = Math.Max(
                _options.MinLearningRate,
                _options.LearningRate / (1 + _totalPointsSeen * 0.0001)
            );
        }

        return nearest;
    }

    /// <summary>
    /// Initializes centers randomly for streaming mode.
    /// </summary>
    /// <param name="numFeatures">Number of features in the data.</param>
    /// <param name="minValues">Minimum values for each feature.</param>
    /// <param name="maxValues">Maximum values for each feature.</param>
    public void InitializeWithRandom(int numFeatures, T[]? minValues = null, T[]? maxValues = null)
    {
        int k = _options.NumClusters;
        int d = numFeatures;
        NumFeatures = d;
        NumClusters = k;

        _centers = new Matrix<T>(k, d);
        _clusterCounts = new int[k];

        var rand = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < d; j++)
            {
                T min = minValues is not null ? minValues[j] : NumOps.Zero;
                T max = maxValues is not null ? maxValues[j] : NumOps.One;
                T range = NumOps.Subtract(max, min);
                _centers[c, j] = NumOps.Add(min, NumOps.Multiply(NumOps.FromDouble(rand.NextDouble()), range));
            }
        }

        IsTrained = true;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> x)
    {
        ValidateIsTrained();

        int n = x.Rows;
        int k = NumClusters;
        if (n == 0 || k == 0)
        {
            return new Vector<T>(n);
        }

        var centers = _centers.Rows == k ? _centers : _centers.GetSubMatrix(0, 0, k, _centers.Columns);
        return ClusteringTensorOps.FromIntArray<T>(
            ClusteringTensorOps.ArgMinRows(ClusteringTensorOps.SquaredDistances(x, centers)));
    }

    /// <inheritdoc />
    public override Vector<T> FitPredict(Matrix<T> x)
    {
        Train(x);
        if (Labels is null)
        {
            throw new InvalidOperationException("Train did not produce labels. This indicates a bug in the training implementation.");
        }
        return Labels;
    }
}
