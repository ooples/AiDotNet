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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Clustering)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Vector<>), typeof(Vector<>))]
public class OnlineKMeans<T> : ClusteringBase<T>
{
    private readonly OnlineKMeansOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private T[,]? _centers;
    private int[]? _clusterCounts;
    private long _totalPointsSeen;

    /// <summary>
    /// Initializes a new Online K-Means instance.
    /// </summary>
    /// <param name="options">The Online K-Means options.</param>
    public OnlineKMeans(OnlineKMeansOptions<T>? options = null)
        : base(options ?? new OnlineKMeansOptions<T>())
    {
        _options = options ?? new OnlineKMeansOptions<T>();
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
    protected override ModelType GetModelType() => ModelType.Clustering;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new OnlineKMeans<T>(new OnlineKMeansOptions<T>
        {
            NumClusters = _options.NumClusters,
            LearningRate = _options.LearningRate,
            DecayLearningRate = _options.DecayLearningRate,
            MinLearningRate = _options.MinLearningRate,
            MaxIterations = _options.MaxIterations,
            DistanceMetric = _options.DistanceMetric
        });
    }

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
        if (_centers is null)
        {
            InitializeCenters(x, k, d, rand);
        }

        var labels = new int[n];
        CurrentLearningRate = _options.LearningRate;

        // Process each point
        for (int i = 0; i < n; i++)
        {
            var point = new T[d];
            for (int j = 0; j < d; j++)
            {
                point[j] = x[i, j];
            }

            // Find nearest center
            int nearest = FindNearestCenter(point, d, k);
            labels[i] = nearest;

            // Update center
            UpdateCenter(point, nearest, d);

            _totalPointsSeen++;
            _clusterCounts![nearest]++;

            // Decay learning rate
            if (_options.DecayLearningRate)
            {
                CurrentLearningRate = Math.Max(
                    _options.MinLearningRate,
                    _options.LearningRate / (1 + _totalPointsSeen * 0.0001)
                );
            }
        }

        // Set results
        ClusterCenters = new Matrix<T>(k, d);
        Labels = new Vector<T>(n);

        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < d; j++)
            {
                ClusterCenters[c, j] = _centers![c, j];
            }
        }

        for (int i = 0; i < n; i++)
        {
            Labels[i] = NumOps.FromDouble(labels[i]);
        }

        IsTrained = true;
    }

    private void InitializeCenters(Matrix<T> x, int k, int d, Random rand)
    {
        _centers = new T[k, d];
        _clusterCounts = new int[k];

        // Initialize from random data points
        var indices = Enumerable.Range(0, x.Rows).OrderBy(_ => rand.Next()).Take(k).ToArray();
        for (int c = 0; c < k; c++)
        {
            int idx = indices[c];
            for (int j = 0; j < d; j++)
            {
                _centers[c, j] = x[idx, j];
            }
        }
    }

    private int FindNearestCenter(T[] point, int d, int k)
    {
        int nearest = 0;
        T minDist = NumOps.MaxValue;

        for (int c = 0; c < k; c++)
        {
            T dist = NumOps.Zero;
            for (int j = 0; j < d; j++)
            {
                T diff = NumOps.Subtract(point[j], _centers![c, j]);
                dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
            }

            if (NumOps.LessThan(dist, minDist))
            {
                minDist = dist;
                nearest = c;
            }
        }

        return nearest;
    }

    private void UpdateCenter(T[] point, int clusterIdx, int d)
    {
        T lr = NumOps.FromDouble(CurrentLearningRate);
        for (int j = 0; j < d; j++)
        {
            T diff = NumOps.Subtract(point[j], _centers![clusterIdx, j]);
            _centers[clusterIdx, j] = NumOps.Add(_centers[clusterIdx, j], NumOps.Multiply(lr, diff));
        }
    }

    /// <summary>
    /// Processes a single data point (true online/streaming mode).
    /// </summary>
    /// <param name="point">The data point to process.</param>
    /// <returns>The cluster assignment for this point.</returns>
    public int PartialFit(Vector<T> point)
    {
        int d = point.Length;

        if (_centers is null)
        {
            throw new InvalidOperationException("Model must be initialized before calling PartialFit. " +
                "Call Train with initial data first, or use InitializeWithRandom.");
        }

        var pointArray = new T[d];
        for (int j = 0; j < d; j++)
        {
            pointArray[j] = point[j];
        }

        int nearest = FindNearestCenter(pointArray, d, NumClusters);
        UpdateCenter(pointArray, nearest, d);

        _totalPointsSeen++;
        _clusterCounts![nearest]++;

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

        _centers = new T[k, d];
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
        int d = NumFeatures;
        int k = NumClusters;
        var labels = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            var point = new T[d];
            for (int j = 0; j < d; j++)
            {
                point[j] = x[i, j];
            }

            int nearest = FindNearestCenter(point, d, k);
            labels[i] = NumOps.FromDouble(nearest);
        }

        return labels;
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
