using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TimeSeries.AnomalyDetection;

#pragma warning disable CS8601, CS8618 // Generic T defaults use default(T) - always used with value types

/// <summary>
/// Implements Isolation Forest for time series anomaly detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>The Time Series Anomaly Detection Challenge:</b>
/// Traditional anomaly detection treats each data point independently. For time series,
/// we need to consider temporal context - a value might be normal on its own but
/// anomalous given what came before or the time of day.
/// </para>
/// <para>
/// <b>How Time Series Isolation Forest Works:</b>
/// 1. **Feature Engineering**: Transform raw time series into feature vectors including:
///    - Lag features (past values)
///    - Rolling statistics (mean, std, min, max over recent windows)
///    - Trend indicators (derivative, acceleration)
///    - Seasonal residuals (deviation from expected seasonal pattern)
///
/// 2. **Isolation Forest**: For each feature vector:
///    - Randomly select a feature and split value
///    - Recursively partition until isolated
///    - Count path length to isolation
///    - Anomalies have shorter paths (easier to isolate)
///
/// 3. **Anomaly Scoring**: Compute anomaly score from average path length across all trees
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine you're trying to describe where someone lives.
/// For most people, you need many questions: "Which continent? Which country? Which city?..."
/// But if someone lives on a tiny island, you can identify them quickly: "Do you live on that island? Yes."
///
/// Isolation Forest uses this idea: anomalies are "easy to describe" (short paths),
/// while normal points need more questions to distinguish them.
///
/// For time series, we add context: "Is this value unusual compared to yesterday?
/// Is it unusual for this time of day? Is it unusual given the recent trend?"
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var testMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var trainingLabels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var trainingMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Create a time-series-aware isolation forest for anomaly detection
/// var options = new TimeSeriesIsolationForestOptions&lt;double&gt;();
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new TimeSeriesIsolationForest&lt;double&gt;(options))
///     .Build(trainingMatrix, trainingLabels);
/// Vector&lt;double&gt; anomalyScores = result.Predict(testMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Isolation Forest", "https://doi.org/10.1109/ICDM.2008.17", Year = 2008, Authors = "Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou")]
public partial class TimeSeriesIsolationForest<T> : TimeSeriesModelBase<T>
{
    private readonly TimeSeriesIsolationForestOptions<T> _options;

    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;

    private List<IsolationTree<T>>? _forest;
    private double _anomalyThreshold;
    private int _effectiveSampleSize;
    private int _effectiveMaxDepth;

    // Feature engineering statistics
    private T _featureMean;
    private T _featureStd;

    /// <summary>
    /// Initializes a new instance of the Time Series Isolation Forest.
    /// </summary>
    /// <param name="options">Configuration options. Uses defaults if null.</param>
    public TimeSeriesIsolationForest(TimeSeriesIsolationForestOptions<T>? options = null)
        : base(options ??= new TimeSeriesIsolationForestOptions<T>())
    {
        _options = options;
        Options = _options;
        _random = RandomHelper.CreateSeededRandom(_options.Seed ?? 42);
        _featureMean = _numOps.Zero;
        _featureStd = _numOps.One;
    }

    /// <summary>
    /// Trains the isolation forest on the time series data.
    /// </summary>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Extract time series from input
        var timeSeries = x.Rows > 0 ? x.GetColumn(0) : y;

        // Guard against too-short time series
        int minRequiredLength = Math.Max(_options.LagFeatures, _options.RollingWindowSize) + 1;
        if (timeSeries.Length < minRequiredLength)
        {
            throw new ArgumentException(
                $"Time series is too short ({timeSeries.Length} points). " +
                $"Minimum required length is {minRequiredLength} based on LagFeatures ({_options.LagFeatures}) " +
                $"and RollingWindowSize ({_options.RollingWindowSize}).",
                nameof(x));
        }

        // Create feature matrix from time series
        var (features, validIndices) = CreateFeatureMatrix(timeSeries);

        int n = features.Rows;

        // Additional safety check: ensure we have enough samples to build a forest
        if (n == 0)
        {
            throw new InvalidOperationException(
                "Feature matrix has zero rows. Time series may be too short for the configured options.");
        }

        _effectiveSampleSize = _options.SampleSize ?? Math.Min(256, n);
        _effectiveMaxDepth = _options.MaxDepth ?? (int)Math.Ceiling(Math.Log(_effectiveSampleSize) / Math.Log(2));

        // Build forest
        _forest = new List<IsolationTree<T>>();
        for (int t = 0; t < _options.NumTrees; t++)
        {
            // Sample data for this tree
            var sampleIndices = SampleIndices(n, _effectiveSampleSize);
            var sampleFeatures = ExtractRows(features, sampleIndices);

            // Build tree
            var tree = BuildTree(sampleFeatures, 0);
            _forest.Add(tree);
        }

        // Compute anomaly scores for training data to set threshold
        var scores = ComputeAnomalyScores(features);

        // Set threshold based on contamination rate
        // Descending order of the scores as doubles (NaN last, like OrderByDescending)
        double[] sortedScores = Array.ConvertAll(scores.ToArray(), _numOps.ToDouble);
        Array.Sort(sortedScores);
        Array.Reverse(sortedScores);
        int thresholdIndex = (int)(_options.ContaminationRate * sortedScores.Length);
        _anomalyThreshold = sortedScores[Math.Min(thresholdIndex, sortedScores.Length - 1)];

        // Store ModelParameters for GetParameters() — use anomaly threshold + tree count
        ModelParameters = new Vector<T>(2);
        ModelParameters[0] = _numOps.FromDouble(_anomalyThreshold);
        ModelParameters[1] = _numOps.FromDouble(_forest.Count);
    }

    /// <summary>
    /// Detects anomalies in the time series and returns anomaly scores.
    /// </summary>
    /// <param name="timeSeries">The time series to analyze.</param>
    /// <returns>Anomaly scores for each point (higher = more anomalous).</returns>
    public Vector<T> DetectAnomalies(Vector<T> timeSeries)
    {
        if (_forest == null)
            throw new InvalidOperationException("Model not trained. Call Train() first.");

        var (features, validIndices) = CreateFeatureMatrix(timeSeries);
        var scores = ComputeAnomalyScores(features);

        // Map scores back to original time series length; early indices get the neutral score (0.5)
        var fullScores = Engine.TensorFullLike(new Tensor<T>(new[] { timeSeries.Length }), _numOps.FromDouble(0.5));
        if (validIndices.Count > 0)
        {
            var positions = new Tensor<int>(new[] { validIndices.Count }, new Vector<int>(validIndices.ToArray()));
            Engine.TensorIndexPutInPlace(fullScores, new[] { positions },
                Tensor<T>.FromVector(scores.GetSubVector(0, validIndices.Count)));
        }

        return fullScores.ToVector();
    }

    /// <summary>
    /// Returns binary anomaly labels (true = anomaly).
    /// </summary>
    /// <param name="timeSeries">The time series to analyze.</param>
    /// <returns>Boolean vector indicating which points are anomalies.</returns>
    public bool[] GetAnomalyLabels(Vector<T> timeSeries)
    {
        var scores = DetectAnomalies(timeSeries);
        double[] values = Array.ConvertAll(scores.GetSubVector(0, timeSeries.Length).ToArray(), _numOps.ToDouble);
        return Array.ConvertAll(values, IsAboveThreshold);
    }

    /// <summary>
    /// Gets the indices of detected anomalies.
    /// </summary>
    /// <param name="timeSeries">The time series to analyze.</param>
    /// <returns>List of indices where anomalies were detected.</returns>
    /// <summary>Whether a score (as a double) exceeds the anomaly threshold.</summary>
    private bool IsAboveThreshold(double score) => score > _anomalyThreshold;

    public List<int> GetAnomalyIndices(Vector<T> timeSeries)
    {
        var labels = GetAnomalyLabels(timeSeries);
        return Enumerable.Range(0, labels.Length).Where(i => labels[i]).ToList();
    }

    private (Matrix<T> features, List<int> validIndices) CreateFeatureMatrix(Vector<T> timeSeries)
    {
        int n = timeSeries.Length;
        int startIndex = Math.Max(_options.LagFeatures, _options.RollingWindowSize);

        // Calculate number of features
        int numFeatures = 1 + // Original value
                         _options.LagFeatures + // Lag features
                         4 + // Rolling statistics (mean, std, min, max)
                         (_options.UseTrendFeatures ? 2 : 0) + // Trend features (derivative, acceleration)
                         (_options.UseSeasonalDecomposition ? 1 : 0); // Seasonal residual

        // One feature row for every time step i = startIndex .. n - 1
        int rows = Math.Max(0, n - startIndex);
        var validIndices = Enumerable.Range(startIndex, rows).ToList();
        if (rows == 0)
        {
            return (new Matrix<T>(0, numFeatures), validIndices);
        }

        // Original value
        var columns = new List<Vector<T>> { timeSeries.GetSubVector(startIndex, rows) };

        // Lag features: x[i - lag] for lag = 1..LagFeatures
        columns.AddRange(Enumerable.Range(1, Math.Max(0, _options.LagFeatures))
            .Select(lag => timeSeries.GetSubVector(startIndex - lag, rows)));

        // Rolling statistics over the window x[i - RollingWindowSize + 1 .. i]
        int window = _options.RollingWindowSize;
        if (window <= 0)
        {
            throw new ArgumentOutOfRangeException("index"); // an empty window has no minimum
        }

        var windows = Engine.TensorUnfold(
            Tensor<T>.FromVector(timeSeries.GetSubVector(startIndex - window + 1, rows + window - 1)), 0, window, 1);
        T count = _numOps.FromDouble(window);
        var mean = Engine.TensorDivideScalar(Engine.ReduceSum(windows, new[] { 1 }, keepDims: false), count);
        var centered = Engine.TensorSubtract(windows, Engine.TensorBroadcastTo(Engine.Reshape(mean, new[] { rows, 1 }), new[] { rows, window }));
        var variance = Engine.TensorDivideScalar(
            Engine.ReduceSum(Engine.TensorMultiply(centered, centered), new[] { 1 }, keepDims: false), count);
        columns.Add(mean.ToVector());
        columns.Add(Engine.TensorSqrt(variance).ToVector());
        columns.Add(WindowExtreme(windows, rows, minimum: true));
        columns.Add(WindowExtreme(windows, rows, minimum: false));

        // Trend features
        if (_options.UseTrendFeatures)
        {
            // First derivative (rate of change)
            var d1 = Engine.Subtract(timeSeries.GetSubVector(startIndex, rows), timeSeries.GetSubVector(startIndex - 1, rows));
            columns.Add(d1);

            // Second derivative (acceleration) where i >= 2, zero before
            int leading = Math.Min(rows, Math.Max(0, 2 - startIndex));
            var rest = d1.GetSubVector(leading, rows - leading);
            var d0 = Engine.Subtract(timeSeries.GetSubVector(startIndex + leading - 1, rows - leading),
                timeSeries.GetSubVector(startIndex + leading - 2, rows - leading));
            columns.Add(Vector<T>.Concatenate(new Vector<T>(leading), Engine.Subtract(rest, d0)));
        }

        // Seasonal residual: simple seasonal difference x[i] - x[i - SeasonalPeriod] where i >= SeasonalPeriod
        if (_options.UseSeasonalDecomposition)
        {
            int period = _options.SeasonalPeriod;
            int leading = Math.Min(rows, Math.Max(0, period - startIndex));
            var seasonal = Engine.Subtract(timeSeries.GetSubVector(startIndex + leading, rows - leading),
                timeSeries.GetSubVector(startIndex + leading - period, rows - leading));
            columns.Add(Vector<T>.Concatenate(new Vector<T>(leading), seasonal));
        }

        // Convert to matrix
        var matrix = Matrix<T>.FromRowVectors(columns).Transpose();
        return (matrix, validIndices);
    }

    /// <summary>
    /// Minimum (or maximum) of each window row as a running scan from the first element: NaN values
    /// after the first are skipped, a NaN first element stays.
    /// </summary>
    private Vector<T> WindowExtreme(Tensor<T> windows, int rows, bool minimum)
    {
        var isNan = Engine.TensorIsNan(windows);
        var clean = Engine.TensorWhere(isNan, Engine.TensorFullLike(windows, minimum ? _numOps.MaxValue : _numOps.MinValue), windows);
        var extreme = minimum
            ? Engine.TensorAmin(clean, new[] { 1 }, keepDims: false)
            : Engine.ReduceMax(clean, new[] { 1 }, keepDims: false);
        var first = Engine.Reshape(Engine.TensorSlice(windows, new[] { 0, 0 }, new[] { rows, 1 }), new[] { rows });
        return Engine.TensorWhere(Engine.TensorIsNan(first), first, extreme).ToVector();
    }

    private List<int> SampleIndices(int n, int sampleSize)
    {
        return Enumerable.Range(0, n)
            .OrderBy(_ => _random.Next())
            .Take(Math.Min(sampleSize, n))
            .ToList();
    }

    private Matrix<T> ExtractRows(Matrix<T> matrix, List<int> indices)
    {
        if (indices.Count == 0)
        {
            return new Matrix<T>(0, matrix.Columns);
        }

        var rows = new Tensor<int>(new[] { indices.Count }, new Vector<int>(indices.ToArray()));
        return Engine.TensorIndexSelect(Tensor<T>.FromMatrix(matrix), rows, 0).ToMatrix();
    }

    private IsolationTree<T> BuildTree(Matrix<T> data, int depth)
    {
        int n = data.Rows;

        // Terminal conditions
        if (depth >= _effectiveMaxDepth || n <= 1)
        {
            return new IsolationTree<T> { IsLeaf = true, Size = n };
        }

        // Select random feature
        int featureIndex = _random.Next(data.Columns);

        // Find min and max for this feature (as doubles; Math.Min/Max propagate NaN)
        double[] values = Array.ConvertAll(data.GetColumn(featureIndex).ToArray(), _numOps.ToDouble);
        double minVal = values.Aggregate(double.MaxValue, Math.Min);
        double maxVal = values.Aggregate(double.MinValue, Math.Max);

        // If all values are the same, create leaf
        if (Math.Abs(maxVal - minVal) < 1e-10)
        {
            return new IsolationTree<T> { IsLeaf = true, Size = n };
        }

        // Random split value
        double splitValue = minVal + _random.NextDouble() * (maxVal - minVal);

        // Partition data
        var leftIndices = Enumerable.Range(0, n).Where(i => values[i] < splitValue).ToList();
        var rightIndices = Enumerable.Range(0, n).Where(i => !(values[i] < splitValue)).ToList();

        // Handle edge case where all points go to one side
        if (leftIndices.Count == 0 || rightIndices.Count == 0)
        {
            return new IsolationTree<T> { IsLeaf = true, Size = n };
        }

        // Recursively build subtrees
        return new IsolationTree<T>
        {
            IsLeaf = false,
            FeatureIndex = featureIndex,
            SplitValue = _numOps.FromDouble(splitValue),
            Left = BuildTree(ExtractRows(data, leftIndices), depth + 1),
            Right = BuildTree(ExtractRows(data, rightIndices), depth + 1)
        };
    }

    private Vector<T> ComputeAnomalyScores(Matrix<T> features)
    {
        var scores = new List<double>(features.Rows);
        double c = ComputeC(_effectiveSampleSize);

        for (int i = 0; i < features.Rows; i++)
        {
            var sample = features.GetRow(i);
            double totalPathLength = 0;

            foreach (var tree in (_forest ?? throw new InvalidOperationException("_forest is null where it is required.")))
            {
                totalPathLength += ComputePathLength(tree, sample, 0);
            }

            double avgPathLength = totalPathLength / _forest.Count;

            // Anomaly score: s(x, n) = 2^(-E(h(x))/c(n))
            double score = Math.Pow(2, -avgPathLength / c);
            scores.Add(score);
        }

        return new Vector<T>(Array.ConvertAll(scores.ToArray(), _numOps.FromDouble));
    }

    private double ComputePathLength(IsolationTree<T> tree, Vector<T> sample, int currentDepth)
    {
        if (tree.IsLeaf)
        {
            // Add adjustment for remaining path length
            return currentDepth + ComputeC(tree.Size);
        }

        double sampleValue = _numOps.ToDouble(sample[tree.FeatureIndex]);
        double splitValue = _numOps.ToDouble(tree.SplitValue);

        if (sampleValue < splitValue)
            return ComputePathLength((tree.Left ?? throw new InvalidOperationException("tree.Left is null where it is required.")), sample, currentDepth + 1);
        else
            return ComputePathLength((tree.Right ?? throw new InvalidOperationException("tree.Right is null where it is required.")), sample, currentDepth + 1);
    }

    /// <summary>
    /// Computes c(n), the average path length of unsuccessful search in BST.
    /// </summary>
    private static double ComputeC(int n)
    {
        if (n <= 1) return 0;
        if (n == 2) return 1;

        // c(n) = 2*H(n-1) - 2*(n-1)/n
        // where H(i) is the harmonic number
        double H = Math.Log(n - 1) + 0.5772156649; // Euler-Mascheroni constant
        return 2 * H - 2.0 * (n - 1) / n;
    }

    /// <inheritdoc/>
    public override T PredictSingle(Vector<T> input)
    {
        // Return anomaly score for the last point
        var scores = DetectAnomalies(input);
        return scores[scores.Length - 1];
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                ["Architecture"] = "TimeSeriesIsolationForest",
                ["NumTrees"] = _options.NumTrees,
                ["SampleSize"] = _effectiveSampleSize,
                ["MaxDepth"] = _effectiveMaxDepth,
                ["ContaminationRate"] = _options.ContaminationRate,
                ["LagFeatures"] = _options.LagFeatures,
                ["RollingWindowSize"] = _options.RollingWindowSize,
                ["AnomalyThreshold"] = _anomalyThreshold
            }
        };
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new TimeSeriesIsolationForest<T>(new TimeSeriesIsolationForestOptions<T>(_options));
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    private void SerializeTree(BinaryWriter writer, IsolationTree<T> tree)
    {
        writer.Write(tree.IsLeaf);
        writer.Write(tree.Size);

        if (!tree.IsLeaf)
        {
            writer.Write(tree.FeatureIndex);
            writer.Write(_numOps.ToDouble(tree.SplitValue));
            SerializeTree(writer, (tree.Left ?? throw new InvalidOperationException("tree.Left is null where it is required.")));
            SerializeTree(writer, (tree.Right ?? throw new InvalidOperationException("tree.Right is null where it is required.")));
        }
    }

    private IsolationTree<T> DeserializeTree(BinaryReader reader)
    {
        bool isLeaf = reader.ReadBoolean();
        int size = reader.ReadInt32();

        if (isLeaf)
        {
            return new IsolationTree<T> { IsLeaf = true, Size = size };
        }

        int featureIndex = reader.ReadInt32();
        T splitValue = _numOps.FromDouble(reader.ReadDouble());

        return new IsolationTree<T>
        {
            IsLeaf = false,
            Size = size,
            FeatureIndex = featureIndex,
            SplitValue = splitValue,
            Left = DeserializeTree(reader),
            Right = DeserializeTree(reader)
        };
    }
}

/// <summary>
/// Represents a node in an isolation tree.
/// </summary>
internal class IsolationTree<T>
{
    public bool IsLeaf { get; set; }
    public int Size { get; set; }
    public int FeatureIndex { get; set; }
    public T SplitValue { get; set; } = default;
    public IsolationTree<T>? Left { get; set; }
    public IsolationTree<T>? Right { get; set; }
}
