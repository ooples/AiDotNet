using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TimeSeries.AnomalyDetection;

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
public class TimeSeriesIsolationForest<T> : TimeSeriesModelBase<T>
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
        : base(options ?? new TimeSeriesIsolationForestOptions<T>())
    {
        _options = options ?? new TimeSeriesIsolationForestOptions<T>();
        _random = RandomHelper.CreateSeededRandom(_options.RandomSeed ?? 42);
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

        // Create feature matrix from time series
        var (features, validIndices) = CreateFeatureMatrix(timeSeries);

        int n = features.Rows;
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
        var sortedScores = scores.OrderByDescending(s => _numOps.ToDouble(s)).ToList();
        int thresholdIndex = (int)(_options.ContaminationRate * sortedScores.Count);
        _anomalyThreshold = _numOps.ToDouble(sortedScores[Math.Min(thresholdIndex, sortedScores.Count - 1)]);
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

        // Map scores back to original time series length
        var fullScores = new Vector<T>(timeSeries.Length);
        for (int i = 0; i < validIndices.Count; i++)
        {
            fullScores[validIndices[i]] = scores[i];
        }

        // Fill early indices with neutral score (0.5)
        for (int i = 0; i < timeSeries.Length; i++)
        {
            if (_numOps.ToDouble(fullScores[i]) == 0 && !validIndices.Contains(i))
            {
                fullScores[i] = _numOps.FromDouble(0.5);
            }
        }

        return fullScores;
    }

    /// <summary>
    /// Returns binary anomaly labels (true = anomaly).
    /// </summary>
    /// <param name="timeSeries">The time series to analyze.</param>
    /// <returns>Boolean vector indicating which points are anomalies.</returns>
    public bool[] GetAnomalyLabels(Vector<T> timeSeries)
    {
        var scores = DetectAnomalies(timeSeries);
        var labels = new bool[timeSeries.Length];

        for (int i = 0; i < timeSeries.Length; i++)
        {
            labels[i] = _numOps.ToDouble(scores[i]) > _anomalyThreshold;
        }

        return labels;
    }

    /// <summary>
    /// Gets the indices of detected anomalies.
    /// </summary>
    /// <param name="timeSeries">The time series to analyze.</param>
    /// <returns>List of indices where anomalies were detected.</returns>
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

        var validIndices = new List<int>();
        var featureRows = new List<T[]>();

        for (int i = startIndex; i < n; i++)
        {
            var features = new T[numFeatures];
            int idx = 0;

            // Original value
            features[idx++] = timeSeries[i];

            // Lag features
            for (int lag = 1; lag <= _options.LagFeatures; lag++)
            {
                features[idx++] = timeSeries[i - lag];
            }

            // Rolling statistics
            var windowValues = new List<T>();
            for (int j = i - _options.RollingWindowSize + 1; j <= i; j++)
            {
                windowValues.Add(timeSeries[j]);
            }

            features[idx++] = ComputeMean(windowValues);
            features[idx++] = ComputeStd(windowValues);
            features[idx++] = ComputeMin(windowValues);
            features[idx++] = ComputeMax(windowValues);

            // Trend features
            if (_options.UseTrendFeatures)
            {
                // First derivative (rate of change)
                features[idx++] = _numOps.Subtract(timeSeries[i], timeSeries[i - 1]);

                // Second derivative (acceleration)
                if (i >= 2)
                {
                    var d1 = _numOps.Subtract(timeSeries[i], timeSeries[i - 1]);
                    var d0 = _numOps.Subtract(timeSeries[i - 1], timeSeries[i - 2]);
                    features[idx++] = _numOps.Subtract(d1, d0);
                }
                else
                {
                    features[idx++] = _numOps.Zero;
                }
            }

            // Seasonal residual
            if (_options.UseSeasonalDecomposition && i >= _options.SeasonalPeriod)
            {
                // Simple seasonal difference
                features[idx++] = _numOps.Subtract(timeSeries[i], timeSeries[i - _options.SeasonalPeriod]);
            }
            else if (_options.UseSeasonalDecomposition)
            {
                features[idx++] = _numOps.Zero;
            }

            featureRows.Add(features);
            validIndices.Add(i);
        }

        // Convert to matrix
        var matrix = new Matrix<T>(featureRows.Count, numFeatures);
        for (int i = 0; i < featureRows.Count; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                matrix[i, j] = featureRows[i][j];
            }
        }

        return (matrix, validIndices);
    }

    private T ComputeMean(List<T> values)
    {
        var sum = _numOps.Zero;
        foreach (var v in values)
        {
            sum = _numOps.Add(sum, v);
        }
        return _numOps.Divide(sum, _numOps.FromDouble(values.Count));
    }

    private T ComputeStd(List<T> values)
    {
        var mean = ComputeMean(values);
        var sumSq = _numOps.Zero;
        foreach (var v in values)
        {
            var diff = _numOps.Subtract(v, mean);
            sumSq = _numOps.Add(sumSq, _numOps.Multiply(diff, diff));
        }
        var variance = _numOps.Divide(sumSq, _numOps.FromDouble(values.Count));
        return _numOps.Sqrt(variance);
    }

    private T ComputeMin(List<T> values)
    {
        var min = values[0];
        foreach (var v in values)
        {
            if (_numOps.ToDouble(v) < _numOps.ToDouble(min))
                min = v;
        }
        return min;
    }

    private T ComputeMax(List<T> values)
    {
        var max = values[0];
        foreach (var v in values)
        {
            if (_numOps.ToDouble(v) > _numOps.ToDouble(max))
                max = v;
        }
        return max;
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
        var result = new Matrix<T>(indices.Count, matrix.Columns);
        for (int i = 0; i < indices.Count; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = matrix[indices[i], j];
            }
        }
        return result;
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

        // Find min and max for this feature
        double minVal = double.MaxValue;
        double maxVal = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            double val = _numOps.ToDouble(data[i, featureIndex]);
            minVal = Math.Min(minVal, val);
            maxVal = Math.Max(maxVal, val);
        }

        // If all values are the same, create leaf
        if (Math.Abs(maxVal - minVal) < 1e-10)
        {
            return new IsolationTree<T> { IsLeaf = true, Size = n };
        }

        // Random split value
        double splitValue = minVal + _random.NextDouble() * (maxVal - minVal);

        // Partition data
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        for (int i = 0; i < n; i++)
        {
            if (_numOps.ToDouble(data[i, featureIndex]) < splitValue)
                leftIndices.Add(i);
            else
                rightIndices.Add(i);
        }

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
        var scores = new Vector<T>(features.Rows);
        double c = ComputeC(_effectiveSampleSize);

        for (int i = 0; i < features.Rows; i++)
        {
            var sample = features.GetRow(i);
            double totalPathLength = 0;

            foreach (var tree in _forest!)
            {
                totalPathLength += ComputePathLength(tree, sample, 0);
            }

            double avgPathLength = totalPathLength / _forest.Count;

            // Anomaly score: s(x, n) = 2^(-E(h(x))/c(n))
            double score = Math.Pow(2, -avgPathLength / c);
            scores[i] = _numOps.FromDouble(score);
        }

        return scores;
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
            return ComputePathLength(tree.Left!, sample, currentDepth + 1);
        else
            return ComputePathLength(tree.Right!, sample, currentDepth + 1);
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
            ModelType = ModelType.TimeSeriesRegression,
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
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write options
        writer.Write(_options.NumTrees);
        writer.Write(_options.SampleSize ?? 256);
        writer.Write(_options.MaxDepth ?? -1);
        writer.Write(_options.ContaminationRate);
        writer.Write(_options.LagFeatures);
        writer.Write(_options.RollingWindowSize);
        writer.Write(_options.UseSeasonalDecomposition);
        writer.Write(_options.SeasonalPeriod);
        writer.Write(_options.UseTrendFeatures);
        writer.Write(_options.RandomSeed ?? 42);

        // Write computed values
        writer.Write(_anomalyThreshold);
        writer.Write(_effectiveSampleSize);
        writer.Write(_effectiveMaxDepth);

        // Write forest
        writer.Write(_forest?.Count ?? 0);
        if (_forest != null)
        {
            foreach (var tree in _forest)
            {
                SerializeTree(writer, tree);
            }
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read options (skip, they're set via constructor)
        _ = reader.ReadInt32(); // NumTrees
        _ = reader.ReadInt32(); // SampleSize
        _ = reader.ReadInt32(); // MaxDepth
        _ = reader.ReadDouble(); // ContaminationRate
        _ = reader.ReadInt32(); // LagFeatures
        _ = reader.ReadInt32(); // RollingWindowSize
        _ = reader.ReadBoolean(); // UseSeasonalDecomposition
        _ = reader.ReadInt32(); // SeasonalPeriod
        _ = reader.ReadBoolean(); // UseTrendFeatures
        _ = reader.ReadInt32(); // RandomSeed

        // Read computed values
        _anomalyThreshold = reader.ReadDouble();
        _effectiveSampleSize = reader.ReadInt32();
        _effectiveMaxDepth = reader.ReadInt32();

        // Read forest
        int forestSize = reader.ReadInt32();
        _forest = new List<IsolationTree<T>>();
        for (int i = 0; i < forestSize; i++)
        {
            _forest.Add(DeserializeTree(reader));
        }
    }

    private void SerializeTree(BinaryWriter writer, IsolationTree<T> tree)
    {
        writer.Write(tree.IsLeaf);
        writer.Write(tree.Size);

        if (!tree.IsLeaf)
        {
            writer.Write(tree.FeatureIndex);
            writer.Write(_numOps.ToDouble(tree.SplitValue));
            SerializeTree(writer, tree.Left!);
            SerializeTree(writer, tree.Right!);
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
    public T SplitValue { get; set; } = default!;
    public IsolationTree<T>? Left { get; set; }
    public IsolationTree<T>? Right { get; set; }
}
