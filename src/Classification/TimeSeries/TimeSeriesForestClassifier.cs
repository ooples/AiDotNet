using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.TimeSeries;

/// <summary>
/// Implements the Time Series Forest classifier.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Time Series Forest builds an ensemble of decision trees, where each
/// tree is trained on features extracted from a randomly selected interval of the time series.
/// This approach captures patterns at different time scales and positions.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>For each tree, randomly select an interval (start, end) from the time series</item>
/// <item>Extract summary features from that interval (mean, std, slope)</item>
/// <item>Train a decision tree on these interval features</item>
/// <item>Repeat for all trees in the ensemble</item>
/// <item>Predict by majority voting across all trees</item>
/// </list>
/// </para>
///
/// <para><b>Key features:</b>
/// <list type="bullet">
/// <item>Captures local patterns at different positions in the sequence</item>
/// <item>Robust to noise through ensemble averaging</item>
/// <item>Interpretable through interval selection</item>
/// <item>Handles variable-length sequences naturally</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Deng et al., "A Time Series Forest for Classification and Feature Extraction" (2013)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TimeSeriesForestClassifier<T> : ClassifierBase<T>, ITimeSeriesClassifier<T>
{
    private readonly TimeSeriesForestOptions<T> _options;
    private readonly Random _random;
    private List<IntervalTree>? _trees;
    private bool _isFitted;

    /// <summary>
    /// Gets the expected sequence length.
    /// </summary>
    public int SequenceLength { get; private set; }

    /// <summary>
    /// Gets the number of channels (variables) in the time series.
    /// </summary>
    public int NumChannels { get; private set; }

    /// <summary>
    /// Gets whether this classifier supports variable-length sequences.
    /// </summary>
    public bool SupportsVariableLengths => true;

    /// <summary>
    /// Represents an interval-based decision tree.
    /// </summary>
    private class IntervalTree
    {
        public int IntervalStart { get; set; }
        public int IntervalEnd { get; set; }
        public int ChannelIdx { get; set; }
        public DecisionTreeNode? Root { get; set; }
    }

    /// <summary>
    /// A node in the decision tree.
    /// </summary>
    private class DecisionTreeNode
    {
        public int FeatureIndex { get; set; } = -1;
        public double Threshold { get; set; }
        public bool IsLeaf { get; set; }
        public T PredictedClass { get; set; } = default!;
        public double[]? ClassProbabilities { get; set; }
        public DecisionTreeNode? Left { get; set; }
        public DecisionTreeNode? Right { get; set; }
    }

    /// <summary>
    /// Creates a new Time Series Forest classifier.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public TimeSeriesForestClassifier(TimeSeriesForestOptions<T>? options = null)
        : base(options)
    {
        _options = options ?? new TimeSeriesForestOptions<T>();
        _random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.TimeSeriesClassifier;

    /// <summary>
    /// Trains the Time Series Forest on time series sequences.
    /// </summary>
    public void TrainOnSequences(Tensor<T> sequences, Vector<T> labels)
    {
        ValidateSequenceInput(sequences, labels);

        int numSamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        NumChannels = sequences.Shape.Length > 2 ? sequences.Shape[2] : 1;
        SequenceLength = seqLen;

        ClassLabels = ExtractClassLabels(labels);
        NumClasses = ClassLabels.Length;

        int minInterval = Math.Max(3, (int)(seqLen * _options.MinIntervalFraction));
        int maxInterval = Math.Min(seqLen, (int)(seqLen * _options.MaxIntervalFraction));

        _trees = new List<IntervalTree>(_options.NumTrees);

        for (int t = 0; t < _options.NumTrees; t++)
        {
            // Randomly select interval and channel
            int intervalLength = _random.Next(minInterval, maxInterval + 1);
            int start = _random.Next(0, seqLen - intervalLength + 1);
            int end = start + intervalLength;
            int channel = _random.Next(0, NumChannels);

            // Extract features for all samples
            var features = ExtractIntervalFeatures(sequences, start, end, channel);

            // Build decision tree
            var tree = new IntervalTree
            {
                IntervalStart = start,
                IntervalEnd = end,
                ChannelIdx = channel,
                Root = BuildTree(features, labels, 0)
            };

            _trees.Add(tree);
        }

        _isFitted = true;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Convert matrix to tensor for sequence training
        var tensor = new Tensor<T>(new[] { x.Rows, x.Columns, 1 });
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                tensor[new[] { i, j, 0 }] = x[i, j];
            }
        }

        TrainOnSequences(tensor, y);
    }

    /// <summary>
    /// Predicts class labels for time series sequences.
    /// </summary>
    public Vector<T> PredictSequences(Tensor<T> sequences)
    {
        ValidateSequenceInput(sequences, null);

        int numSamples = sequences.Shape[0];
        var predictions = new Vector<T>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            // Ensemble voting
            var votes = new Dictionary<int, int>();
            for (int c = 0; c < NumClasses; c++)
            {
                votes[c] = 0;
            }

            foreach (var tree in _trees!)
            {
                var features = ExtractIntervalFeaturesForSample(sequences, i,
                    tree.IntervalStart, tree.IntervalEnd, tree.ChannelIdx);
                var pred = PredictTree(tree.Root!, features);
                int classIdx = GetClassIndexFromLabel(pred);
                if (classIdx >= 0)
                {
                    votes[classIdx]++;
                }
            }

            // Find class with most votes
            int bestClass = 0;
            int maxVotes = int.MinValue;
            foreach (var kv in votes)
            {
                if (kv.Value > maxVotes)
                {
                    maxVotes = kv.Value;
                    bestClass = kv.Key;
                }
            }
            predictions[i] = ClassLabels![bestClass];
        }

        return predictions;
    }

    /// <summary>
    /// Predicts class probabilities for time series sequences.
    /// </summary>
    public Matrix<T> PredictSequenceProbabilities(Tensor<T> sequences)
    {
        ValidateSequenceInput(sequences, null);

        int numSamples = sequences.Shape[0];
        var probabilities = new Matrix<T>(numSamples, NumClasses);

        for (int i = 0; i < numSamples; i++)
        {
            var voteCounts = new double[NumClasses];

            foreach (var tree in _trees!)
            {
                var features = ExtractIntervalFeaturesForSample(sequences, i,
                    tree.IntervalStart, tree.IntervalEnd, tree.ChannelIdx);
                var pred = PredictTree(tree.Root!, features);
                int classIdx = GetClassIndexFromLabel(pred);
                if (classIdx >= 0)
                {
                    voteCounts[classIdx]++;
                }
            }

            // Convert votes to probabilities
            double total = _trees.Count;
            for (int c = 0; c < NumClasses; c++)
            {
                probabilities[i, c] = NumOps.FromDouble(voteCounts[c] / total);
            }
        }

        return probabilities;
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> input)
    {
        var tensor = new Tensor<T>(new[] { input.Rows, input.Columns, 1 });
        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < input.Columns; j++)
            {
                tensor[new[] { i, j, 0 }] = input[i, j];
            }
        }

        return PredictSequences(tensor);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        // Serialize tree structure (simplified - just returns tree count)
        return new Vector<T>(1) { [0] = NumOps.FromDouble(_trees?.Count ?? 0) };
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        // Trees are structural - cannot be set from simple parameters
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var clone = new TimeSeriesForestClassifier<T>(_options);
        clone._trees = _trees;
        clone._isFitted = _isFitted;
        clone.ClassLabels = ClassLabels is not null ? new Vector<T>(ClassLabels.ToArray()) : null;
        clone.NumClasses = NumClasses;
        clone.SequenceLength = SequenceLength;
        clone.NumChannels = NumChannels;
        return clone;
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new TimeSeriesForestClassifier<T>(_options);
    }

    /// <inheritdoc />
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Tree-based model - no gradient computation
        return new Vector<T>(0);
    }

    /// <inheritdoc />
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Tree-based model - no gradient application
    }

    private Matrix<T> ExtractIntervalFeatures(Tensor<T> sequences, int start, int end, int channel)
    {
        int numSamples = sequences.Shape[0];
        int numFeatures = 3; // mean, std, slope
        var features = new Matrix<T>(numSamples, numFeatures);

        for (int s = 0; s < numSamples; s++)
        {
            var intervalFeatures = ExtractIntervalFeaturesForSample(sequences, s, start, end, channel);
            for (int f = 0; f < numFeatures; f++)
            {
                features[s, f] = intervalFeatures[f];
            }
        }

        return features;
    }

    private Vector<T> ExtractIntervalFeaturesForSample(Tensor<T> sequences, int sampleIdx,
        int start, int end, int channel)
    {
        int seqLen = sequences.Shape[1];
        int actualStart = Math.Max(0, Math.Min(start, seqLen - 1));
        int actualEnd = Math.Max(actualStart + 1, Math.Min(end, seqLen));
        int length = actualEnd - actualStart;

        // Calculate mean
        double sum = 0;
        for (int t = actualStart; t < actualEnd; t++)
        {
            int[] indices = NumChannels > 1
                ? new[] { sampleIdx, t, channel }
                : new[] { sampleIdx, t };
            sum += NumOps.ToDouble(sequences[indices]);
        }
        double mean = sum / length;

        // Calculate std
        double sqSum = 0;
        for (int t = actualStart; t < actualEnd; t++)
        {
            int[] indices = NumChannels > 1
                ? new[] { sampleIdx, t, channel }
                : new[] { sampleIdx, t };
            double val = NumOps.ToDouble(sequences[indices]);
            sqSum += (val - mean) * (val - mean);
        }
        double std = Math.Sqrt(sqSum / length);

        // Calculate slope using linear regression
        double slope = 0;
        if (length > 1)
        {
            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
            for (int t = actualStart; t < actualEnd; t++)
            {
                int[] indices = NumChannels > 1
                    ? new[] { sampleIdx, t, channel }
                    : new[] { sampleIdx, t };
                double x = t - actualStart;
                double y = NumOps.ToDouble(sequences[indices]);
                sumX += x;
                sumY += y;
                sumXY += x * y;
                sumX2 += x * x;
            }
            double denom = length * sumX2 - sumX * sumX;
            if (Math.Abs(denom) > 1e-10)
            {
                slope = (length * sumXY - sumX * sumY) / denom;
            }
        }

        return new Vector<T>(3)
        {
            [0] = NumOps.FromDouble(mean),
            [1] = NumOps.FromDouble(std),
            [2] = NumOps.FromDouble(slope)
        };
    }

    private DecisionTreeNode BuildTree(Matrix<T> features, Vector<T> labels, int depth)
    {
        int n = features.Rows;

        // Check stopping conditions
        if (n < _options.MinSamplesSplit ||
            (_options.MaxDepth > 0 && depth >= _options.MaxDepth) ||
            IsHomogeneous(labels))
        {
            return CreateLeafNode(labels);
        }

        // Find best split
        var (bestFeature, bestThreshold, bestGain) = FindBestSplit(features, labels);

        if (bestGain <= 0)
        {
            return CreateLeafNode(labels);
        }

        // Split data
        var (leftFeatures, leftLabels, rightFeatures, rightLabels) =
            SplitData(features, labels, bestFeature, bestThreshold);

        if (leftLabels.Length == 0 || rightLabels.Length == 0)
        {
            return CreateLeafNode(labels);
        }

        return new DecisionTreeNode
        {
            FeatureIndex = bestFeature,
            Threshold = bestThreshold,
            IsLeaf = false,
            Left = BuildTree(leftFeatures, leftLabels, depth + 1),
            Right = BuildTree(rightFeatures, rightLabels, depth + 1)
        };
    }

    private DecisionTreeNode CreateLeafNode(Vector<T> labels)
    {
        // Find majority class
        var classCounts = new Dictionary<int, int>();
        for (int i = 0; i < labels.Length; i++)
        {
            int classIdx = GetClassIndexFromLabel(labels[i]);
            if (!classCounts.ContainsKey(classIdx))
            {
                classCounts[classIdx] = 0;
            }
            classCounts[classIdx]++;
        }

        int majorityClass = 0;
        int maxCount = int.MinValue;
        foreach (var kv in classCounts)
        {
            if (kv.Value > maxCount)
            {
                maxCount = kv.Value;
                majorityClass = kv.Key;
            }
        }

        // Calculate class probabilities
        var probs = new double[NumClasses];
        foreach (var kv in classCounts)
        {
            probs[kv.Key] = (double)kv.Value / labels.Length;
        }

        return new DecisionTreeNode
        {
            IsLeaf = true,
            PredictedClass = ClassLabels![majorityClass],
            ClassProbabilities = probs
        };
    }

    private bool IsHomogeneous(Vector<T> labels)
    {
        if (labels.Length <= 1) return true;

        T first = labels[0];
        for (int i = 1; i < labels.Length; i++)
        {
            if (NumOps.Compare(labels[i], first) != 0)
            {
                return false;
            }
        }
        return true;
    }

    private (int Feature, double Threshold, double Gain) FindBestSplit(Matrix<T> features, Vector<T> labels)
    {
        int bestFeature = -1;
        double bestThreshold = 0;
        double bestGain = 0;

        double parentEntropy = CalculateEntropy(labels);

        for (int f = 0; f < features.Columns; f++)
        {
            // Get sorted unique values for this feature
            var values = new List<double>();
            for (int i = 0; i < features.Rows; i++)
            {
                values.Add(NumOps.ToDouble(features[i, f]));
            }
            values.Sort();

            // Try midpoints as thresholds
            for (int i = 0; i < values.Count - 1; i++)
            {
                if (Math.Abs(values[i] - values[i + 1]) < 1e-10) continue;

                double threshold = (values[i] + values[i + 1]) / 2;

                // Calculate information gain
                var (leftLabels, rightLabels) = SplitLabels(features, labels, f, threshold);

                if (leftLabels.Length == 0 || rightLabels.Length == 0) continue;

                double leftWeight = (double)leftLabels.Length / labels.Length;
                double rightWeight = (double)rightLabels.Length / labels.Length;

                double gain = parentEntropy -
                    leftWeight * CalculateEntropy(leftLabels) -
                    rightWeight * CalculateEntropy(rightLabels);

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = f;
                    bestThreshold = threshold;
                }
            }
        }

        return (bestFeature, bestThreshold, bestGain);
    }

    private double CalculateEntropy(Vector<T> labels)
    {
        var counts = new Dictionary<int, int>();
        for (int i = 0; i < labels.Length; i++)
        {
            int classIdx = GetClassIndexFromLabel(labels[i]);
            if (!counts.ContainsKey(classIdx))
            {
                counts[classIdx] = 0;
            }
            counts[classIdx]++;
        }

        double entropy = 0;
        foreach (var count in counts.Values)
        {
            double p = (double)count / labels.Length;
            if (p > 0)
            {
                entropy -= p * (Math.Log(p) / Math.Log(2));
            }
        }

        return entropy;
    }

    private (Vector<T> Left, Vector<T> Right) SplitLabels(Matrix<T> features, Vector<T> labels,
        int featureIdx, double threshold)
    {
        var leftList = new List<T>();
        var rightList = new List<T>();

        for (int i = 0; i < features.Rows; i++)
        {
            if (NumOps.ToDouble(features[i, featureIdx]) <= threshold)
            {
                leftList.Add(labels[i]);
            }
            else
            {
                rightList.Add(labels[i]);
            }
        }

        return (new Vector<T>(leftList.ToArray()), new Vector<T>(rightList.ToArray()));
    }

    private (Matrix<T>, Vector<T>, Matrix<T>, Vector<T>) SplitData(Matrix<T> features, Vector<T> labels,
        int featureIdx, double threshold)
    {
        var leftFeatures = new List<T[]>();
        var leftLabels = new List<T>();
        var rightFeatures = new List<T[]>();
        var rightLabels = new List<T>();

        for (int i = 0; i < features.Rows; i++)
        {
            var row = new T[features.Columns];
            for (int j = 0; j < features.Columns; j++)
            {
                row[j] = features[i, j];
            }

            if (NumOps.ToDouble(features[i, featureIdx]) <= threshold)
            {
                leftFeatures.Add(row);
                leftLabels.Add(labels[i]);
            }
            else
            {
                rightFeatures.Add(row);
                rightLabels.Add(labels[i]);
            }
        }

        return (
            ListToMatrix(leftFeatures, features.Columns),
            new Vector<T>(leftLabels.ToArray()),
            ListToMatrix(rightFeatures, features.Columns),
            new Vector<T>(rightLabels.ToArray())
        );
    }

    private Matrix<T> ListToMatrix(List<T[]> list, int cols)
    {
        if (list.Count == 0) return new Matrix<T>(0, cols);

        var matrix = new Matrix<T>(list.Count, cols);
        for (int i = 0; i < list.Count; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = list[i][j];
            }
        }
        return matrix;
    }

    private T PredictTree(DecisionTreeNode node, Vector<T> features)
    {
        if (node.IsLeaf)
        {
            return node.PredictedClass;
        }

        double value = NumOps.ToDouble(features[node.FeatureIndex]);
        if (value <= node.Threshold)
        {
            return PredictTree(node.Left!, features);
        }
        else
        {
            return PredictTree(node.Right!, features);
        }
    }

    private void ValidateSequenceInput(Tensor<T> sequences, Vector<T>? labels)
    {
        if (sequences is null)
        {
            throw new ArgumentNullException(nameof(sequences));
        }

        if (sequences.Shape.Length < 2)
        {
            throw new ArgumentException("Sequences must be at least 2D [samples, sequence_length].");
        }

        if (labels is not null && labels.Length != sequences.Shape[0])
        {
            throw new ArgumentException("Number of labels must match number of samples.");
        }
    }
}
