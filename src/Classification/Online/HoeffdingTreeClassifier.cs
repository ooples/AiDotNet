using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.Online;

/// <summary>
/// Implements the Hoeffding Tree (Very Fast Decision Tree) for online classification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Hoeffding Tree is a decision tree that can be built incrementally
/// from streaming data. It uses the Hoeffding bound to determine when enough samples have been
/// seen to make a confident split decision.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Start with a single leaf node</item>
/// <item>As samples arrive, update statistics at each leaf</item>
/// <item>When enough samples are seen, use Hoeffding bound to decide if the best split is significantly better</item>
/// <item>If yes, split the leaf into internal node with two children</item>
/// <item>Repeat for all leaves as new samples arrive</item>
/// </list>
/// </para>
///
/// <para><b>The Hoeffding bound:</b>
/// With probability (1 - δ), the true best attribute is within ε of the observed best,
/// where ε = sqrt(R² * ln(1/δ) / (2n)) and n is the number of samples.</para>
///
/// <para><b>Advantages:</b>
/// <list type="bullet">
/// <item>Constant memory per node (bounded statistics)</item>
/// <item>Processes each sample only once</item>
/// <item>Converges to batch decision tree with enough samples</item>
/// <item>Suitable for infinite data streams</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Domingos & Hulten, "Mining High-Speed Data Streams" (2000)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class HoeffdingTreeClassifier<T> : ClassifierBase<T>, IOnlineClassifier<T>
{
    private readonly HoeffdingTreeOptions<T> _options;
    private readonly Random _random;
    private HoeffdingNode? _root;
    private readonly List<T> _knownClasses;

    /// <summary>
    /// Gets the total number of samples the model has seen.
    /// </summary>
    public long SamplesSeen { get; private set; }

    /// <summary>
    /// Gets whether the model has seen at least one sample.
    /// </summary>
    public bool IsWarm => SamplesSeen > 0;

    /// <summary>
    /// Represents a node in the Hoeffding tree.
    /// </summary>
    private class HoeffdingNode
    {
        public bool IsLeaf { get; set; } = true;
        public int Depth { get; set; }

        // Leaf statistics
        public Dictionary<int, long> ClassCounts { get; } = new();
        public long TotalCount { get; set; }

        // Split information (for internal nodes)
        public int SplitFeature { get; set; } = -1;
        public double SplitThreshold { get; set; }
        public HoeffdingNode? Left { get; set; }
        public HoeffdingNode? Right { get; set; }

        // Feature statistics (for leaves)
        public Dictionary<int, FeatureStats>? FeatureStatistics { get; set; }
    }

    /// <summary>
    /// Statistics for a single feature at a node.
    /// </summary>
    private class FeatureStats
    {
        public double Min { get; set; } = double.MaxValue;
        public double Max { get; set; } = double.MinValue;
        public Dictionary<int, BinStats>? BinsByClass { get; set; }
    }

    /// <summary>
    /// Statistics for feature bins per class.
    /// </summary>
    private class BinStats
    {
        public long[] Counts { get; set; } = Array.Empty<long>();
    }

    /// <summary>
    /// Creates a new Hoeffding Tree classifier.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public HoeffdingTreeClassifier(HoeffdingTreeOptions<T>? options = null)
        : base(options)
    {
        _options = options ?? new HoeffdingTreeOptions<T>();
        _random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();
        _knownClasses = new List<T>();
        _root = CreateLeaf(0);
    }

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.OnlineLearning;

    /// <summary>
    /// Updates the model with a single training sample.
    /// </summary>
    public void PartialFit(Vector<T> features, T label)
    {
        // Initialize feature statistics on first sample
        if (NumFeatures == 0)
        {
            NumFeatures = features.Length;
            InitializeFeatureStats(_root!, NumFeatures);
        }

        // Register new class if needed
        int classIdx = GetOrCreateClassIndex(label);

        // Sort sample to appropriate leaf
        var leaf = SortToLeaf(_root!, features);

        // Update leaf statistics
        UpdateLeafStats(leaf, features, classIdx);
        SamplesSeen++;

        // Check if we should attempt a split
        if (leaf.TotalCount >= _options.GracePeriod &&
            leaf.TotalCount % _options.GracePeriod == 0)
        {
            AttemptSplit(leaf);
        }
    }

    /// <summary>
    /// Updates the model with a batch of training samples.
    /// </summary>
    public void PartialFit(Matrix<T> features, Vector<T> labels)
    {
        for (int i = 0; i < features.Rows; i++)
        {
            var sample = new Vector<T>(features.Columns);
            for (int j = 0; j < features.Columns; j++)
            {
                sample[j] = features[i, j];
            }
            PartialFit(sample, labels[i]);
        }
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        PartialFit(x, y);
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            var features = new Vector<T>(input.Columns);
            for (int j = 0; j < input.Columns; j++)
            {
                features[j] = input[i, j];
            }
            predictions[i] = PredictSingle(features);
        }

        return predictions;
    }

    private T PredictSingle(Vector<T> features)
    {
        if (_root is null || !IsWarm)
        {
            return _knownClasses.Count > 0 ? _knownClasses[0] : default!;
        }

        var leaf = SortToLeaf(_root, features);
        return GetMajorityClass(leaf);
    }

    private HoeffdingNode SortToLeaf(HoeffdingNode node, Vector<T> features)
    {
        if (node.IsLeaf)
        {
            return node;
        }

        double value = NumOps.ToDouble(features[node.SplitFeature]);
        if (value <= node.SplitThreshold)
        {
            return SortToLeaf(node.Left!, features);
        }
        else
        {
            return SortToLeaf(node.Right!, features);
        }
    }

    private T GetMajorityClass(HoeffdingNode node)
    {
        if (node.ClassCounts.Count == 0)
        {
            return _knownClasses.Count > 0 ? _knownClasses[0] : default!;
        }

        int majorityIdx = 0;
        long maxCount = 0;
        foreach (var kv in node.ClassCounts)
        {
            if (kv.Value > maxCount)
            {
                maxCount = kv.Value;
                majorityIdx = kv.Key;
            }
        }
        return majorityIdx < _knownClasses.Count ? _knownClasses[majorityIdx] : default!;
    }

    private HoeffdingNode CreateLeaf(int depth)
    {
        return new HoeffdingNode
        {
            IsLeaf = true,
            Depth = depth
        };
    }

    private void InitializeFeatureStats(HoeffdingNode node, int numFeatures)
    {
        node.FeatureStatistics = new Dictionary<int, FeatureStats>();
        for (int f = 0; f < numFeatures; f++)
        {
            node.FeatureStatistics[f] = new FeatureStats
            {
                BinsByClass = new Dictionary<int, BinStats>()
            };
        }
    }

    private int GetOrCreateClassIndex(T label)
    {
        for (int i = 0; i < _knownClasses.Count; i++)
        {
            if (NumOps.Compare(_knownClasses[i], label) == 0)
            {
                return i;
            }
        }

        _knownClasses.Add(label);
        NumClasses = _knownClasses.Count;
        ClassLabels = new Vector<T>(_knownClasses.ToArray());
        return _knownClasses.Count - 1;
    }

    private void UpdateLeafStats(HoeffdingNode leaf, Vector<T> features, int classIdx)
    {
        // Update class count
        if (!leaf.ClassCounts.ContainsKey(classIdx))
        {
            leaf.ClassCounts[classIdx] = 0;
        }
        leaf.ClassCounts[classIdx]++;
        leaf.TotalCount++;

        // Update feature statistics
        if (leaf.FeatureStatistics is null)
        {
            InitializeFeatureStats(leaf, features.Length);
        }

        for (int f = 0; f < features.Length; f++)
        {
            double value = NumOps.ToDouble(features[f]);
            var stats = leaf.FeatureStatistics![f];

            // Update min/max
            stats.Min = Math.Min(stats.Min, value);
            stats.Max = Math.Max(stats.Max, value);

            // Initialize bins for this class if needed
            if (!stats.BinsByClass!.ContainsKey(classIdx))
            {
                stats.BinsByClass[classIdx] = new BinStats
                {
                    Counts = new long[_options.NumBins]
                };
            }

            // Update bin count
            int binIdx = GetBinIndex(value, stats.Min, stats.Max);
            stats.BinsByClass[classIdx].Counts[binIdx]++;
        }
    }

    private int GetBinIndex(double value, double min, double max)
    {
        if (Math.Abs(max - min) < 1e-10)
        {
            return 0;
        }

        int bin = (int)(((value - min) / (max - min)) * (_options.NumBins - 1));
        return Math.Max(0, Math.Min(_options.NumBins - 1, bin));
    }

    private void AttemptSplit(HoeffdingNode leaf)
    {
        if (_options.MaxDepth > 0 && leaf.Depth >= _options.MaxDepth)
        {
            return;
        }

        if (leaf.FeatureStatistics is null || NumFeatures == 0)
        {
            return;
        }

        // Calculate information gain for each feature/threshold combination
        var splitCandidates = new List<(int Feature, double Threshold, double Gain)>();
        double currentEntropy = CalculateEntropy(leaf.ClassCounts, leaf.TotalCount);

        for (int f = 0; f < NumFeatures; f++)
        {
            var stats = leaf.FeatureStatistics[f];
            if (Math.Abs(stats.Max - stats.Min) < 1e-10) continue;

            // Try bin boundaries as thresholds
            for (int b = 1; b < _options.NumBins; b++)
            {
                double threshold = stats.Min + (stats.Max - stats.Min) * b / _options.NumBins;
                double gain = CalculateInformationGain(leaf, f, threshold, currentEntropy);
                splitCandidates.Add((f, threshold, gain));
            }
        }

        if (splitCandidates.Count < 2)
        {
            return;
        }

        // Sort by gain
        splitCandidates.Sort((a, b) => b.Gain.CompareTo(a.Gain));
        var best = splitCandidates[0];
        var secondBest = splitCandidates[1];

        // Calculate Hoeffding bound
        double r = Math.Log(Math.Max(2, NumClasses)) / Math.Log(2); // Range of information gain
        double epsilon = Math.Sqrt(r * r * Math.Log(1.0 / _options.Delta) / (2.0 * leaf.TotalCount));

        // Check if we should split
        bool shouldSplit = (best.Gain - secondBest.Gain > epsilon) ||
                          (epsilon < _options.TieThreshold && best.Gain > secondBest.Gain);

        if (shouldSplit && best.Gain > 0)
        {
            PerformSplit(leaf, best.Feature, best.Threshold);
        }
    }

    private double CalculateEntropy(Dictionary<int, long> classCounts, long total)
    {
        if (total == 0) return 0;

        double entropy = 0;
        foreach (var count in classCounts.Values)
        {
            if (count > 0)
            {
                double p = (double)count / total;
                entropy -= p * (Math.Log(p) / Math.Log(2));
            }
        }
        return entropy;
    }

    private double CalculateInformationGain(HoeffdingNode leaf, int feature, double threshold,
        double currentEntropy)
    {
        var stats = leaf.FeatureStatistics![feature];
        var leftCounts = new Dictionary<int, long>();
        var rightCounts = new Dictionary<int, long>();
        long leftTotal = 0, rightTotal = 0;

        // Estimate split using bin statistics
        int splitBin = GetBinIndex(threshold, stats.Min, stats.Max);

        foreach (var kvp in stats.BinsByClass!)
        {
            int classIdx = kvp.Key;
            var bins = kvp.Value.Counts;

            long leftCount = 0, rightCount = 0;
            for (int b = 0; b <= splitBin; b++)
            {
                leftCount += bins[b];
            }
            for (int b = splitBin + 1; b < bins.Length; b++)
            {
                rightCount += bins[b];
            }

            if (leftCount > 0) leftCounts[classIdx] = leftCount;
            if (rightCount > 0) rightCounts[classIdx] = rightCount;
            leftTotal += leftCount;
            rightTotal += rightCount;
        }

        if (leftTotal == 0 || rightTotal == 0)
        {
            return 0;
        }

        long total = leftTotal + rightTotal;
        double leftWeight = (double)leftTotal / total;
        double rightWeight = (double)rightTotal / total;

        double leftEntropy = CalculateEntropy(leftCounts, leftTotal);
        double rightEntropy = CalculateEntropy(rightCounts, rightTotal);

        return currentEntropy - leftWeight * leftEntropy - rightWeight * rightEntropy;
    }

    private void PerformSplit(HoeffdingNode leaf, int feature, double threshold)
    {
        leaf.IsLeaf = false;
        leaf.SplitFeature = feature;
        leaf.SplitThreshold = threshold;

        leaf.Left = CreateLeaf(leaf.Depth + 1);
        leaf.Right = CreateLeaf(leaf.Depth + 1);

        InitializeFeatureStats(leaf.Left, NumFeatures);
        InitializeFeatureStats(leaf.Right, NumFeatures);

        // Clear leaf statistics (no longer needed)
        leaf.FeatureStatistics = null;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        // Tree structure is complex - return empty for now
        return new Vector<T>(0);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        // Tree is structural, cannot set from flat parameters
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var clone = new HoeffdingTreeClassifier<T>(_options);
        clone._root = _root;
        clone._knownClasses.AddRange(_knownClasses);
        clone.NumClasses = NumClasses;
        clone.NumFeatures = NumFeatures;
        clone.ClassLabels = ClassLabels is not null ? new Vector<T>(ClassLabels.ToArray()) : null;
        return clone;
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new HoeffdingTreeClassifier<T>(_options);
    }

    /// <inheritdoc />
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Tree-based model - no gradients
        return new Vector<T>(0);
    }

    /// <inheritdoc />
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Tree-based model - no gradient application
    }
}
