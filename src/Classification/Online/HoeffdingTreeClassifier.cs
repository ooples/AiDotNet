using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
/// <para><b>Reference:</b> Domingos &amp; Hulten, "Mining High-Speed Data Streams" (2000)</para>
/// </remarks>
/// <example>
/// <code>
/// // Create Hoeffding tree for incremental stream classification
/// var options = new HoeffdingTreeOptions&lt;double&gt;();
///
/// // Prepare streaming data batch
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 1.1 }, { 1.2, 0.9 }, { 0.8, 1.0 }, { 5.0, 5.1 }, { 5.2, 4.9 }, { 4.8, 5.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train incrementally, splitting nodes when Hoeffding bound is satisfied
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new HoeffdingTreeClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict class for new sample
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.1, 1.0 } });
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Mining High-Speed Data Streams", "https://doi.org/10.1145/347090.347107", Year = 2000, Authors = "Pedro Domingos, Geoff Hulten")]
public partial class HoeffdingTreeClassifier<T> : ClassifierBase<T>, IOnlineClassifier<T>
{

    // Its own comment: "Tree-based model - structure cannot be set from flat parameters".
    private readonly HoeffdingTreeOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly Random _random;
    private HoeffdingNode? _root;
    private readonly List<T> _knownClasses;

    /// <summary>
    /// Numeric operations for the static deserializer and the node defaults.
    /// </summary>
    private static readonly INumericOperations<T> StaticOps = MathHelper.GetNumericOperations<T>();

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
        public T SplitThreshold { get; set; } = StaticOps.Zero;
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
        public T Min { get; set; } = StaticOps.MaxValue;
        public T Max { get; set; } = StaticOps.MinValue;
        /// <summary>
        /// When true, Min/Max range is frozen (established during batch pass 1).
        /// Bin assignments use the stable range without shifting boundaries.
        /// </summary>
        public bool RangeFrozen { get; set; }
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

        if (_options.GracePeriod <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "GracePeriod must be > 0.");
        if (_options.NumBins <= 1)
            throw new ArgumentOutOfRangeException(nameof(options), "NumBins must be > 1.");

        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        _knownClasses = new List<T>();
        _root = CreateLeaf(0);
    }

    /// <inheritdoc />

    /// <summary>
    /// Updates the model with a single training sample.
    /// </summary>
    public void PartialFit(Vector<T> features, T label)
    {
        var root = _root ?? throw new InvalidOperationException("Root node has not been initialized.");

        // Initialize feature statistics on first sample
        if (NumFeatures == 0)
        {
            NumFeatures = features.Length;
            InitializeFeatureStats(root, NumFeatures);
        }
        else if (features.Length != NumFeatures)
        {
            throw new ArgumentException(
                $"Expected feature length {NumFeatures} but got {features.Length}.",
                nameof(features));
        }

        // Register new class if needed
        int classIdx = GetOrCreateClassIndex(label);

        // Sort sample to appropriate leaf
        var leaf = SortToLeaf(root, features);

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
        if (features.Rows != labels.Length)
            throw new ArgumentException(
                $"Feature rows ({features.Rows}) must match label count ({labels.Length}).",
                nameof(labels));

        var labelValues = labels.ToArray();
        for (int i = 0; i < features.Rows; i++)
        {
            PartialFit(features.GetRow(i), labelValues[i]);
        }
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Two-pass batch training:
        // Pass 1: Feed all data to establish accurate Min/Max ranges for all features.
        //         This prevents the bin drift problem where early samples get incorrect
        //         bin assignments because the range hasn't stabilized yet.
        // Pass 2: Reset statistics and re-feed data with the correct, stable bin ranges.
        //         Then force splits using greedy criterion (no Hoeffding bound needed
        //         since we've seen all the data).

        // Pass 1: Establish ranges
        PartialFit(x, y);

        // Pass 2: Reset and re-fit with stable ranges (preserve the established Min/Max)
        if (_root != null && _root.FeatureStatistics != null)
        {
            // Save the established ranges
            var savedRanges = new (T Min, T Max)[NumFeatures];
            for (int f = 0; f < NumFeatures; f++)
            {
                var stats = _root.FeatureStatistics[f];
                savedRanges[f] = (stats.Min, stats.Max);
            }

            // Reset the tree but keep known classes
            var savedClasses = new List<T>(_knownClasses);
            _root = CreateLeaf(0);
            SamplesSeen = 0;

            // Re-initialize with saved ranges and freeze them so bin assignment is
            // consistent for ALL samples in pass 2 (no range drift).
            InitializeFeatureStats(_root, NumFeatures);
            var featureStats = _root.FeatureStatistics ?? throw new InvalidOperationException("Feature statistics not initialized.");
            for (int f = 0; f < NumFeatures; f++)
            {
                var stats = featureStats[f];
                stats.Min = savedRanges[f].Min;
                stats.Max = savedRanges[f].Max;
                stats.RangeFrozen = true;
            }

            // Restore known classes
            _knownClasses.Clear();
            _knownClasses.AddRange(savedClasses);
            NumClasses = _knownClasses.Count;
            ClassLabels = Vector<T>.Wrap(_knownClasses.ToArray());

            // Re-feed all data with stable bin ranges
            PartialFit(x, y);
        }

        // Force splits on all leaves using greedy criterion
        ForceLeafSplits(_root);
    }

    /// <summary>
    /// Forces greedy splits on all leaves in the tree. Called by ensemble methods
    /// (AdaptiveRandomForest) after batch training to ensure trees that received
    /// fewer than GracePeriod samples can still make informed predictions.
    /// </summary>
    internal void ForceBatchSplits()
    {
        ForceLeafSplits(_root);
    }

    /// <summary>
    /// Recursively attempts splits on all leaf nodes using a greedy criterion
    /// (best split with positive information gain), bypassing the Hoeffding bound.
    /// Called after batch training when no more data is expected. The Hoeffding bound
    /// provides statistical guarantees for streaming, but after batch training the
    /// entire dataset has been seen and the best split IS the best split — no need
    /// to wait for more evidence.
    /// </summary>
    private void ForceLeafSplits(HoeffdingNode? node)
    {
        if (node is null) return;

        if (node.IsLeaf && node.TotalCount > 0)
        {
            ForceBestSplit(node);
            if (!node.IsLeaf)
            {
                ForceLeafSplits(node.Left);
                ForceLeafSplits(node.Right);
            }
        }
        else
        {
            ForceLeafSplits(node.Left);
            ForceLeafSplits(node.Right);
        }
    }

    /// <summary>
    /// Performs the best split on a leaf node if it has positive information gain,
    /// without requiring the Hoeffding bound to be satisfied.
    /// </summary>
    private void ForceBestSplit(HoeffdingNode leaf)
    {
        if (_options.MaxDepth > 0 && leaf.Depth >= _options.MaxDepth)
            return;

        if (leaf.FeatureStatistics is null || NumFeatures == 0)
            return;

        var splitCandidates = new List<(int Feature, T Threshold, T Gain)>();
        T currentEntropy = CalculateEntropy(leaf.ClassCounts, leaf.TotalCount);

        for (int f = 0; f < NumFeatures; f++)
        {
            var stats = leaf.FeatureStatistics[f];
            T range = NumOps.Subtract(stats.Max, stats.Min);
            if (NumOps.Compare(NumOps.Abs(range), NumOps.FromDouble(1e-10)) < 0) continue;

            for (int b = 1; b < _options.NumBins; b++)
            {
                T threshold = NumOps.Add(stats.Min,
                    NumOps.Divide(NumOps.Multiply(range, NumOps.FromDouble(b)), NumOps.FromDouble(_options.NumBins)));
                T gain = CalculateInformationGain(leaf, f, threshold, currentEntropy);
                splitCandidates.Add((f, threshold, gain));
            }
        }

        if (splitCandidates.Count == 0) return;

        splitCandidates.Sort((a, b) => NumOps.Compare(b.Gain, a.Gain));
        var best = splitCandidates[0];

        // Split if the best candidate has positive information gain
        if (NumOps.Compare(best.Gain, NumOps.FromDouble(1e-10)) > 0)
        {
            PerformSplit(leaf, best.Feature, best.Threshold);
        }
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (NumFeatures > 0 && input.Columns != NumFeatures)
        {
            throw new ArgumentException(
                $"Expected feature length {NumFeatures} but got {input.Columns}.",
                nameof(input));
        }

        var predictions = new T[input.Rows];

        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }

        return Vector<T>.Wrap(predictions);
    }

    private T PredictSingle(Vector<T> features)
    {
        if (_root is null || !IsWarm)
        {
            if (_knownClasses.Count == 0)
                throw new InvalidOperationException("No known classes available. The classifier must be trained first.");
            return _knownClasses[0];
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

        T value = features[node.SplitFeature];
        if (NumOps.Compare(value, node.SplitThreshold) <= 0)
        {
            var left = node.Left ?? throw new InvalidOperationException("Left child node has not been initialized.");
            return SortToLeaf(left, features);
        }
        else
        {
            var right = node.Right ?? throw new InvalidOperationException("Right child node has not been initialized.");
            return SortToLeaf(right, features);
        }
    }

    private T GetMajorityClass(HoeffdingNode node)
    {
        if (node.ClassCounts.Count == 0)
        {
            if (_knownClasses.Count == 0)
                throw new InvalidOperationException("No known classes available. The classifier must be trained first.");
            return _knownClasses[0];
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
        if (majorityIdx >= _knownClasses.Count)
            throw new InvalidOperationException("Majority class index is out of range. The classifier may not be properly trained.");
        return _knownClasses[majorityIdx];
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
        // Known classes are stored in first-seen order; equality test against all of them at once
        if (_knownClasses.Count > 0)
        {
            var matches = Engine.TensorEquals(Vector<T>.Wrap(_knownClasses.ToArray()).AsTensor(), label);
            if (NumOps.Compare(Engine.TensorSum(matches), NumOps.Zero) != 0)
            {
                return Engine.TensorArgMax(matches.Reshape(1, _knownClasses.Count), 1).ToArray()[0];
            }
        }

        _knownClasses.Add(label);
        NumClasses = _knownClasses.Count;
        ClassLabels = Vector<T>.Wrap(_knownClasses.ToArray());
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

        var featureStatistics = leaf.FeatureStatistics ?? throw new InvalidOperationException("FeatureStatistics has not been initialized.");
        for (int f = 0; f < features.Length; f++)
        {
            T value = features[f];
            var stats = featureStatistics[f];

            // Update min/max range — used for BOTH bin assignment and split threshold mapping.
            // When RangeFrozen is true (batch pass 2), the range was established in pass 1
            // and must not change, ensuring all samples use consistent bin boundaries.
            if (!stats.RangeFrozen)
            {
                stats.Min = NumOps.Compare(value, stats.Min) < 0 ? value : stats.Min;
                stats.Max = NumOps.Compare(value, stats.Max) > 0 ? value : stats.Max;
            }

            var binsByClass = stats.BinsByClass ?? throw new InvalidOperationException("BinsByClass has not been initialized.");
            if (!binsByClass.ContainsKey(classIdx))
            {
                stats.BinsByClass[classIdx] = new BinStats
                {
                    Counts = new long[_options.NumBins]
                };
            }

            // Assign to bin using the same Min/Max range that CalculateInformationGain uses
            int binIdx = GetBinIndex(value, stats.Min, stats.Max);
            stats.BinsByClass[classIdx].Counts[binIdx]++;
        }
    }

    private int GetBinIndex(T value, T min, T max)
    {
        T range = NumOps.Subtract(max, min);
        if (NumOps.Compare(NumOps.Abs(range), NumOps.FromDouble(1e-10)) < 0)
        {
            return 0;
        }

        // bin = trunc(((value - min) / (max - min)) * (NumBins - 1)), clamped to [0, NumBins - 1]
        T scaled = NumOps.Multiply(NumOps.Divide(NumOps.Subtract(value, min), range), NumOps.FromDouble(_options.NumBins - 1));
        if (NumOps.Compare(scaled, NumOps.Zero) <= 0)
        {
            return 0;
        }

        T last = NumOps.FromDouble(_options.NumBins - 1);
        return NumOps.Compare(scaled, last) >= 0 ? _options.NumBins - 1 : NumOps.ToInt32(NumOps.Floor(scaled));
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
        var splitCandidates = new List<(int Feature, T Threshold, T Gain)>();
        T currentEntropy = CalculateEntropy(leaf.ClassCounts, leaf.TotalCount);

        for (int f = 0; f < NumFeatures; f++)
        {
            var stats = leaf.FeatureStatistics[f];
            T range = NumOps.Subtract(stats.Max, stats.Min);
            if (NumOps.Compare(NumOps.Abs(range), NumOps.FromDouble(1e-10)) < 0) continue;

            // Try bin boundaries as thresholds
            for (int b = 1; b < _options.NumBins; b++)
            {
                T threshold = NumOps.Add(stats.Min,
                    NumOps.Divide(NumOps.Multiply(range, NumOps.FromDouble(b)), NumOps.FromDouble(_options.NumBins)));
                T gain = CalculateInformationGain(leaf, f, threshold, currentEntropy);
                splitCandidates.Add((f, threshold, gain));
            }
        }

        if (splitCandidates.Count < 2)
        {
            return;
        }

        // Sort by gain
        splitCandidates.Sort((a, b) => NumOps.Compare(b.Gain, a.Gain));
        var best = splitCandidates[0];
        var secondBest = splitCandidates[1];

        // Calculate Hoeffding bound
        double r = Math.Log(Math.Max(2, NumClasses)) / Math.Log(2); // Range of information gain
        double epsilon = Math.Sqrt(r * r * Math.Log(1.0 / _options.Delta) / (2.0 * leaf.TotalCount));

        // Check if we should split
        bool shouldSplit = (NumOps.Compare(NumOps.Subtract(best.Gain, secondBest.Gain), NumOps.FromDouble(epsilon)) > 0) ||
                          (epsilon < _options.TieThreshold && NumOps.Compare(best.Gain, secondBest.Gain) > 0);

        if (shouldSplit && NumOps.Compare(best.Gain, NumOps.Zero) > 0)
        {
            PerformSplit(leaf, best.Feature, best.Threshold);
        }
    }

    private T CalculateEntropy(Dictionary<int, long> classCounts, long total)
    {
        if (total == 0) return NumOps.Zero;

        T entropy = NumOps.Zero;
        T logTwo = NumOps.Log(NumOps.FromDouble(2.0));
        T totalCount = NumOps.FromDouble(total);
        foreach (var count in classCounts.Values)
        {
            if (count > 0)
            {
                T p = NumOps.Divide(NumOps.FromDouble(count), totalCount);
                entropy = NumOps.Subtract(entropy, NumOps.Multiply(p, NumOps.Divide(NumOps.Log(p), logTwo)));
            }
        }
        return entropy;
    }

    private T CalculateInformationGain(HoeffdingNode leaf, int feature, T threshold,
        T currentEntropy)
    {
        var leafFeatureStats = leaf.FeatureStatistics ?? throw new InvalidOperationException("FeatureStatistics has not been initialized.");
        var stats = leafFeatureStats[feature];
        var leftCounts = new Dictionary<int, long>();
        var rightCounts = new Dictionary<int, long>();
        long leftTotal = 0, rightTotal = 0;

        // Estimate split using bin statistics
        int splitBin = GetBinIndex(threshold, stats.Min, stats.Max);

        var statsBinsByClass = stats.BinsByClass ?? throw new InvalidOperationException("BinsByClass has not been initialized.");
        foreach (var kvp in statsBinsByClass)
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
            return NumOps.Zero;
        }

        T total = NumOps.FromDouble(leftTotal + rightTotal);
        T leftWeight = NumOps.Divide(NumOps.FromDouble(leftTotal), total);
        T rightWeight = NumOps.Divide(NumOps.FromDouble(rightTotal), total);

        T leftEntropy = CalculateEntropy(leftCounts, leftTotal);
        T rightEntropy = CalculateEntropy(rightCounts, rightTotal);

        return NumOps.Subtract(NumOps.Subtract(currentEntropy, NumOps.Multiply(leftWeight, leftEntropy)),
            NumOps.Multiply(rightWeight, rightEntropy));
    }

    private void PerformSplit(HoeffdingNode leaf, int feature, T threshold)
    {
        leaf.IsLeaf = false;
        leaf.SplitFeature = feature;
        leaf.SplitThreshold = threshold;

        leaf.Left = CreateLeaf(leaf.Depth + 1);
        leaf.Right = CreateLeaf(leaf.Depth + 1);

        InitializeFeatureStats(leaf.Left, NumFeatures);
        InitializeFeatureStats(leaf.Right, NumFeatures);

        // Copy parent's feature ranges to children so they can compute valid bin indices
        // without needing to see data again (batch replay won't feed data past this point).
        if (leaf.FeatureStatistics is not null)
        {
            foreach (var kvp in leaf.FeatureStatistics)
            {
                int f = kvp.Key;
                var parentStats = kvp.Value;
                if (leaf.Left.FeatureStatistics is not null && leaf.Left.FeatureStatistics.ContainsKey(f))
                {
                    leaf.Left.FeatureStatistics[f].Min = parentStats.Min;
                    leaf.Left.FeatureStatistics[f].Max = parentStats.Max;
                    leaf.Left.FeatureStatistics[f].RangeFrozen = parentStats.RangeFrozen;
                }
                if (leaf.Right.FeatureStatistics is not null && leaf.Right.FeatureStatistics.ContainsKey(f))
                {
                    leaf.Right.FeatureStatistics[f].Min = parentStats.Min;
                    leaf.Right.FeatureStatistics[f].Max = parentStats.Max;
                    leaf.Right.FeatureStatistics[f].RangeFrozen = parentStats.RangeFrozen;
                }
            }
        }

        // Distribute parent's class counts to children based on the split.
        // In streaming mode, new samples populate children naturally. In batch mode
        // (after ForceLeafSplits), no more data arrives, so children need initial counts
        // to make predictions. We estimate counts from the bin statistics.
        if (leaf.FeatureStatistics is not null && leaf.FeatureStatistics.ContainsKey(feature))
        {
            var stats = leaf.FeatureStatistics[feature];
            int splitBin = GetBinIndex(threshold, stats.Min, stats.Max);

            if (stats.BinsByClass is not null)
            {
                foreach (var kvp in stats.BinsByClass)
                {
                    int classIdx = kvp.Key;
                    var bins = kvp.Value.Counts;

                    long leftCount = 0, rightCount = 0;
                    for (int b = 0; b <= splitBin && b < bins.Length; b++)
                        leftCount += bins[b];
                    for (int b = splitBin + 1; b < bins.Length; b++)
                        rightCount += bins[b];

                    if (leftCount > 0)
                    {
                        leaf.Left.ClassCounts[classIdx] = leftCount;
                        leaf.Left.TotalCount += leftCount;
                    }
                    if (rightCount > 0)
                    {
                        leaf.Right.ClassCounts[classIdx] = rightCount;
                        leaf.Right.TotalCount += rightCount;
                    }
                }
            }
        }

        // Clear leaf statistics (no longer needed)
        leaf.FeatureStatistics = null;
    }

    /// <summary>
    /// Returns a fresh instance — tree structure cannot be reconstructed from flat parameters.
    /// </summary>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
        => new HoeffdingTreeClassifier<T>(_options);

    private static Dictionary<string, object?> SerializeNode(HoeffdingNode node)
    {
        var dict = new Dictionary<string, object?>
        {
            ["IsLeaf"] = node.IsLeaf,
            ["Depth"] = node.Depth,
            ["TotalCount"] = node.TotalCount,
            ["SplitFeature"] = node.SplitFeature,
            ["SplitThreshold"] = node.SplitThreshold,
            ["ClassCounts"] = node.ClassCounts.ToDictionary(kv => kv.Key.ToString(), kv => (object)kv.Value)
        };

        if (node.FeatureStatistics is not null)
        {
            var fsDict = new Dictionary<string, object?>();
            foreach (var kvp in node.FeatureStatistics)
            {
                var featureDict = new Dictionary<string, object?>
                {
                    ["Min"] = kvp.Value.Min,
                    ["Max"] = kvp.Value.Max,
                    ["RangeFrozen"] = kvp.Value.RangeFrozen
                };

                if (kvp.Value.BinsByClass is not null)
                {
                    var binsDict = new Dictionary<string, object?>();
                    foreach (var bc in kvp.Value.BinsByClass)
                    {
                        binsDict[bc.Key.ToString()] = bc.Value.Counts;
                    }
                    featureDict["BinsByClass"] = binsDict;
                }

                fsDict[kvp.Key.ToString()] = featureDict;
            }
            dict["FeatureStatistics"] = fsDict;
        }

        if (node.Left is not null)
            dict["Left"] = SerializeNode(node.Left);
        if (node.Right is not null)
            dict["Right"] = SerializeNode(node.Right);

        return dict;
    }

    private static HoeffdingNode DeserializeNode(JObject jObj)
    {
        var node = new HoeffdingNode
        {
            IsLeaf = jObj["IsLeaf"]?.ToObject<bool>() ?? true,
            Depth = jObj["Depth"]?.ToObject<int>() ?? 0,
            TotalCount = jObj["TotalCount"]?.ToObject<long>() ?? 0,
            SplitFeature = jObj["SplitFeature"]?.ToObject<int>() ?? -1,
            SplitThreshold = jObj["SplitThreshold"] is JToken threshold ? threshold.ToObject<T>() ?? StaticOps.Zero : StaticOps.Zero
        };

        var ccObj = jObj["ClassCounts"] as JObject;
        if (ccObj is not null)
        {
            foreach (var prop in ccObj.Properties())
            {
                if (int.TryParse(prop.Name, out int classIdx))
                {
                    node.ClassCounts[classIdx] = prop.Value.ToObject<long>();
                }
            }
        }

        if (jObj["FeatureStatistics"] is JObject fsObj)
        {
            node.FeatureStatistics = new Dictionary<int, FeatureStats>();
            foreach (var prop in fsObj.Properties())
            {
                if (int.TryParse(prop.Name, out int featureIdx) && prop.Value is JObject featureObj)
                {
                    var stats = new FeatureStats
                    {
                        Min = featureObj["Min"] is JToken min ? min.ToObject<T>() ?? StaticOps.MaxValue : StaticOps.MaxValue,
                        Max = featureObj["Max"] is JToken max ? max.ToObject<T>() ?? StaticOps.MinValue : StaticOps.MinValue,
                        RangeFrozen = featureObj["RangeFrozen"]?.ToObject<bool>() ?? false
                    };

                    if (featureObj["BinsByClass"] is JObject binsObj)
                    {
                        stats.BinsByClass = new Dictionary<int, BinStats>();
                        foreach (var binProp in binsObj.Properties())
                        {
                            if (int.TryParse(binProp.Name, out int binClassIdx))
                            {
                                stats.BinsByClass[binClassIdx] = new BinStats
                                {
                                    Counts = binProp.Value.ToObject<long[]>() ?? Array.Empty<long>()
                                };
                            }
                        }
                    }

                    node.FeatureStatistics[featureIdx] = stats;
                }
            }
        }

        if (jObj["Left"] is JObject leftObj)
            node.Left = DeserializeNode(leftObj);
        if (jObj["Right"] is JObject rightObj)
            node.Right = DeserializeNode(rightObj);

        if (!node.IsLeaf && node.Left is null && node.Right is null)
            throw new InvalidOperationException(
                $"Deserialization failed: non-leaf node at depth {node.Depth} has no children.");

        return node;
    }
}
