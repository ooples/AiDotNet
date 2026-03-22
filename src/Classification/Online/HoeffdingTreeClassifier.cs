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
/// var classifier = new HoeffdingTreeClassifier&lt;double&gt;(options);
///
/// // Prepare streaming data batch
/// var features = Matrix&lt;double&gt;.Build.Dense(6, 2, new double[] {
///     1.0, 1.1,  1.2, 0.9,  0.8, 1.0,
///     5.0, 5.1,  5.2, 4.9,  4.8, 5.0 });
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train incrementally, splitting nodes when Hoeffding bound is satisfied
/// classifier.Train(features, labels);
///
/// // Predict class for new sample
/// var newSample = Matrix&lt;double&gt;.Build.Dense(1, 2, new double[] { 1.1, 1.0 });
/// var prediction = classifier.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Mining High-Speed Data Streams", "https://doi.org/10.1145/347090.347107", Year = 2000, Authors = "Pedro Domingos, Geoff Hulten")]
public class HoeffdingTreeClassifier<T> : ClassifierBase<T>, IOnlineClassifier<T>
{
    private readonly HoeffdingTreeOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
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

        _random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
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
            var savedRanges = new (double Min, double Max)[NumFeatures];
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
            for (int f = 0; f < NumFeatures; f++)
            {
                var stats = _root.FeatureStatistics![f];
                stats.Min = savedRanges[f].Min;
                stats.Max = savedRanges[f].Max;
                stats.RangeFrozen = true;
            }

            // Restore known classes
            _knownClasses.Clear();
            _knownClasses.AddRange(savedClasses);
            NumClasses = _knownClasses.Count;
            ClassLabels = new Vector<T>(_knownClasses.ToArray());

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

        var splitCandidates = new List<(int Feature, double Threshold, double Gain)>();
        double currentEntropy = CalculateEntropy(leaf.ClassCounts, leaf.TotalCount);

        for (int f = 0; f < NumFeatures; f++)
        {
            var stats = leaf.FeatureStatistics[f];
            if (Math.Abs(stats.Max - stats.Min) < 1e-10) continue;

            for (int b = 1; b < _options.NumBins; b++)
            {
                double threshold = stats.Min + (stats.Max - stats.Min) * b / _options.NumBins;
                double gain = CalculateInformationGain(leaf, f, threshold, currentEntropy);
                splitCandidates.Add((f, threshold, gain));
            }
        }

        if (splitCandidates.Count == 0) return;

        splitCandidates.Sort((a, b) => b.Gain.CompareTo(a.Gain));
        var best = splitCandidates[0];

        // Split if the best candidate has positive information gain
        if (best.Gain > 1e-10)
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

        double value = NumOps.ToDouble(features[node.SplitFeature]);
        if (value <= node.SplitThreshold)
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

        var featureStatistics = leaf.FeatureStatistics ?? throw new InvalidOperationException("FeatureStatistics has not been initialized.");
        for (int f = 0; f < features.Length; f++)
        {
            double value = NumOps.ToDouble(features[f]);
            var stats = featureStatistics[f];

            // Update min/max range — used for BOTH bin assignment and split threshold mapping.
            // When RangeFrozen is true (batch pass 2), the range was established in pass 1
            // and must not change, ensuring all samples use consistent bin boundaries.
            if (!stats.RangeFrozen)
            {
                stats.Min = Math.Min(stats.Min, value);
                stats.Max = Math.Max(stats.Max, value);
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
        // Return a cold instance to avoid inconsistent state and shared mutable references.
        // Tree structure cannot be set from flat parameters - deep cloning would be needed
        // to properly copy the tree, which is non-trivial.
        return new HoeffdingTreeClassifier<T>(_options);
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

    /// <summary>
    /// Serializes the trained Hoeffding tree including all nodes and statistics.
    /// </summary>
    public override byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "NumClasses", NumClasses },
            { "NumFeatures", NumFeatures },
            { "TaskType", (int)TaskType },
            { "ClassLabels", ClassLabels?.ToArray() ?? Array.Empty<T>() },
            { "SamplesSeen", SamplesSeen }
        };

        var knownClassValues = new double[_knownClasses.Count];
        for (int i = 0; i < _knownClasses.Count; i++)
            knownClassValues[i] = NumOps.ToDouble(_knownClasses[i]);
        modelData["KnownClasses"] = knownClassValues;

        if (_root is not null)
        {
            modelData["Root"] = SerializeNode(_root);
        }

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));
        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    /// <summary>
    /// Deserializes the trained Hoeffding tree including all nodes and statistics.
    /// </summary>
    public override void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);
        if (modelMetadata?.ModelData is null)
            throw new InvalidOperationException("Deserialization failed: invalid model data.");

        var dataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var dataObj = JsonConvert.DeserializeObject<JObject>(dataString);
        if (dataObj is null)
            throw new InvalidOperationException("Deserialization failed: invalid model data.");

        NumClasses = dataObj["NumClasses"]?.ToObject<int>() ?? 0;
        NumFeatures = dataObj["NumFeatures"]?.ToObject<int>() ?? 0;
        TaskType = (ClassificationTaskType)(dataObj["TaskType"]?.ToObject<int>() ?? 0);
        SamplesSeen = dataObj["SamplesSeen"]?.ToObject<long>() ?? 0;

        var classLabelsToken = dataObj["ClassLabels"];
        if (classLabelsToken is not null)
        {
            var arr = classLabelsToken.ToObject<double[]>() ?? Array.Empty<double>();
            if (arr.Length > 0)
            {
                ClassLabels = new Vector<T>(arr.Length);
                for (int i = 0; i < arr.Length; i++)
                    ClassLabels[i] = NumOps.FromDouble(arr[i]);
            }
        }

        _knownClasses.Clear();
        var knownArr = dataObj["KnownClasses"]?.ToObject<double[]>();
        if (knownArr is not null)
        {
            foreach (var val in knownArr)
                _knownClasses.Add(NumOps.FromDouble(val));
        }

        if (dataObj["Root"] is JObject rootObj)
        {
            _root = DeserializeNode(rootObj);
        }
    }

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
                    ["Max"] = kvp.Value.Max
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
            SplitThreshold = jObj["SplitThreshold"]?.ToObject<double>() ?? 0
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
                        Min = featureObj["Min"]?.ToObject<double>() ?? double.MaxValue,
                        Max = featureObj["Max"]?.ToObject<double>() ?? double.MinValue
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
