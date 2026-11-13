using System.Text;
using AiDotNet.Attributes;
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
/// <example>
/// <code>
/// // Create Time Series Forest with ensemble of interval-based decision trees
/// var options = new TimeSeriesForestOptions&lt;double&gt;();
///
/// // Prepare time series data: rows are samples, columns are time steps
/// var features = new Matrix&lt;double&gt;(4, 5);
/// features[0, 0] = 1.0; features[0, 1] = 1.2; features[0, 2] = 1.5; features[0, 3] = 1.3; features[0, 4] = 1.1;
/// features[1, 0] = 1.1; features[1, 1] = 1.3; features[1, 2] = 1.4; features[1, 3] = 1.2; features[1, 4] = 1.0;
/// features[2, 0] = 2.0; features[2, 1] = 2.5; features[2, 2] = 2.3; features[2, 3] = 2.8; features[2, 4] = 3.0;
/// features[3, 0] = 2.1; features[3, 1] = 2.4; features[3, 2] = 2.6; features[3, 3] = 2.9; features[3, 4] = 3.1;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 1, 1 });
///
/// // Train: classify using mean, std, and slope features from random intervals
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new TimeSeriesForestClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict class for new time series
/// var newSample = new Matrix&lt;double&gt;(1, 5);
/// newSample[0, 0] = 1.0; newSample[0, 1] = 1.1; newSample[0, 2] = 1.3; newSample[0, 3] = 1.2; newSample[0, 4] = 1.0;
/// var predictions = result.Predict(newSample);
/// // predictions[0] contains the predicted class
/// </code>
/// </example>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Vector<>))]
[ResearchPaper("A Time Series Forest for Classification and Feature Extraction", "https://doi.org/10.1016/j.ins.2013.01.006", Year = 2013, Authors = "Houtao Deng, George Runger, Eugene Tuv, Martyanov Vladimir")]
public partial class TimeSeriesForestClassifier<T> : ClassifierBase<T>, ITimeSeriesClassifier<T>
{

    // Returned the tree count, its own comment calling it "simplified". An honest zero replaces a
    // placeholder.
    private readonly TimeSeriesForestOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
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
        public int[] IntervalStarts { get; set; } = Array.Empty<int>();
        public int[] IntervalEnds { get; set; } = Array.Empty<int>();
        public int[] ChannelIndices { get; set; } = Array.Empty<int>();
        public DecisionTreeNode? Root { get; set; }
    }

    /// <summary>
    /// A node in the decision tree.
    /// </summary>
    private class DecisionTreeNode
    {
        public int FeatureIndex { get; set; } = -1;
#pragma warning disable CS8601, CS8618 // T defaults to default(T) - used with value types
        public T Threshold { get; set; } = default;
#pragma warning restore CS8601, CS8618
        public bool IsLeaf { get; set; }
        public Vector<T>? ClassProbabilities { get; set; }
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
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc />

    /// <summary>
    /// Trains the Time Series Forest on time series sequences.
    /// </summary>
    public void TrainOnSequences(Tensor<T> sequences, Vector<T> labels)
    {
        if (_options.NumTrees <= 0)
        {
            throw new ArgumentException(
                $"NumTrees must be greater than 0, but was {_options.NumTrees}.",
                nameof(_options));
        }

        ValidateSequenceInput(sequences, labels);

        int numSamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        if (seqLen == 0)
            throw new ArgumentException("Sequence length must be greater than 0.", nameof(sequences));
        NumChannels = sequences.Shape.Length > 2 ? sequences.Shape[2] : 1;
        SequenceLength = seqLen;

        ClassLabels = ExtractClassLabels(labels);
        NumClasses = ClassLabels.Length;

        int minInterval = Math.Max(1, Math.Min(seqLen, (int)(seqLen * _options.MinIntervalFraction)));
        int maxInterval = Math.Min(seqLen, Math.Max(minInterval, (int)(seqLen * _options.MaxIntervalFraction)));

        _trees = new List<IntervalTree>(_options.NumTrees);

        // Deng et al. 2013: each tree sees sqrt(m) random intervals, summarized by mean, standard deviation and
        // slope, and is grown with the entropy gain on those 3 sqrt(m) features.
        int intervalsPerTree = Math.Max(1, (int)Math.Sqrt(seqLen));
        var oneHotLabels = OneHotLabels(labels);

        for (int t = 0; t < _options.NumTrees; t++)
        {
            var starts = new int[intervalsPerTree];
            var ends = new int[intervalsPerTree];
            var channels = new int[intervalsPerTree];
            for (int k = 0; k < intervalsPerTree; k++)
            {
                // Randomly select interval and channel
                int intervalLength = _random.Next(minInterval, maxInterval + 1);
                starts[k] = _random.Next(0, seqLen - intervalLength + 1);
                ends[k] = starts[k] + intervalLength;
                channels[k] = _random.Next(0, NumChannels);
            }

            var tree = new IntervalTree { IntervalStarts = starts, IntervalEnds = ends, ChannelIndices = channels };
            var features = TreeFeatures(sequences, tree);
            var rows = new Tensor<int>(Enumerable.Range(0, numSamples).ToArray(), new[] { numSamples });
            tree.Root = BuildTree(features, oneHotLabels, rows, 0);
            _trees.Add(tree);
        }

        _isFitted = true;
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // A matrix is a 2D tensor [samples, sequence_length] of single-channel time series.
        TrainOnSequences(new Tensor<T>(new[] { x.Rows, x.Columns }, x), y);
    }

    /// <summary>
    /// Predicts class labels for time series sequences: the class with the highest mean tree probability.
    /// </summary>
    public Vector<T> PredictSequences(Tensor<T> sequences)
    {
        var probabilities = PredictSequenceProbabilities(sequences);
        var classes = ClassLabels ?? throw new InvalidOperationException("Model has not been fitted.");
        var engine = AiDotNetEngine.Current;
        var rows = new Tensor<T>(new[] { probabilities.Rows, probabilities.Columns }, probabilities);
        return engine.TensorTake(classes.AsTensor(), FirstArgMaxPerRow(rows)).ToVector();
    }

    /// <summary>
    /// Predicts class probabilities for time series sequences.
    /// </summary>
    public Matrix<T> PredictSequenceProbabilities(Tensor<T> sequences)
    {
        ValidateSequenceInput(sequences, null);

        int numSamples = sequences.Shape[0];
        var trees = _trees ?? throw new InvalidOperationException("_trees is null where it is required.");
        var engine = AiDotNetEngine.Current;

        // The forest's probability is the mean of its trees' leaf class distributions (scikit-learn / sktime
        // predict_proba).
        var total = new Tensor<T>(new[] { numSamples, NumClasses });
        foreach (var tree in trees)
        {
            var root = tree.Root ?? throw new InvalidOperationException("tree.Root is null where it is required.");
            total = engine.TensorAdd(total, PredictTree(root, TreeFeatures(sequences, tree)));
        }

        return engine.TensorDivideScalar(total, NumOps.FromDouble(trees.Count)).ToMatrix();
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> input)
    {
        return PredictSequences(new Tensor<T>(new[] { input.Rows, input.Columns }, input));
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

    /// <summary>The number of random intervals each fitted tree summarizes (0 before fitting).</summary>
    internal int IntervalsPerTree => _trees is { Count: > 0 } ? _trees[0].IntervalStarts.Length : 0;

    /// <summary>The 3 features (mean, standard deviation, slope) of every interval of a tree: [samples, 3 * intervals].</summary>
    private Matrix<T> TreeFeatures(Tensor<T> sequences, IntervalTree tree)
    {
        var pieces = new Tensor<T>[tree.IntervalStarts.Length];
        for (int k = 0; k < pieces.Length; k++)
        {
            pieces[k] = ExtractIntervalFeatures(sequences, tree.IntervalStarts[k], tree.IntervalEnds[k], tree.ChannelIndices[k]);
        }

        return (pieces.Length == 1 ? pieces[0] : AiDotNetEngine.Current.TensorConcatenate(pieces, 1)).ToMatrix();
    }

    /// <summary>Mean, population standard deviation and least-squares slope of one interval: [samples, 3].</summary>
    internal Tensor<T> ExtractIntervalFeatures(Tensor<T> sequences, int start, int end, int channel)
    {
        var engine = AiDotNetEngine.Current;
        int numSamples = sequences.Shape[0];
        int seqLen = sequences.Shape[1];
        int actualStart = Math.Max(0, Math.Min(start, seqLen - 1));
        int actualEnd = Math.Max(actualStart + 1, Math.Min(end, seqLen));
        int length = actualEnd - actualStart;

        // By rank, not channel count: a [samples, length, 1] input has one channel and still needs the 3-D slice.
        var values = sequences.Shape.Length > 2
            ? engine.Reshape(
                engine.TensorSlice(sequences, new[] { 0, actualStart, channel }, new[] { numSamples, length, 1 }),
                new[] { numSamples, length })
            : engine.TensorSlice(sequences, new[] { 0, actualStart }, new[] { numSamples, length });
        var shape = new[] { numSamples, length };
        var mean = engine.ReduceMean(values, new[] { 1 }, true);
        var centered = engine.TensorSubtract(values, engine.TensorBroadcastTo(mean, shape));
        var std = engine.TensorSqrt(engine.ReduceMean(engine.TensorMultiply(centered, centered), new[] { 1 }, true));

        // slope = sum((x - xbar) (y - ybar)) / sum((x - xbar)^2) over x = 0..length-1; 0 for a single point.
        Tensor<T> slope;
        if (length > 1)
        {
            var positions = engine.TensorArange<T>(0, length, 1);
            var centeredPositions = engine.TensorSubtractScalar(positions, NumOps.FromDouble((length - 1) / 2.0));
            T spread = engine.TensorSum(engine.TensorMultiply(centeredPositions, centeredPositions));
            slope = engine.TensorDivideScalar(
                engine.TensorMatMul(values, engine.Reshape(centeredPositions, new[] { length, 1 })), spread);
        }
        else
        {
            slope = new Tensor<T>(new[] { numSamples, 1 });
        }

        return engine.TensorConcatenate(new[] { mean, std, slope }, 1);
    }

    /// <summary>The labels as one-hot rows over <see cref="ClassifierBase{T}.ClassLabels"/>: [samples, classes].</summary>
    private Tensor<T> OneHotLabels(Vector<T> labels)
    {
        var engine = AiDotNetEngine.Current;
        var classes = ClassLabels ?? throw new InvalidOperationException("Model has not been fitted.");
        var shape = new[] { labels.Length, classes.Length };
        return engine.TensorEquals(
            engine.TensorBroadcastTo(engine.Reshape(labels.AsTensor(), new[] { labels.Length, 1 }), shape),
            engine.TensorBroadcastTo(engine.Reshape(classes.AsTensor(), new[] { 1, classes.Length }), shape));
    }

    private DecisionTreeNode BuildTree(Matrix<T> features, Tensor<T> oneHot, Tensor<int> rows, int depth)
    {
        var engine = AiDotNetEngine.Current;
        int n = rows.Length;
        var x = engine.TensorGather(new Tensor<T>(new[] { features.Rows, features.Columns }, features), rows, 0);
        var y = engine.TensorGather(oneHot, rows, 0);

        // Check stopping conditions
        if (n < _options.MinSamplesSplit ||
            (_options.MaxDepth > 0 && depth >= _options.MaxDepth) ||
            IsHomogeneous(y))
        {
            return CreateLeafNode(y);
        }

        var (bestFeature, bestThreshold, found) = FindBestSplit(x, y);
        if (!found)
        {
            return CreateLeafNode(y);
        }

        var (left, right) = Partition(x, rows, bestFeature, bestThreshold);
        if (left.Length == 0 || right.Length == 0)
        {
            return CreateLeafNode(y);
        }

        return new DecisionTreeNode
        {
            FeatureIndex = bestFeature,
            Threshold = bestThreshold,
            IsLeaf = false,
            Left = BuildTree(features, oneHot, left, depth + 1),
            Right = BuildTree(features, oneHot, right, depth + 1)
        };
    }

    /// <summary>
    /// A leaf predicting the majority class (ties go to the class seen first) with the node's class frequencies.
    /// </summary>
    private DecisionTreeNode CreateLeafNode(Tensor<T> oneHot)
    {
        var engine = AiDotNetEngine.Current;
        int n = oneHot.Shape[0], classes = oneHot.Shape[1];
        var counts = engine.ReduceSum(oneHot, new[] { 0 }, false);

        // Key = count * (n + 1) + (n - first row of the class): the largest count, then the earliest class.
        var lateness = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorAddScalar(engine.TensorNegate(engine.TensorArange<T>(0, n, 1)), NumOps.FromDouble(n)), new[] { n, 1 }),
            new[] { n, classes });
        var earliness = engine.ReduceMax(engine.TensorMultiply(oneHot, lateness), new[] { 0 }, false);
        var key = engine.TensorAdd(engine.TensorMultiplyScalar(counts, NumOps.FromDouble(n + 1)), earliness);
        int majority = FirstArgMaxPerRow(engine.Reshape(key, new[] { 1, classes })).GetFlat(0);

        return new DecisionTreeNode
        {
            IsLeaf = true,
            FeatureIndex = majority,
            ClassProbabilities = engine.TensorDivideScalar(counts, NumOps.FromDouble(Math.Max(1, n))).ToVector()
        };
    }

    private bool IsHomogeneous(Tensor<T> oneHot)
    {
        if (oneHot.Shape[0] <= 1) return true;
        var engine = AiDotNetEngine.Current;
        var present = engine.TensorGreaterThan(engine.ReduceSum(oneHot, new[] { 0 }, false), NumOps.Zero);
        return NumOps.LessThanOrEquals(engine.TensorSum(present), NumOps.One);
    }

    /// <summary>
    /// The split with the largest entropy gain over every feature and every midpoint between consecutive distinct
    /// sorted values (the first such split on ties, features in order then thresholds ascending).
    /// </summary>
    private (int Feature, T Threshold, bool Found) FindBestSplit(Tensor<T> x, Tensor<T> oneHot)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0], featureCount = x.Shape[1], classes = oneHot.Shape[1];
        int candidates = n - 1;
        if (candidates <= 0) return (-1, NumOps.Zero, false);

        // Sort every feature column; carry the one-hot labels into each column's order: [features, n, classes].
        var (sorted, order) = engine.TensorSort(x, 0, false);
        var sortedT = engine.TensorTranspose(sorted);
        var labelsByFeature = engine.Reshape(
            engine.TensorGather(oneHot, engine.Reshape(engine.TensorTranspose(order), new[] { featureCount * n }), 0),
            new[] { featureCount, n, classes });

        // Left side of candidate i holds the first i + 1 sorted rows.
        var leftCounts = engine.TensorSlice(
            engine.TensorCumSum(labelsByFeature, 1), new[] { 0, 0, 0 }, new[] { featureCount, candidates, classes });
        var totals = engine.ReduceSum(oneHot, new[] { 0 }, false);
        var rightCounts = engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(totals, new[] { 1, 1, classes }), new[] { featureCount, candidates, classes }),
            leftCounts);
        var leftSizes = engine.TensorAddScalar(engine.TensorArange<T>(0, candidates, 1), NumOps.One);
        var rightSizes = engine.TensorSubtract(
            Filled(new[] { candidates }, NumOps.FromDouble(n)), leftSizes);

        T parentEntropy = Entropy(engine.Reshape(totals, new[] { 1, 1, classes }), new Tensor<T>(new[] { NumOps.FromDouble(n) }, new[] { 1 })).GetFlat(0);
        var gain = engine.TensorSubtract(
            Filled(new[] { featureCount, candidates }, parentEntropy),
            engine.TensorAdd(
                engine.TensorMultiplyScalar(engine.TensorMultiply(Entropy(leftCounts, leftSizes), Row(leftSizes, featureCount)), NumOps.FromDouble(1.0 / n)),
                engine.TensorMultiplyScalar(engine.TensorMultiply(Entropy(rightCounts, rightSizes), Row(rightSizes, featureCount)), NumOps.FromDouble(1.0 / n))));

        // A candidate exists only between two distinct values.
        var gaps = engine.TensorSubtract(
            engine.TensorSlice(sortedT, new[] { 0, 1 }, new[] { featureCount, candidates }),
            engine.TensorSlice(sortedT, new[] { 0, 0 }, new[] { featureCount, candidates }));
        var distinct = engine.TensorGreaterEqual(engine.TensorAbs(gaps), Filled(new[] { featureCount, candidates }, NumOps.FromDouble(1e-10)));
        var usable = engine.TensorWhere(distinct, gain, new Tensor<T>(new[] { featureCount, candidates }));

        T best = engine.TensorMaxValue(usable);
        if (!NumOps.GreaterThan(best, NumOps.Zero)) return (-1, NumOps.Zero, false);

        int flat = FirstArgMaxPerRow(engine.Reshape(usable, new[] { 1, featureCount * candidates })).GetFlat(0);
        int feature = flat / candidates, position = flat % candidates;
        var pair = engine.TensorSlice(sortedT, new[] { feature, position }, new[] { 1, 2 });
        // scikit-learn's splitter: the midpoint as a/2 + b/2 (no overflow near the type's maximum), and the lower value
        // when rounding lands the midpoint on the upper one, which would send both sides of the gap left.
        T lower = pair.GetFlat(0), upper = pair.GetFlat(1);
        T half = NumOps.FromDouble(0.5);
        T threshold = NumOps.Add(NumOps.Multiply(lower, half), NumOps.Multiply(upper, half));
        if (!NumOps.LessThan(threshold, upper)) threshold = lower;
        return (feature, threshold, true);
    }

    /// <summary>Entropy in bits of the class counts [.., classes] over sizes broadcast on the candidate axis.</summary>
    private static Tensor<T> Entropy(Tensor<T> counts, Tensor<T> sizes)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int featureCount = counts.Shape[0], candidates = counts.Shape[1], classes = counts.Shape[2];
        var shape = new[] { featureCount, candidates, classes };
        var p = engine.TensorDivide(counts, engine.TensorBroadcastTo(engine.Reshape(sizes, new[] { 1, candidates, 1 }), shape));
        var positive = engine.TensorGreaterThan(p, numOps.Zero);
        var safe = engine.TensorWhere(positive, p, engine.TensorOnesLike(p));
        var terms = engine.TensorMultiply(p, engine.TensorMultiplyScalar(engine.TensorLog(safe), numOps.FromDouble(1.0 / Math.Log(2.0))));
        return engine.TensorNegate(engine.ReduceSum(terms, new[] { 2 }, false));
    }

    private static Tensor<T> Filled(int[] shape, T value)
    {
        var tensor = new Tensor<T>(shape);
        AiDotNetEngine.Current.TensorFill(tensor, value);
        return tensor;
    }

    private static Tensor<T> Row(Tensor<T> values, int rows)
        => AiDotNetEngine.Current.TensorBroadcastTo(
            AiDotNetEngine.Current.Reshape(values, new[] { 1, values.Length }), new[] { rows, values.Length });

    /// <summary>
    /// Splits <paramref name="rows"/> into those whose feature is at most the threshold and the rest, each in its
    /// original order.
    /// </summary>
    private static (Tensor<int> Left, Tensor<int> Right) Partition(Tensor<T> x, Tensor<int> rows, int feature, T threshold)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = rows.Length;
        var column = engine.Reshape(engine.TensorSlice(x, new[] { 0, feature }, new[] { n, 1 }), new[] { n });
        var goesRight = engine.TensorGreaterThan(column, threshold);
        // Counted and keyed in int: T may not hold the row count or n + position exactly (Half is exact only to 2048),
        // which miscounts the right rows and lets a right row's key collide with a left row's.
        var goesRightIndex = goesRight.Cast<int>();
        int rightCount = engine.TensorSum(goesRightIndex);
        var none = new Tensor<int>(new[] { 0 });
        if (rightCount == 0) return (rows, none);
        if (rightCount == n) return (none, rows);

        // key = goesRight * n + position orders the left rows first, then the right rows, each by position.
        var key = engine.TensorAdd(engine.TensorMultiplyScalar(goesRightIndex, n), engine.TensorArange<int>(0, n, 1));
        var order = engine.TensorArgsort(key, 0, false);
        var arranged = engine.TensorGather(rows, order, 0);
        int leftCount = n - rightCount;
        return (
            engine.TensorSlice(arranged, new[] { 0 }, new[] { leftCount }),
            engine.TensorSlice(arranged, new[] { leftCount }, new[] { rightCount }));
    }

    /// <summary>Routes every row through the tree and returns each row's leaf class distribution: [rows, classes].</summary>
    private Tensor<T> PredictTree(DecisionTreeNode node, Matrix<T> features)
    {
        var engine = AiDotNetEngine.Current;
        int n = features.Rows;
        var x = new Tensor<T>(new[] { n, features.Columns }, features);
        var blocks = new List<Tensor<T>>();
        var positions = new List<Tensor<int>>();
        Route(node, x, new Tensor<int>(Enumerable.Range(0, n).ToArray(), new[] { n }), blocks, positions);

        var leafRows = engine.TensorConcatenate(blocks.ToArray(), 0);
        var placed = positions.Count == 1 ? positions[0] : engine.TensorConcatenate(positions.ToArray(), 0);
        return engine.TensorGather(leafRows, engine.TensorArgsort(placed, 0, false), 0);
    }

    private void Route(DecisionTreeNode node, Tensor<T> x, Tensor<int> rows, List<Tensor<T>> blocks, List<Tensor<int>> positions)
    {
        if (rows.Length == 0) return;
        var engine = AiDotNetEngine.Current;
        if (node.IsLeaf)
        {
            var probabilities = node.ClassProbabilities ?? throw new InvalidOperationException("A leaf has no class distribution.");
            blocks.Add(engine.TensorBroadcastTo(
                engine.Reshape(probabilities.AsTensor(), new[] { 1, probabilities.Length }), new[] { rows.Length, probabilities.Length }));
            positions.Add(rows);
            return;
        }

        var (left, right) = Partition(engine.TensorGather(x, rows, 0), rows, node.FeatureIndex, node.Threshold);
        Route(node.Left ?? throw new InvalidOperationException("node.Left is null where it is required."), x, left, blocks, positions);
        Route(node.Right ?? throw new InvalidOperationException("node.Right is null where it is required."), x, right, blocks, positions);
    }

    /// <summary>The index of each row's first maximum: the earliest maximal entry carries the largest weight.</summary>
    private static Tensor<int> FirstArgMaxPerRow(Tensor<T> rows)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int count = rows.Shape[0], width = rows.Shape[1];
        var shape = new[] { count, width };
        var rowMax = engine.TensorBroadcastTo(engine.ReduceMax(rows, new[] { 1 }, true), shape);
        var descending = engine.TensorAddScalar(engine.TensorNegate(engine.TensorArange<T>(0, width, 1)), numOps.FromDouble(width));
        var weights = engine.TensorBroadcastTo(engine.Reshape(descending, new[] { 1, width }), shape);
        return engine.TensorArgMax(engine.TensorMultiply(engine.TensorEquals(rows, rowMax), weights), 1);
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

        if (sequences.Shape[1] == 0)
        {
            throw new ArgumentException("Sequence length must be greater than 0.", nameof(sequences));
        }

        if (labels is not null && labels.Length != sequences.Shape[0])
        {
            throw new ArgumentException("Number of labels must match number of samples.");
        }
    }
}
