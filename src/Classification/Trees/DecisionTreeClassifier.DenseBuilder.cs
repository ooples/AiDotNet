using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Classification.Trees;

/// <summary>
/// Tree construction on ranked, column-major training data (<see cref="DenseTreeData{T}"/>).
/// </summary>
/// <remarks>
/// <para>
/// Each considered feature's node rows are sorted as one 64-bit key per row, the rank above the class index, so a
/// large node sorts by radix on the rank bits alone and a small one by <see cref="Array.Sort{T}(T[], int, int)"/>.
/// One sweep over the sorted keys collects, per distinct value, the class counts at or below it (integers, so no
/// count saturates a narrow T such as Half).
/// </para>
/// <para>
/// Everything that touches values is engine work in T, batched over the candidates of every considered feature:
/// thresholds (midpoints between consecutive distinct values), the impurity of both sides, the weighted gain, and the
/// first best candidate (features in order, thresholds ascending, as scikit-learn's strict comparison). Class
/// frequencies are formed from the integer counts by <see cref="ScaledCounts"/>, which keeps every intermediate
/// representable in T.
/// </para>
/// </remarks>
public partial class DecisionTreeClassifier<T>
{
    /// <summary>Nodes below this many rows sort by comparison; larger ones by radix.</summary>
    private const int RadixSortMinRows = 256;

    /// <summary>Integers up to 2^11 are exact in every supported T, Half included.</summary>
    private const int ExactIntegerBits = 11;

    /// <summary>scikit-learn's tolerance on the minimum impurity decrease (<c>np.finfo(np.double).eps</c>).</summary>
    private const double ImpurityEpsilon = 2.220446049250313e-16;

    [Scratch]
    private DenseTreeData<T> _denseData = EmptyDenseData;

    /// <summary>The number of rows (repeats counted) the tree is grown on; weights each node's impurity decrease.</summary>
    [Scratch]
    private int _denseSampleCount;

    /// <summary>Each training row's index into <see cref="ClassifierBase{T}.ClassLabels"/>, by data row.</summary>
    [Scratch]
    private int[] _denseClasses = new int[0];

    /// <summary>Data rows of the training sample, repeats allowed; each node owns a contiguous range.</summary>
    [Scratch]
    private int[] _denseRows = new int[0];

    /// <summary>Scratch for the split search and the partition, sized to the training sample.</summary>
    [Scratch]
    private ulong[] _sortKeys = new ulong[0];
    [Scratch]
    private ulong[] _radixScratch = new ulong[0];
    [Scratch]
    private int[] _partitionScratch = new int[0];
    [Scratch]
    private int[] _featurePool = new int[0];
    [Scratch]
    private int[] _selectedFeatures = new int[0];

    private static readonly DenseTreeData<T> EmptyDenseData = new(new Matrix<T>(0, 0));

    /// <summary>Ranks <paramref name="x"/> for <see cref="BuildDense"/>.</summary>
    internal static DenseTreeData<T> CreateDenseData(Matrix<T> x) => new(x);

    /// <summary>
    /// Grows the tree from the data rows <paramref name="rows"/> of <paramref name="data"/>, labelled by
    /// <paramref name="y"/> (indexed by data row). A row listed twice counts twice, as in a bootstrap sample.
    /// </summary>
    private DecisionNode<T> BuildDense(DenseTreeData<T> data, Vector<T> y, int[] rows)
    {
        int count = rows.Length;
        _denseData = data;
        _denseClasses = new int[data.Rows];
        _denseRows = (int[])rows.Clone();
        _sortKeys = new ulong[count];
        _radixScratch = new ulong[count];
        _partitionScratch = new int[count];
        _featurePool = new int[NumFeatures];
        _selectedFeatures = new int[NumFeatures];
        _denseSampleCount = count;

        // Each data row's index into the class labels: the column where its label equals a class. Rows outside the
        // sample are never read, so a label absent from the sample's classes does no harm there.
        if (data.Rows > 0)
        {
            var engine = AiDotNetEngine.Current;
            var classes = ClassLabels ?? throw new InvalidOperationException("ClassLabels has not been initialized.");
            var shape = new[] { data.Rows, NumClasses };
            var matches = engine.TensorEquals(
                engine.TensorBroadcastTo(engine.Reshape(y.AsTensor(), new[] { data.Rows, 1 }), shape),
                engine.TensorBroadcastTo(engine.Reshape(classes.AsTensor(), new[] { 1, NumClasses }), shape));
            _denseClasses = engine.TensorArgMax(matches, 1).ToArray();
        }

        try
        {
            return BuildDenseNode(0, count, 0);
        }
        finally
        {
            // The buffers are training scratch, not model state.
            _denseData = EmptyDenseData;
            _denseSampleCount = 0;
            _denseClasses = new int[0];
            _denseRows = new int[0];
            _sortKeys = new ulong[0];
            _radixScratch = new ulong[0];
            _partitionScratch = new int[0];
            _featurePool = new int[0];
            _selectedFeatures = new int[0];
        }
    }

    private DecisionNode<T> BuildDenseNode(int start, int count, int depth)
    {
        var counts = new int[NumClasses];
        for (int r = start; r < start + count; r++)
            counts[_denseClasses[_denseRows[r]]]++;

        if (ShouldStopDense(counts, count, depth))
            return CreateDenseLeaf(counts, count);

        var split = FindBestDenseSplit(start, count, counts);

        // scikit-learn (_tree.pyx): the best split is taken even when it does not lower the impurity (XOR has no
        // first split that does), and the node becomes a leaf only when its impurity decrease weighted by its share
        // of the samples, N_t / N * gain, plus eps falls below min_impurity_decrease. Not "< MinImpurityDecrease":
        // a NaN gain must make a leaf, and only the negated form does.
        if (!split.Found)
            return CreateDenseLeaf(counts, count);

        T weightedGain = NumOps.Multiply(Fraction(count, _denseSampleCount), split.Gain);
        if (!NumOps.GreaterThanOrEquals(
                NumOps.Add(weightedGain, NumOps.FromDouble(ImpurityEpsilon)), NumOps.FromDouble(Options.MinImpurityDecrease)))
            return CreateDenseLeaf(counts, count);

        // Stable partition: the left child keeps rows whose rank is <= the threshold's, in their original order.
        var ranks = _denseData.Ranks;
        int column = split.Feature * _denseData.Rows;
        int leftCount = 0;
        int rightCount = 0;
        for (int r = start; r < start + count; r++)
        {
            int row = _denseRows[r];
            if (ranks[column + row] <= split.ThresholdRank)
                _denseRows[start + leftCount++] = row;
            else
                _partitionScratch[rightCount++] = row;
        }

        Array.Copy(_partitionScratch, 0, _denseRows, start + leftCount, rightCount);

        if (leftCount < Options.MinSamplesLeaf || rightCount < Options.MinSamplesLeaf)
            return CreateDenseLeaf(counts, count);

        UpdateFeatureImportances(split.Feature, weightedGain);

        var leftChild = BuildDenseNode(start, leftCount, depth + 1);
        var rightChild = BuildDenseNode(start + leftCount, rightCount, depth + 1);

        return new DecisionNode<T>
        {
            FeatureIndex = split.Feature,
            Threshold = split.Threshold,
            Left = leftChild,
            Right = rightChild,
            IsLeaf = false,
            NumSamples = count
        };
    }

    private bool ShouldStopDense(int[] counts, int count, int depth)
    {
        if (Options.MaxDepth.HasValue && depth >= Options.MaxDepth.Value)
            return true;

        if (count < Options.MinSamplesSplit)
            return true;

        // A pure node: every row in one class.
        for (int c = 0; c < counts.Length; c++)
            if (counts[c] == count)
                return true;

        return false;
    }

    private DecisionNode<T> CreateDenseLeaf(int[] counts, int count)
    {
        var engine = AiDotNetEngine.Current;
        int shift = ScaleShift(count);
        var classCounts = new Tensor<int>(counts, new[] { counts.Length });
        var probabilities = engine.TensorDivideScalar(ScaledCounts(classCounts, shift), ScaledCount(count, shift)).ToVector();

        // The first class with the most rows, as numpy's argmax.
        int predictedClass = Array.IndexOf(counts, counts.Max());

        return new DecisionNode<T>
        {
            IsLeaf = true,
            ClassProbabilities = probabilities,
            PredictedClass = predictedClass,
            NumSamples = count
        };
    }

    /// <summary>
    /// The features one split considers: all of them, or <see cref="DecisionTreeClassifierOptions{T}.MaxFeatures"/>
    /// drawn without replacement, one <see cref="Random.Next(int)"/> over the remaining features per draw.
    /// </summary>
    private int SelectDenseFeatures()
    {
        if (!Options.MaxFeatures.HasValue || Options.MaxFeatures.Value >= NumFeatures)
        {
            for (int f = 0; f < NumFeatures; f++) _selectedFeatures[f] = f;
            return NumFeatures;
        }

        var random = _random ?? throw new InvalidOperationException("_random has not been initialized.");
        for (int f = 0; f < NumFeatures; f++) _featurePool[f] = f;
        int remaining = NumFeatures;
        int selected = Options.MaxFeatures.Value;
        for (int i = 0; i < selected; i++)
        {
            int idx = random.Next(remaining);
            _selectedFeatures[i] = _featurePool[idx];
            Array.Copy(_featurePool, idx + 1, _featurePool, idx, remaining - idx - 1);
            remaining--;
        }

        return selected;
    }

    /// <summary>
    /// The best split of the node's rows. One sort per considered feature and one integer sweep collect, for each
    /// distinct value, the class counts at or below it; the thresholds, impurities and gains of every candidate of
    /// every feature are then one batch of engine work in T.
    /// </summary>
    private (int Feature, T Threshold, int ThresholdRank, T Gain, bool Found) FindBestDenseSplit(
        int start, int count, int[] totalCounts)
    {
        int classes = NumClasses;
        int featureCount = SelectDenseFeatures();
        var ranks = _denseData.Ranks;

        // Per candidate: its feature, its lower rank, where the values on either side sit in DistinctValues, and the
        // class counts of the rows ranked at most the lower value. Candidates are scored in batches of bounded size so
        // the class-count tables never cover every threshold of every feature at once (rows x features x classes).
        var batch = new CandidateBatch(classes);
        var running = new int[classes];
        var runRanks = new List<int>();
        var runCounts = new List<int>();
        var best = (Feature: -1, Threshold: NumOps.Zero, ThresholdRank: 0, Gain: NumOps.Zero, Found: false);

        for (int s = 0; s < featureCount; s++)
        {
            int feature = _selectedFeatures[s];
            int column = feature * _denseData.Rows;
            for (int r = 0; r < count; r++)
            {
                int row = _denseRows[start + r];
                _sortKeys[r] = ((ulong)(uint)ranks[column + row] << 32) | (uint)_denseClasses[row];
            }

            SortByRank(count);

            // One run per distinct value: its rank and the cumulative class counts after it.
            Array.Clear(running, 0, classes);
            runRanks.Clear();
            runCounts.Clear();
            int nanRank = _denseData.NanRank[feature];
            bool hasNan = false;
            for (int i = 0; i < count; )
            {
                int rank = (int)(_sortKeys[i] >> 32);
                hasNan |= rank == nanRank;
                for (; i < count && (int)(_sortKeys[i] >> 32) == rank; i++)
                    running[(int)(uint)_sortKeys[i]]++;
                runRanks.Add(rank);
                runCounts.AddRange(running);
            }

            // A NaN compares false with every threshold, so no row ever moves left: the feature cannot split here.
            if (hasNan)
                continue;

            int offset = _denseData.ValueOffsets[feature];
            for (int k = 0; k + 1 < runRanks.Count; k++)
            {
                batch.Feature.Add(feature);
                batch.LowerRank.Add(runRanks[k]);
                batch.LowerValue.Add(offset + runRanks[k]);
                batch.UpperValue.Add(offset + runRanks[k + 1]);
                batch.LeftCounts.AddRange(runCounts.GetRange(k * classes, classes));
            }

            if (batch.LeftCounts.Count >= CandidateCountBudget)
                best = Better(best, ScoreCandidates(batch, count, totalCounts));
        }

        return Better(best, ScoreCandidates(batch, count, totalCounts));
    }

    // Class counts (ints) one candidate batch holds before it is scored: 4 MB, whatever the rows, features and classes.
    private const int CandidateCountBudget = 1 << 20;

    /// <summary>Split candidates gathered for one scoring batch.</summary>
    private sealed class CandidateBatch
    {
        internal CandidateBatch(int classes) => Classes = classes;
        internal int Classes { get; }
        internal List<int> Feature { get; } = new List<int>();
        internal List<int> LowerRank { get; } = new List<int>();
        internal List<int> LowerValue { get; } = new List<int>();
        internal List<int> UpperValue { get; } = new List<int>();
        internal List<int> LeftCounts { get; } = new List<int>();

        internal void Clear()
        {
            Feature.Clear();
            LowerRank.Clear();
            LowerValue.Clear();
            UpperValue.Clear();
            LeftCounts.Clear();
        }
    }

    // The earlier batch keeps a tie, so the first candidate with the largest gain wins as it did in one batch.
    private (int Feature, T Threshold, int ThresholdRank, T Gain, bool Found) Better(
        (int Feature, T Threshold, int ThresholdRank, T Gain, bool Found) current,
        (int Feature, T Threshold, int ThresholdRank, T Gain, bool Found) next)
        => next.Found && (!current.Found || NumOps.GreaterThan(next.Gain, current.Gain)) ? next : current;

    /// <summary>
    /// The best candidate of <paramref name="batch"/> (its first with the largest gain), scored as one batch of engine
    /// work in T; the batch is cleared afterwards.
    /// </summary>
    private (int Feature, T Threshold, int ThresholdRank, T Gain, bool Found) ScoreCandidates(
        CandidateBatch batch, int count, int[] totalCounts)
    {
        int classes = batch.Classes;
        int candidates = batch.Feature.Count;
        if (candidates == 0)
            return (-1, NumOps.Zero, 0, NumOps.Zero, false);

        var engine = AiDotNetEngine.Current;
        var countShape = new[] { candidates, classes };

        // threshold = lower/2 + upper/2 (the midpoint without overflowing a narrow T). When rounding lands it on the
        // upper value (adjacent values in a narrow T) it is the lower value instead, as scikit-learn's splitter does,
        // so the split still separates the two runs and the rows at most the lower value are exactly the left side.
        var lower = engine.TensorTake(_denseData.DistinctValues, new Tensor<int>(batch.LowerValue.ToArray(), new[] { candidates }));
        var upper = engine.TensorTake(_denseData.DistinctValues, new Tensor<int>(batch.UpperValue.ToArray(), new[] { candidates }));
        T half = NumOps.FromDouble(0.5);
        var midpoints = engine.TensorAdd(engine.TensorMultiplyScalar(lower, half), engine.TensorMultiplyScalar(upper, half));
        var thresholds = engine.TensorWhere(engine.TensorGreaterEqual(midpoints, upper), lower, midpoints);

        var left = new Tensor<int>(batch.LeftCounts.ToArray(), countShape);
        var totals = engine.TensorBroadcastTo(new Tensor<int>((int[])totalCounts.Clone(), new[] { 1, classes }), countShape);
        var right = engine.TensorSubtract(totals, left);
        var leftSizes = engine.ReduceSum(left, new[] { 1 }, false);
        var rightSizes = engine.TensorSubtract(
            new Tensor<int>(Enumerable.Repeat(count, candidates).ToArray(), new[] { candidates }), leftSizes);

        // Both children must keep MinSamplesLeaf rows.
        var minLeaf = new Tensor<int>(Enumerable.Repeat(Options.MinSamplesLeaf, candidates).ToArray(), new[] { candidates });
        var allowed = engine.TensorMultiply(
            engine.TensorGreaterEqual(leftSizes, minLeaf), engine.TensorGreaterEqual(rightSizes, minLeaf));

        // gain = impurity(node) - (n_left / n) impurity(left) - (n_right / n) impurity(right)
        int shift = ScaleShift(count);
        T nodeImpurity = Impurity(new Tensor<int>((int[])totalCounts.Clone(), new[] { 1, classes }),
            new Tensor<int>(new[] { count }, new[] { 1 }), shift).GetFlat(0);
        T scaledTotal = ScaledCount(count, shift);
        var leftShare = engine.TensorDivideScalar(ScaledCounts(leftSizes, shift), scaledTotal);
        var rightShare = engine.TensorDivideScalar(ScaledCounts(rightSizes, shift), scaledTotal);
        var weighted = engine.TensorAdd(
            engine.TensorMultiply(leftShare, Impurity(left, leftSizes, shift)),
            engine.TensorMultiply(rightShare, Impurity(right, rightSizes, shift)));
        var gains = engine.TensorSubtract(engine.TensorFullLike(weighted, nodeImpurity), weighted);

        // A NaN gain never wins (scikit-learn's ">" is false for it); neither does a disallowed candidate.
        var notNan = engine.TensorWhere(engine.TensorIsNan(gains), engine.TensorZerosLike(gains), engine.TensorOnesLike(gains));
        var usable = engine.TensorMultiply(allowed, notNan.Cast<int>());
        int usableCount = engine.TensorSum(usable);
        if (usableCount == 0)
        {
            batch.Clear();
            return (-1, NumOps.Zero, 0, NumOps.Zero, false);
        }

        var lowest = engine.TensorFullLike(gains, NumOps.MinValue);
        var masked = engine.TensorWhere(usable.Cast<T>(), gains, lowest);

        // The first candidate with the largest gain: the maximal entries weighted by (candidates - position).
        T bestGain = engine.TensorMaxValue(masked);
        var isBest = engine.TensorMultiply(engine.TensorEquals(masked, bestGain).Cast<int>(), usable);
        var priority = engine.TensorMultiply(isBest, new Tensor<int>(
            Enumerable.Range(0, candidates).Select(i => candidates - i).ToArray(), new[] { candidates }));
        int best = engine.TensorArgMax(priority, 0).GetFlat(0);

        var result = (batch.Feature[best], thresholds.GetFlat(best), batch.LowerRank[best], bestGain, true);
        batch.Clear();
        return result;
    }

    /// <summary>
    /// The configured criterion (Gini, or entropy in bits as scikit-learn) of each row of per-class
    /// <paramref name="classCounts"/> [rows, classes] over its <paramref name="sizes"/> [rows].
    /// </summary>
    private Tensor<T> Impurity(Tensor<int> classCounts, Tensor<int> sizes, int shift)
    {
        var engine = AiDotNetEngine.Current;
        int rows = classCounts.Shape[0], classes = classCounts.Shape[1];
        var shape = new[] { rows, classes };
        var p = engine.TensorDivide(
            ScaledCounts(classCounts, shift),
            engine.TensorBroadcastTo(engine.Reshape(ScaledCounts(sizes, shift), new[] { rows, 1 }), shape));

        if (Options.Criterion is ClassificationSplitCriterion.Entropy or ClassificationSplitCriterion.LogLoss)
        {
            // -sum p log2 p over the classes present (0 log 0 = 0).
            var present = engine.TensorGreaterThan(p, NumOps.Zero);
            var safe = engine.TensorWhere(present, p, engine.TensorOnesLike(p));
            return engine.TensorNegate(engine.ReduceSum(engine.TensorMultiply(p, engine.TensorLog2(safe)), new[] { 1 }, false));
        }

        // 1 - sum p^2
        return engine.TensorSubtract(
            engine.TensorFullLike(engine.ReduceSum(p, new[] { 1 }, false), NumOps.One),
            engine.ReduceSum(engine.TensorMultiply(p, p), new[] { 1 }, false));
    }

    /// <summary>
    /// The power of two that brings counts up to <paramref name="largest"/> under 2^11, where every supported T holds
    /// integers exactly (Half saturates at 65504; its integers are exact only to 2048).
    /// </summary>
    private static int ScaleShift(int largest)
    {
        int bits = 0;
        while (bits < 31 && (largest >> bits) != 0) bits++;
        return Math.Max(0, bits - ExactIntegerBits);
    }

    /// <summary>
    /// <paramref name="counts"/> / 2^<paramref name="shift"/> in T, as quotient + remainder * 2^-shift: both parts are
    /// small integers, so no step overflows a narrow T, and for double the value is exact.
    /// </summary>
    private Tensor<T> ScaledCounts(Tensor<int> counts, int shift)
    {
        if (shift == 0)
            return counts.Cast<T>();

        var engine = AiDotNetEngine.Current;
        int mask = (1 << shift) - 1;
        var values = counts.ToArray();
        var quotient = new Tensor<int>(values.Select(c => c >> shift).ToArray(), counts.Shape.ToArray());
        var remainder = new Tensor<int>(values.Select(c => c & mask).ToArray(), counts.Shape.ToArray());
        return engine.TensorAdd(
            quotient.Cast<T>(), engine.TensorMultiplyScalar(remainder.Cast<T>(), NumOps.FromDouble(1.0 / (1 << shift))));
    }

    /// <summary>One count / 2^<paramref name="shift"/> in T (exact in double, one rounding into T).</summary>
    private T ScaledCount(int count, int shift) => NumOps.FromDouble(count / (double)(1 << shift));

    /// <summary><paramref name="numerator"/> / <paramref name="denominator"/> in T without converting either whole.</summary>
    private T Fraction(int numerator, int denominator)
    {
        int shift = ScaleShift(denominator);
        return NumOps.Divide(ScaledCount(numerator, shift), ScaledCount(denominator, shift));
    }

    /// <summary>
    /// Sorts the first <paramref name="count"/> keys by their rank (upper 32 bits). Rows of equal rank may end in any
    /// order: the sweep moves whole runs of a rank at once.
    /// </summary>
    private void SortByRank(int count)
    {
        if (count < RadixSortMinRows)
        {
            Array.Sort(_sortKeys, 0, count);
            return;
        }

        // LSD radix sort, 8 bits per pass, over the rank bits only.
        var source = _sortKeys;
        var target = _radixScratch;
        Span<int> offsets = stackalloc int[256];
        for (int shift = 32; shift < 32 + _denseData.RankBits; shift += 8)
        {
            offsets.Clear();
            for (int i = 0; i < count; i++) offsets[(int)(source[i] >> shift) & 0xFF]++;

            int sum = 0;
            for (int d = 0; d < 256; d++)
            {
                int bucket = offsets[d];
                offsets[d] = sum;
                sum += bucket;
            }

            for (int i = 0; i < count; i++)
            {
                ulong key = source[i];
                target[offsets[(int)(key >> shift) & 0xFF]++] = key;
            }

            (source, target) = (target, source);
        }

        if (!ReferenceEquals(source, _sortKeys))
            Array.Copy(source, _sortKeys, count);
    }
}
