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
/// The sweep scores each boundary between distinct values as it passes it, in T: the threshold (the midpoint of the
/// two values), the impurity of both sides, the weighted gain, and the first best candidate (features in order,
/// thresholds ascending, as scikit-learn's strict comparison). Class frequencies are formed from the integer counts by
/// <see cref="ScaledCountInParts"/>, which keeps every intermediate representable in T. These are per-candidate scalar
/// operations, not engine calls: a node has tens to thousands of candidates, too few to repay an engine dispatch.
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
    [Scratch]
    private int[] _leftCounts = new int[0];
    [Scratch]
    private int[] _rightCounts = new int[0];

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
        _leftCounts = new int[NumClasses];
        _rightCounts = new int[NumClasses];
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
            _leftCounts = new int[0];
            _rightCounts = new int[0];
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

    [AiDotNet.Attributes.ScalarKernel("One probability per class of one leaf: a handful of values, below what an engine dispatch repays.")]
    private DecisionNode<T> CreateDenseLeaf(int[] counts, int count)
    {
        int shift = ScaleShift(count);
        // A share of the node is a product with the reciprocal of its size, as the engine's divide-by-scalar forms it.
        T scaledTotal = ScaledCount(count, shift);
        T inverseTotal = NumOps.Divide(NumOps.One, scaledTotal);
        float inverseTotalFloat = 1.0f / NumOps.ToFloat(scaledTotal);
        var probabilities = new T[counts.Length];
        for (int c = 0; c < counts.Length; c++)
            probabilities[c] = Share(ScaledCountInParts(counts[c], shift), inverseTotal, inverseTotalFloat);

        // The first class with the most rows, as numpy's argmax.
        int predictedClass = Array.IndexOf(counts, counts.Max());

        return new DecisionNode<T>
        {
            IsLeaf = true,
            ClassProbabilities = new Vector<T>(probabilities),
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
    /// The best split of the node's rows. One sort per considered feature and one sweep over it: at each boundary
    /// between two distinct values the left and right class counts are known, and the candidate is scored in place.
    /// </summary>
    /// <remarks>
    /// Scoring runs per candidate in T (in double directly when T is double), with the arithmetic of the batched engine
    /// form it replaces (#2524): the same scaled counts, a share as a product with the reciprocal of the node's size
    /// (the engine's divide-by-scalar), the same products and sums in the same order, and strict comparison so the
    /// first best candidate wins (features in order, thresholds ascending). Trees are identical to that form's. A
    /// node's batch was a few dozen candidates to a few thousand, so dispatching about 40 engine operations and their
    /// tensors per node cost far more than the arithmetic: a 100-tree forest on 20,000 x 20 fitted in 36-49 s on one
    /// thread instead of 3.4 s.
    /// </remarks>
    [AiDotNet.Attributes.ScalarKernel("Scores each split candidate as the sweep passes it; a node has too few candidates to repay engine dispatch (the batched engine form made a 100-tree forest 10-14x slower, #2524).")]
    private (int Feature, T Threshold, int ThresholdRank, T Gain, bool Found) FindBestDenseSplit(
        int start, int count, int[] totalCounts)
    {
        int classes = NumClasses;
        int featureCount = SelectDenseFeatures();
        var ranks = _denseData.Ranks;
        var values = _denseData.DistinctValues;
        var left = _leftCounts;
        var right = _rightCounts;
        int minLeaf = Options.MinSamplesLeaf;

        // gain = impurity(node) - (n_left / n) impurity(left) - (n_right / n) impurity(right)
        int shift = ScaleShift(count);
        T nodeImpurity = Impurity(totalCounts, count, shift);
        // A child's share of the node is a product with the reciprocal of the node's size (the engine's divide-by-scalar).
        T scaledTotal = ScaledCount(count, shift);
        T inverseTotal = NumOps.Divide(NumOps.One, scaledTotal);
        float inverseTotalFloat = 1.0f / NumOps.ToFloat(scaledTotal);
        T half = NumOps.FromDouble(0.5);

        // For double the candidates are scored in double directly: the same operations in the same order as the
        // generic path below, without an interface call per operation (which made scoring 2x slower).
        bool doubles = typeof(T) == typeof(double);
        bool entropy = UsesEntropy;
        double inverseTotalDouble = doubles ? (double)(object)inverseTotal! : 0.0;
        double nodeImpurityDouble = doubles ? (double)(object)nodeImpurity! : 0.0;
        double bestGainDouble = 0.0;

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

            // A NaN compares false with every threshold, so no row ever moves left: the feature cannot split here.
            // NaNs share the rank above every numeric one, so they sort last.
            int nanRank = _denseData.NanRank[feature];
            if (nanRank >= 0 && (int)(_sortKeys[count - 1] >> 32) == nanRank)
                continue;

            int offset = _denseData.ValueOffsets[feature];
            Array.Clear(left, 0, classes);
            int leftSize = 0;
            for (int i = 0; i < count; )
            {
                // Move the run of one distinct value left; the boundary after it is a candidate unless it is the last.
                int rank = (int)(_sortKeys[i] >> 32);
                for (; i < count && (int)(_sortKeys[i] >> 32) == rank; i++)
                {
                    left[(int)(uint)_sortKeys[i]]++;
                    leftSize++;
                }

                if (i >= count)
                    break;

                // Both children must keep MinSamplesLeaf rows.
                int rightSize = count - leftSize;
                if (leftSize < minLeaf || rightSize < minLeaf)
                    continue;

                for (int c = 0; c < classes; c++)
                    right[c] = totalCounts[c] - left[c];

                // A NaN gain never wins (scikit-learn's ">" is false for it); a tie keeps the earlier candidate.
                T gain;
                if (doubles)
                {
                    double weightedDouble =
                        ScaledCountAsDouble(leftSize, shift) * inverseTotalDouble
                            * ImpurityOfDoubles(left, classes, leftSize, shift, entropy)
                        + ScaledCountAsDouble(rightSize, shift) * inverseTotalDouble
                            * ImpurityOfDoubles(right, classes, rightSize, shift, entropy);
                    double gainDouble = nodeImpurityDouble - weightedDouble;
                    if (double.IsNaN(gainDouble) || (best.Found && !(gainDouble > bestGainDouble)))
                        continue;

                    bestGainDouble = gainDouble;
                    gain = (T)(object)gainDouble;
                }
                else
                {
                    T leftShare = Share(ScaledCountInParts(leftSize, shift), inverseTotal, inverseTotalFloat);
                    T rightShare = Share(ScaledCountInParts(rightSize, shift), inverseTotal, inverseTotalFloat);
                    T weighted = NumOps.Add(
                        NumOps.Multiply(leftShare, Impurity(left, leftSize, shift)),
                        NumOps.Multiply(rightShare, Impurity(right, rightSize, shift)));
                    gain = NumOps.Subtract(nodeImpurity, weighted);
                    if (NumOps.IsNaN(gain) || (best.Found && !NumOps.GreaterThan(gain, best.Gain)))
                        continue;
                }

                // threshold = lower/2 + upper/2 (the midpoint without overflowing a narrow T). When rounding lands it
                // on the upper value (adjacent values in a narrow T) it is the lower value instead, as scikit-learn's
                // splitter does, so the split still separates the two runs and the left side is exactly the rows at
                // most the lower value.
                T lower = values[offset + rank];
                T upper = values[offset + (int)(_sortKeys[i] >> 32)];
                T midpoint = NumOps.Add(NumOps.Multiply(lower, half), NumOps.Multiply(upper, half));
                T threshold = NumOps.GreaterThanOrEquals(midpoint, upper) ? lower : midpoint;
                best = (feature, threshold, rank, gain, true);
            }
        }

        return best;
    }

    /// <summary>
    /// The configured criterion (Gini, or entropy in bits as scikit-learn) of per-class <paramref name="classCounts"/>
    /// over <paramref name="size"/> rows.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("The impurity of one candidate side over a handful of classes, called once per candidate by the split sweep.")]
    private T Impurity(int[] classCounts, int size, int shift)
    {
        T denominator = ScaledCountInParts(size, shift);
        // A narrow T (Half, BFloat16) sums its per-class terms in float and rounds once, as the engine's reduction
        // does; summing in T would round after every addition and lose small terms against a large one.
        bool narrow = IsNarrowFloat;
        T sum = NumOps.Zero;
        float sumFloat = 0.0f;
        if (UsesEntropy)
        {
            // -sum p log2 p over the classes present (0 log 0 = 0).
            for (int c = 0; c < NumClasses; c++)
            {
                T p = NumOps.Divide(ScaledCountInParts(classCounts[c], shift), denominator);
                T log = NumOps.GreaterThan(p, NumOps.Zero)
                    ? NumOps.FromDouble(Log2(NumOps.ToDouble(p)))
                    : NumOps.Zero;
                T term = NumOps.Multiply(p, log);
                if (narrow) sumFloat += NumOps.ToFloat(term);
                else sum = NumOps.Add(sum, term);
            }

            return NumOps.Negate(narrow ? NumOps.FromFloat(sumFloat) : sum);
        }

        // 1 - sum p^2
        for (int c = 0; c < NumClasses; c++)
        {
            T p = NumOps.Divide(ScaledCountInParts(classCounts[c], shift), denominator);
            T term = NumOps.Multiply(p, p);
            if (narrow) sumFloat += NumOps.ToFloat(term);
            else sum = NumOps.Add(sum, term);
        }

        return NumOps.Subtract(NumOps.One, narrow ? NumOps.FromFloat(sumFloat) : sum);
    }

    /// <summary>Whether T is a floating type narrower than float, whose CPU engine kernels compute in float.</summary>
    private static readonly bool IsNarrowFloat =
        typeof(T) == typeof(Half) || typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16);

    /// <summary>
    /// <paramref name="part"/> times the reciprocal of the node's size, as the engine's divide-by-scalar forms it: for a
    /// narrow T the reciprocal and the product are formed in float and rounded once, so the reciprocal is not first
    /// rounded to T.
    /// </summary>
    private T Share(T part, T inverseTotal, float inverseTotalFloat)
        => IsNarrowFloat
            ? NumOps.FromFloat(NumOps.ToFloat(part) * inverseTotalFloat)
            : NumOps.Multiply(part, inverseTotal);

    /// <summary>Whether the criterion is entropy (or log loss, the same split criterion) rather than Gini.</summary>
    private bool UsesEntropy =>
        Options.Criterion is ClassificationSplitCriterion.Entropy or ClassificationSplitCriterion.LogLoss;

    /// <summary><see cref="Impurity"/> for double: the same operations in the same order, in double.</summary>
    private static double ImpurityOfDoubles(int[] classCounts, int classes, int size, int shift, bool entropy)
    {
        double denominator = ScaledCountAsDouble(size, shift);
        double sum = 0.0;
        if (entropy)
        {
            for (int c = 0; c < classes; c++)
            {
                double p = ScaledCountAsDouble(classCounts[c], shift) / denominator;
                sum += p * (p > 0.0 ? Log2(p) : 0.0);
            }

            return -sum;
        }

        for (int c = 0; c < classes; c++)
        {
            double p = ScaledCountAsDouble(classCounts[c], shift) / denominator;
            sum += p * p;
        }

        return 1.0 - sum;
    }

    /// <summary><see cref="ScaledCountInParts"/> in double.</summary>
    private static double ScaledCountAsDouble(int count, int shift)
        => shift == 0 ? count : (count >> shift) + (count & ((1 << shift) - 1)) * (1.0 / (1 << shift));

    /// <summary>log2 of <paramref name="value"/>: Math.Log2 where the framework has it, as the engine's TensorLog2.</summary>
    private static double Log2(double value)
    {
#if NET5_0_OR_GREATER
        return Math.Log2(value);
#else
        return Math.Log(value, 2.0);
#endif
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
    /// <paramref name="count"/> / 2^<paramref name="shift"/> in T, as quotient + remainder * 2^-shift: both parts are
    /// small integers, so no step overflows a narrow T, and for double the value is exact.
    /// </summary>
    private T ScaledCountInParts(int count, int shift)
    {
        if (shift == 0)
            return NumOps.FromDouble(count);

        return NumOps.Add(NumOps.FromDouble(count >> shift),
            NumOps.Multiply(NumOps.FromDouble(count & ((1 << shift) - 1)), NumOps.FromDouble(1.0 / (1 << shift))));
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
