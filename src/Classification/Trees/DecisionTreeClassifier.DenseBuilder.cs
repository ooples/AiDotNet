using AiDotNet.Enums;

namespace AiDotNet.Classification.Trees;

/// <summary>
/// Tree construction on ranked, column-major training data (<see cref="DenseTreeData"/>).
/// </summary>
/// <remarks>
/// <para>
/// The builder used to work through the generic matrix: a LINQ list of row indices per node, one generic numeric
/// operation and one linear class-label search per row visit, a fresh pair of lists for every split. A 100-tree
/// forest on 20,000 x 20 took 25 s against scikit-learn's 8.3 s, single-threaded both.
/// </para>
/// <para>
/// This keeps the algorithm and its arithmetic: the same candidate thresholds (midpoints between consecutive distinct
/// values), the same impurity and gain expressions in the same order, the same strict tie-breaking, the same random
/// feature draws in the same depth-first order, and child rows in their original order. For <c>double</c> a tree is
/// identical to the one the generic builder grew; other numeric types now compute splits in double precision.
/// </para>
/// <para>
/// Each considered feature's node rows are sorted as one 64-bit key per row, the rank above the class index, so a
/// large node sorts by radix on the rank bits alone and a small one by <see cref="Array.Sort{T}(T[], int, int)"/>.
/// </para>
/// </remarks>
public partial class DecisionTreeClassifier<T>
{
    /// <summary>Nodes below this many rows sort by comparison; larger ones by radix.</summary>
    private const int RadixSortMinRows = 256;

    private DenseTreeData _denseData = EmptyDenseData;

    /// <summary>Each training row's index into <see cref="ClassifierBase{T}.ClassLabels"/>, by data row.</summary>
    private int[] _denseClasses = new int[0];

    /// <summary>Data rows of the training sample, repeats allowed; each node owns a contiguous range.</summary>
    private int[] _denseRows = new int[0];

    /// <summary>Scratch for the split search and the partition, sized to the training sample.</summary>
    private ulong[] _sortKeys = new ulong[0];
    private ulong[] _radixScratch = new ulong[0];
    private int[] _partitionScratch = new int[0];
    private int[] _featurePool = new int[0];
    private int[] _selectedFeatures = new int[0];

    private static readonly DenseTreeData EmptyDenseData = new(new double[0], 0, 0);

    /// <summary>Ranks <paramref name="x"/> for <see cref="BuildDense"/>.</summary>
    internal static DenseTreeData CreateDenseData(Matrix<T> x)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int rows = x.Rows;
        int features = x.Columns;
        var columnMajor = new double[rows * features];
        for (int i = 0; i < rows; i++)
        {
            var row = x.GetRowReadOnlySpan(i);
            for (int f = 0; f < features; f++)
                columnMajor[f * rows + i] = numOps.ToDouble(row[f]);
        }

        return new DenseTreeData(columnMajor, rows, features);
    }

    /// <summary>
    /// Grows the tree from the data rows <paramref name="rows"/> of <paramref name="data"/>, labelled by
    /// <paramref name="y"/> (indexed by data row). A row listed twice counts twice, as in a bootstrap sample.
    /// </summary>
    private DecisionNode<T> BuildDense(DenseTreeData data, Vector<T> y, int[] rows)
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

        var labelled = new bool[data.Rows];
        foreach (int row in rows)
        {
            if (labelled[row]) continue;
            labelled[row] = true;
            _denseClasses[row] = GetClassIndex(y[row]);
        }

        try
        {
            return BuildDenseNode(0, count, 0);
        }
        finally
        {
            // The buffers are training scratch, not model state.
            _denseData = EmptyDenseData;
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

        var (bestFeature, bestThreshold, bestThresholdRank, bestGain) = FindBestDenseSplit(start, count, counts);
        if (bestFeature < 0 || !(bestGain > Options.MinImpurityDecrease))
            return CreateDenseLeaf(counts, count);

        // Stable partition: the left child keeps rows whose value is <= the threshold, in their original order.
        var ranks = _denseData.Ranks;
        int column = bestFeature * _denseData.Rows;
        int leftCount = 0;
        int rightCount = 0;
        for (int r = start; r < start + count; r++)
        {
            int row = _denseRows[r];
            if (ranks[column + row] <= bestThresholdRank)
                _denseRows[start + leftCount++] = row;
            else
                _partitionScratch[rightCount++] = row;
        }

        Array.Copy(_partitionScratch, 0, _denseRows, start + leftCount, rightCount);

        if (leftCount < Options.MinSamplesLeaf || rightCount < Options.MinSamplesLeaf)
            return CreateDenseLeaf(counts, count);

        UpdateFeatureImportances(bestFeature, NumOps.FromDouble(bestGain), count);

        var leftChild = BuildDenseNode(start, leftCount, depth + 1);
        var rightChild = BuildDenseNode(start + leftCount, rightCount, depth + 1);

        return new DecisionNode<T>
        {
            FeatureIndex = bestFeature,
            Threshold = NumOps.FromDouble(bestThreshold),
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
        var probabilities = new Vector<T>(NumClasses);
        T total = NumOps.FromDouble(count);
        for (int c = 0; c < NumClasses; c++)
            probabilities[c] = NumOps.Divide(NumOps.FromDouble(counts[c]), total);

        int predictedClass = 0;
        for (int c = 1; c < NumClasses; c++)
            if (counts[c] > counts[predictedClass])
                predictedClass = c;

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
    /// The best split of the node's rows: one sort per considered feature and one sweep over it, moving rows into the
    /// left class counts as the threshold rises.
    /// </summary>
    private (int Feature, double Threshold, int ThresholdRank, double Gain) FindBestDenseSplit(
        int start, int count, int[] totalCounts)
    {
        int bestFeature = -1;
        double bestThreshold = 0;
        int bestThresholdRank = 0;
        double bestGain = 0;

        int featureCount = SelectDenseFeatures();
        double currentImpurity = DenseImpurity(totalCounts, count);
        double total = count;
        var leftCounts = new int[NumClasses];
        var rightCounts = new int[NumClasses];
        var ranks = _denseData.Ranks;

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
            if (_denseData.HasNaN[feature] && (int)(_sortKeys[0] >> 32) == 0)
                continue;

            var values = _denseData.Values[feature];
            Array.Clear(leftCounts, 0, leftCounts.Length);
            int leftCount = 0;

            for (int i = 0; i < count; )
            {
                // The next distinct value after the run starting at i.
                int rank = (int)(_sortKeys[i] >> 32);
                int next = i + 1;
                while (next < count && (int)(_sortKeys[next] >> 32) == rank) next++;
                if (next >= count) break;

                int nextRank = (int)(_sortKeys[next] >> 32);
                double threshold = (values[rank] + values[nextRank]) / 2.0;

                // The midpoint can round onto the upper value, which then also compares <= the threshold.
                int thresholdRank = threshold >= values[nextRank] ? nextRank : rank;
                while (leftCount < count && (int)(_sortKeys[leftCount] >> 32) <= thresholdRank)
                    leftCounts[(int)(uint)_sortKeys[leftCount++]]++;
                i = next;

                int rightCount = count - leftCount;
                if (leftCount < Options.MinSamplesLeaf || rightCount < Options.MinSamplesLeaf)
                    continue;

                for (int c = 0; c < NumClasses; c++)
                    rightCounts[c] = totalCounts[c] - leftCounts[c];

                double leftImpurity = DenseImpurity(leftCounts, leftCount);
                double rightImpurity = DenseImpurity(rightCounts, rightCount);
                double weightedImpurity = leftCount / total * leftImpurity + rightCount / total * rightImpurity;
                double gain = currentImpurity - weightedImpurity;

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = feature;
                    bestThreshold = threshold;
                    bestThresholdRank = thresholdRank;
                }
            }
        }

        return (bestFeature, bestThreshold, bestThresholdRank, bestGain);
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

    /// <summary>The configured criterion over per-class counts of <paramref name="total"/> samples.</summary>
    private double DenseImpurity(int[] classCounts, int total)
    {
        if (total == 0)
            return 0;

        double totalD = total;
        if (Options.Criterion is ClassificationSplitCriterion.Entropy or ClassificationSplitCriterion.LogLoss)
        {
            double entropy = 0;
            for (int c = 0; c < NumClasses; c++)
            {
                if (classCounts[c] > 0)
                {
                    double prob = classCounts[c] / totalD;
                    entropy -= prob * Math.Log(prob);
                }
            }

            return entropy;
        }

        double gini = 1;
        for (int c = 0; c < NumClasses; c++)
        {
            double prob = classCounts[c] / totalD;
            gini -= prob * prob;
        }

        return gini;
    }
}
