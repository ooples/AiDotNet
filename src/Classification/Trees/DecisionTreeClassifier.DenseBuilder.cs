using AiDotNet.Enums;

namespace AiDotNet.Classification.Trees;

/// <summary>
/// Tree construction on a dense column-major <see cref="double"/> copy of the training data.
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
/// </remarks>
public partial class DecisionTreeClassifier<T>
{
    /// <summary>Feature values, column-major: feature <c>f</c> of row <c>i</c> is at <c>f * rows + i</c>.</summary>
    private double[] _denseFeatures = new double[0];

    /// <summary>Each training row's index into <see cref="ClassifierBase{T}.ClassLabels"/>.</summary>
    private int[] _denseClasses = new int[0];

    /// <summary>Row indices; each node owns a contiguous range, partitioned in place for its children.</summary>
    private int[] _denseRows = new int[0];

    /// <summary>Scratch for the split search and the partition, sized to the training rows.</summary>
    private double[] _sortKeys = new double[0];
    private int[] _sortClasses = new int[0];
    private int[] _partitionScratch = new int[0];

    private int _denseRowCount;

    /// <summary>Copies the training data into the dense buffers and grows the tree from every row.</summary>
    private DecisionNode<T> BuildDense(Matrix<T> x, Vector<T> y)
    {
        int rows = x.Rows;
        int features = x.Columns;
        _denseRowCount = rows;
        _denseFeatures = new double[rows * features];
        _denseClasses = new int[rows];
        _denseRows = new int[rows];
        _sortKeys = new double[rows];
        _sortClasses = new int[rows];
        _partitionScratch = new int[rows];

        for (int i = 0; i < rows; i++)
        {
            _denseClasses[i] = GetClassIndex(y[i]);
            _denseRows[i] = i;
            var row = x.GetRowReadOnlySpan(i);
            for (int f = 0; f < features; f++)
                _denseFeatures[f * rows + i] = NumOps.ToDouble(row[f]);
        }

        try
        {
            return BuildDenseNode(0, rows, 0);
        }
        finally
        {
            // The buffers are training scratch, not model state.
            _denseFeatures = new double[0];
            _denseClasses = new int[0];
            _denseRows = new int[0];
            _sortKeys = new double[0];
            _sortClasses = new int[0];
            _partitionScratch = new int[0];
        }
    }

    private DecisionNode<T> BuildDenseNode(int start, int count, int depth)
    {
        var counts = new int[NumClasses];
        for (int r = start; r < start + count; r++)
            counts[_denseClasses[_denseRows[r]]]++;

        if (ShouldStopDense(counts, count, depth))
            return CreateDenseLeaf(counts, count);

        var (bestFeature, bestThreshold, bestGain) = FindBestDenseSplit(start, count, counts);
        if (bestFeature < 0 || !(bestGain > Options.MinImpurityDecrease))
            return CreateDenseLeaf(counts, count);

        // Stable partition: the left child keeps rows comparing <= the threshold, in their original order.
        int column = bestFeature * _denseRowCount;
        int leftCount = 0;
        int rightCount = 0;
        for (int r = start; r < start + count; r++)
        {
            int row = _denseRows[r];
            if (_denseFeatures[column + row] <= bestThreshold)
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
    /// The best split of the node's rows: one sort per considered feature and one sweep over it, moving rows into the
    /// left class counts as the threshold rises.
    /// </summary>
    private (int Feature, double Threshold, double Gain) FindBestDenseSplit(int start, int count, int[] totalCounts)
    {
        int bestFeature = -1;
        double bestThreshold = 0;
        double bestGain = 0;

        var featuresToConsider = GetFeaturesToConsider();
        double currentImpurity = DenseImpurity(totalCounts, count);
        double total = count;
        var leftCounts = new int[NumClasses];
        var rightCounts = new int[NumClasses];

        foreach (int feature in featuresToConsider)
        {
            int column = feature * _denseRowCount;
            for (int r = 0; r < count; r++)
            {
                int row = _denseRows[start + r];
                _sortKeys[r] = _denseFeatures[column + row];
                _sortClasses[r] = _denseClasses[row];
            }

            Array.Sort(_sortKeys, _sortClasses, 0, count);
            Array.Clear(leftCounts, 0, leftCounts.Length);
            int leftCount = 0;

            for (int i = 0; i < count; )
            {
                // The next distinct value after the run starting at i.
                int next = i + 1;
                while (next < count && _sortKeys[next] == _sortKeys[i]) next++;
                if (next >= count) break;

                double threshold = (_sortKeys[i] + _sortKeys[next]) / 2.0;
                while (leftCount < count && _sortKeys[leftCount] <= threshold)
                    leftCounts[_sortClasses[leftCount++]]++;
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
                }
            }
        }

        return (bestFeature, bestThreshold, bestGain);
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
