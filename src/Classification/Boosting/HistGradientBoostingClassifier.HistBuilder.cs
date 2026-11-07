using AiDotNet.Attributes;
namespace AiDotNet.Classification.Boosting;

/// <summary>
/// Histogram tree construction on column-major bins, with each node's samples kept as a contiguous range of one index
/// array.
/// </summary>
/// <remarks>
/// <para>
/// The builder used to read bins from a row-major <c>int[,]</c> (a strided walk per feature), allocate three
/// histogram arrays per feature per node and two growing lists per split, and after each tree walk it again for every
/// sample to update the predictions. 100 rounds on 20,000 x 20 took 2.3 s against LightGBM's 0.46 s, single-threaded.
/// </para>
/// <para>
/// This keeps the arithmetic: histograms accumulate the same samples in the same order, the gain expression and its
/// strict tie-breaking are unchanged, and the partition is stable, so every tree and prediction is identical. Each
/// leaf writes its value for its own samples, which replaces the per-sample tree walk.
/// </para>
/// </remarks>
public partial class HistGradientBoostingClassifier<T>
{
    /// <summary>Bin of sample <c>i</c> in feature <c>j</c>, at <c>j * rows + i</c>.</summary>
    [Scratch]
    private ushort[] _binColumns = new ushort[0];

    /// <summary>Sample indices; each node owns a contiguous range, partitioned in place for its children.</summary>
    [Scratch]
    private int[] _histSamples = new int[0];
    [Scratch]
    private int[] _histScratch = new int[0];

    /// <summary>Gradient sums, hessian sums and counts of one feature's bins.</summary>
    [Scratch]
    private double[] _binGradSum = new double[0];
    [Scratch]
    private double[] _binHessSum = new double[0];
    [Scratch]
    private int[] _binCount = new int[0];

    /// <summary>The value of the leaf each sample landed in, for the tree being built.</summary>
    [Scratch]
    private double[] _sampleLeafValue = new double[0];

    [Scratch]
    private int _histRows;

    /// <summary>Bins <paramref name="x"/> column-major and sizes the builder's scratch for it.</summary>
    private void PrepareHistBuilder(Matrix<T> x, double[][] boundaries)
    {
        int n = x.Rows;
        int p = x.Columns;
        _histRows = n;
        _binColumns = new ushort[n * p];
        int maxBins = 1;
        for (int j = 0; j < p; j++)
        {
            if (boundaries[j].Length >= ushort.MaxValue)
            {
                throw new InvalidOperationException(
                    $"Feature {j} has {boundaries[j].Length + 1} bins; at most {ushort.MaxValue} are supported.");
            }

            maxBins = Math.Max(maxBins, boundaries[j].Length + 1);
        }

        for (int i = 0; i < n; i++)
        {
            var row = x.GetRowReadOnlySpan(i);
            for (int j = 0; j < p; j++)
            {
                int bin = Array.BinarySearch(boundaries[j], NumOps.ToDouble(row[j]));
                if (bin < 0) bin = ~bin;
                _binColumns[j * n + i] = (ushort)bin;
            }
        }

        _histSamples = new int[n];
        _histScratch = new int[n];
        _binGradSum = new double[maxBins];
        _binHessSum = new double[maxBins];
        _binCount = new int[maxBins];
        _sampleLeafValue = new double[n];
    }

    /// <summary>Releases the training scratch.</summary>
    private void ReleaseHistBuilder()
    {
        _binColumns = new ushort[0];
        _histSamples = new int[0];
        _histScratch = new int[0];
        _binGradSum = new double[0];
        _binHessSum = new double[0];
        _binCount = new int[0];
        _sampleLeafValue = new double[0];
    }

    /// <summary>
    /// Builds one tree over every sample. Afterwards <see cref="_sampleLeafValue"/> holds each sample's leaf value.
    /// </summary>
    private HistTree BuildHistTreeDense(double[] gradients, double[] hessians)
    {
        for (int i = 0; i < _histRows; i++) _histSamples[i] = i;
        return BuildHistNode(gradients, hessians, 0, _histRows, 0);
    }

    private HistTree BuildHistNode(double[] gradients, double[] hessians, int start, int count, int depth)
    {
        double sumGrad = 0;
        double sumHess = 0;
        for (int r = start; r < start + count; r++)
        {
            int i = _histSamples[r];
            sumGrad += gradients[i];
            sumHess += hessians[i];
        }

        double leafValue = -sumGrad / (sumHess + _l2Regularization);

        if (depth >= _maxDepth || count < 2 * _minSamplesLeaf)
            return CreateHistLeaf(leafValue, start, count);

        int bestFeature = -1;
        int bestBin = -1;
        double bestGain = 0;
        double parentVal = sumGrad * sumGrad / (sumHess + _l2Regularization);
        var boundaries = _binBoundaries ?? throw new InvalidOperationException("Bin boundaries are not computed.");

        for (int j = 0; j < NumFeatures; j++)
        {
            int numBins = boundaries[j].Length + 1;
            Array.Clear(_binGradSum, 0, numBins);
            Array.Clear(_binHessSum, 0, numBins);
            Array.Clear(_binCount, 0, numBins);

            int column = j * _histRows;
            for (int r = start; r < start + count; r++)
            {
                int i = _histSamples[r];
                int bin = _binColumns[column + i];
                _binGradSum[bin] += gradients[i];
                _binHessSum[bin] += hessians[i];
                _binCount[bin]++;
            }

            double leftGradSum = 0;
            double leftHessSum = 0;
            int leftCount = 0;
            for (int b = 0; b < numBins - 1; b++)
            {
                leftGradSum += _binGradSum[b];
                leftHessSum += _binHessSum[b];
                leftCount += _binCount[b];

                int rightCount = count - leftCount;
                if (leftCount < _minSamplesLeaf || rightCount < _minSamplesLeaf)
                    continue;

                double rightGradSum = sumGrad - leftGradSum;
                double rightHessSum = sumHess - leftHessSum;
                double leftVal = leftGradSum * leftGradSum / (leftHessSum + _l2Regularization);
                double rightVal = rightGradSum * rightGradSum / (rightHessSum + _l2Regularization);
                double gain = 0.5 * (leftVal + rightVal - parentVal);

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = j;
                    bestBin = b;
                }
            }
        }

        if (bestFeature < 0)
            return CreateHistLeaf(leafValue, start, count);

        // Stable partition: the left child keeps samples in bins <= bestBin, in their original order.
        int bestColumn = bestFeature * _histRows;
        int leftTotal = 0;
        int rightTotal = 0;
        for (int r = start; r < start + count; r++)
        {
            int i = _histSamples[r];
            if (_binColumns[bestColumn + i] <= bestBin)
                _histSamples[start + leftTotal++] = i;
            else
                _histScratch[rightTotal++] = i;
        }

        Array.Copy(_histScratch, 0, _histSamples, start + leftTotal, rightTotal);

        var leftTree = BuildHistNode(gradients, hessians, start, leftTotal, depth + 1);
        var rightTree = BuildHistNode(gradients, hessians, start + leftTotal, rightTotal, depth + 1);

        return new HistTree
        {
            FeatureIndex = bestFeature,
            BinThreshold = bestBin,
            LeftChild = leftTree,
            RightChild = rightTree
        };
    }

    private HistTree CreateHistLeaf(double leafValue, int start, int count)
    {
        for (int r = start; r < start + count; r++)
            _sampleLeafValue[_histSamples[r]] = leafValue;

        return new HistTree { LeafValue = leafValue };
    }
}
