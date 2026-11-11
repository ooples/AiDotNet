using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Attributes;

namespace AiDotNet.Classification.Boosting;

/// <summary>
/// Histogram tree construction, level by level, on row-major bins, with each node's samples kept as a contiguous range
/// of one index array.
/// </summary>
/// <remarks>
/// <para>
/// The builder used to read bins from a row-major <c>int[,]</c> one feature at a time, allocate three histogram arrays
/// per feature per node and two growing lists per split, and after each tree walk it again for every sample to update
/// the predictions. 100 rounds on 20,000 x 20 took 2.3 s against LightGBM's 0.48 s, single-threaded.
/// </para>
/// <para>
/// A node's histogram over all features is accumulated in one pass over its samples: each sample's bins are one short
/// row (a byte per feature when every feature has at most 256 bins), and each bin holds its gradient sum, hessian sum
/// and count side by side. Of two children only the smaller one's histogram is accumulated; the larger one's is the
/// parent's minus it, as LightGBM, XGBoost and scikit-learn do. An empty bin is never read, so the rounding a
/// subtraction leaves in one does not matter.
/// </para>
/// <para>
/// The tree grows one level at a time: the nodes of a level are split, and their children's histograms built, in
/// parallel (<see cref="NJobs"/>). Histograms are built in parallel by feature as well as by node, as LightGBM does:
/// each task fills a block of features from all of a node's samples, so every cell sums its samples in their order,
/// as one thread would. Every node's work is independent of the others and of the thread count, so the model is the
/// same for any <see cref="NJobs"/>. Each leaf writes its value for its own samples,
/// which replaces walking every tree again to update the predictions.
/// </para>
/// </remarks>
public partial class HistGradientBoostingClassifier<T>
{
    /// <summary>Doubles per histogram bin: gradient sum, hessian sum, count.</summary>
    private const int BinStride = 3;

    /// <summary>Rows per block of the per-row passes (gradients, prediction updates) run in parallel.</summary>
    private const int RowBlockRows = 8192;

    /// <summary>
    /// Gets or sets the number of threads that build each tree: a positive count, or negative to count back from the
    /// processor count (-1, the default, uses every processor, as scikit-learn's HistGradientBoostingClassifier
    /// does). The trained model is the same for any value.
    /// </summary>
    public int NJobs { get; set; } = -1;

    /// <summary>Bin of sample <c>i</c> in feature <c>j</c>, at <c>i * features + j</c>, when no feature has over 256 bins.</summary>
    [Scratch]
    private byte[] _binBytes = new byte[0];

    /// <summary>As <see cref="_binBytes"/>, when some feature has more than 256 bins.</summary>
    [Scratch]
    private ushort[] _binShorts = new ushort[0];

    /// <summary>Where feature <c>j</c>'s bins start in a node histogram; the last entry is the total bin count.</summary>
    [Scratch]
    private int[] _featureBinOffsets = new int[0];

    /// <summary>Sample indices of the current level; each node owns a contiguous range.</summary>
    [Scratch]
    private int[] _histSamples = new int[0];

    /// <summary>
    /// Sample indices of the next level: a split node partitions its range of <see cref="_histSamples"/> into the
    /// same range here, and the two arrays swap once the level has split.
    /// </summary>
    [Scratch]
    private int[] _histScratch = new int[0];

    /// <summary>Node histograms not in use, reused across nodes and trees.</summary>
    [Scratch]
    private Stack<double[]> _freeHistograms = new();

    /// <summary>The value of the leaf each sample landed in, for the tree being built.</summary>
    [Scratch]
    private double[] _sampleLeafValue = new double[0];

    [Scratch]
    private int _histRows;

    [Scratch]
    private int _histJobs = 1;

    /// <summary>A node of the tree being grown.</summary>
    private sealed class GrowingNode
    {
        public GrowingNode(int start, int count, int depth)
        {
            Start = start;
            Count = count;
            Depth = depth;
        }

        public int Start { get; }
        public int Count { get; }
        public int Depth { get; }
        public HistTree Tree { get; } = new();

        /// <summary>This node's histogram, when it may split.</summary>
        public double[]? Histogram { get; set; }

        public bool IsSplit { get; set; }
        public int LeftCount { get; set; }

        /// <summary>Gradient and hessian sums over the node's samples, in their order.</summary>
        public double SumGrad { get; set; }
        public double SumHess { get; set; }

        /// <summary>The children's sums, gathered while partitioning.</summary>
        public double LeftSumGrad { get; set; }
        public double LeftSumHess { get; set; }
        public double RightSumGrad { get; set; }
        public double RightSumHess { get; set; }
        public GrowingNode? Left { get; set; }
        public GrowingNode? Right { get; set; }
    }

    /// <summary>
    /// Computes every feature's bin boundaries (quantiles of its sorted values), bins <paramref name="x"/> row-major
    /// and sizes the builder's scratch. Returns the boundaries.
    /// </summary>
    private double[][] PrepareHistBuilder(Matrix<T> x)
    {
        int n = x.Rows;
        int p = x.Columns;
        _histRows = n;
        _histJobs = ResolveHistJobs();

        var columns = new double[p][];
        for (int j = 0; j < p; j++) columns[j] = new double[n];
        for (int i = 0; i < n; i++)
        {
            var row = x.GetRowReadOnlySpan(i);
            for (int j = 0; j < p; j++)
            {
                double val = NumOps.ToDouble(row[j]);
                if (double.IsNaN(val) || double.IsInfinity(val))
                {
                    throw new ArgumentException(
                        $"Missing/NaN/Infinity values are not supported. Found at row {i}, feature {j}. Please impute before training.",
                        nameof(x));
                }

                columns[j][i] = val;
            }
        }

        // Quantile boundaries over each feature's sorted values. Features are independent, so they run in parallel.
        var boundaries = new double[p][];
        var orders = new int[p][];
        ForEachFeature(p, j =>
        {
            var values = columns[j];
            var order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;
            Array.Sort(values, order);
            orders[j] = order;

            int numBins = Math.Min(_maxBins, n);
            var binBounds = new List<double>();
            for (int b = 1; b < numBins; b++)
            {
                int idx = (int)((double)b / numBins * n);
                idx = Math.Min(idx, n - 1);
                if (binBounds.Count == 0 || values[idx] > binBounds[^1])
                {
                    binBounds.Add(values[idx]);
                }
            }

            boundaries[j] = [.. binBounds];
        });

        _featureBinOffsets = new int[p + 1];
        int widest = 0;
        for (int j = 0; j < p; j++)
        {
            if (boundaries[j].Length >= ushort.MaxValue)
            {
                throw new InvalidOperationException(
                    $"Feature {j} has {boundaries[j].Length + 1} bins; at most {ushort.MaxValue} are supported.");
            }

            _featureBinOffsets[j + 1] = _featureBinOffsets[j] + boundaries[j].Length + 1;
            widest = Math.Max(widest, boundaries[j].Length + 1);
        }

        // A value's bin is the number of boundaries below it (as a binary search over the boundaries finds).
        bool bytes = widest <= 256;
        _binBytes = bytes ? new byte[n * p] : new byte[0];
        _binShorts = bytes ? new ushort[0] : new ushort[n * p];
        ForEachFeature(p, j =>
        {
            var values = columns[j];
            var order = orders[j];
            var bounds = boundaries[j];
            int bin = 0;
            for (int s = 0; s < n; s++)
            {
                while (bin < bounds.Length && bounds[bin] < values[s]) bin++;
                if (bytes)
                    _binBytes[order[s] * p + j] = (byte)bin;
                else
                    _binShorts[order[s] * p + j] = (ushort)bin;
            }
        });

        _histSamples = new int[n];
        _histScratch = new int[n];
        _freeHistograms = new Stack<double[]>();
        _sampleLeafValue = new double[n];
        return boundaries;
    }

    /// <summary>Releases the training scratch.</summary>
    private void ReleaseHistBuilder()
    {
        _binBytes = new byte[0];
        _binShorts = new ushort[0];
        _featureBinOffsets = new int[0];
        _histSamples = new int[0];
        _histScratch = new int[0];
        _freeHistograms = new Stack<double[]>();
        _sampleLeafValue = new double[0];
    }

    private int ResolveHistJobs()
    {
        if (NJobs == 0)
        {
            throw new InvalidOperationException(
                $"{nameof(NJobs)} must be positive, or negative to count back from the processor count; got 0.");
        }

        return NJobs > 0 ? NJobs : Math.Max(1, Environment.ProcessorCount + 1 + NJobs);
    }

    /// <summary>Runs <paramref name="body"/> for every feature, in parallel when jobs allow.</summary>
    private void ForEachFeature(int features, Action<int> body)
    {
        if (_histJobs > 1 && features > 1)
            Parallel.For(0, features, new ParallelOptions { MaxDegreeOfParallelism = _histJobs }, body);
        else
            for (int j = 0; j < features; j++) body(j);
    }

    /// <summary>
    /// Runs <paramref name="body"/> over fixed blocks <c>[from, to)</c> of the rows, in parallel when jobs allow. For
    /// work that treats each row on its own, so the result does not depend on how the blocks were scheduled.
    /// </summary>
    private void ForEachRowBlock(int rows, Action<int, int> body)
    {
        int blocks = (rows + RowBlockRows - 1) / RowBlockRows;
        if (_histJobs > 1 && blocks > 1)
        {
            Parallel.For(0, blocks, new ParallelOptions { MaxDegreeOfParallelism = _histJobs },
                b => body(b * RowBlockRows, Math.Min(rows, (b + 1) * RowBlockRows)));
        }
        else
        {
            body(0, rows);
        }
    }

    private int HistogramLength => _featureBinOffsets[NumFeatures] * BinStride;

    private double[] RentHistogram() =>
        _freeHistograms.Count > 0 ? _freeHistograms.Pop() : new double[HistogramLength];

    private void ReturnHistogram(double[]? histogram)
    {
        if (histogram is not null) _freeHistograms.Push(histogram);
    }

    /// <summary>Whether a node of <paramref name="count"/> samples at <paramref name="depth"/> may split.</summary>
    private bool NeedsHistogram(int count, int depth) => depth < _maxDepth && count >= 2 * _minSamplesLeaf;

    /// <summary>Runs <paramref name="action"/> on every node, in parallel when there are several and jobs allow.</summary>
    private void ForEachNode(List<GrowingNode> nodes, Action<GrowingNode> action)
    {
        if (_histJobs > 1 && nodes.Count > 1)
            Parallel.ForEach(nodes, new ParallelOptions { MaxDegreeOfParallelism = _histJobs }, action);
        else
            foreach (var node in nodes) action(node);
    }

    /// <summary>
    /// Builds one tree over every sample. Afterwards <see cref="_sampleLeafValue"/> holds each sample's leaf value.
    /// </summary>
    private HistTree BuildHistTreeDense(double[] gradients, double[] hessians)
    {
        for (int i = 0; i < _histRows; i++) _histSamples[i] = i;

        var root = new GrowingNode(0, _histRows, 0);
        double rootGrad = 0;
        double rootHess = 0;
        for (int i = 0; i < _histRows; i++)
        {
            rootGrad += gradients[i];
            rootHess += hessians[i];
        }

        root.SumGrad = rootGrad;
        root.SumHess = rootHess;
        if (NeedsHistogram(_histRows, 0))
        {
            var rootHistogram = RentHistogram();
            root.Histogram = rootHistogram;
            int blocks = FeatureBlocks(1);
            ForEachTask(blocks, b =>
            {
                var (from, to) = FeatureRange(b, blocks);
                AccumulateRows(rootHistogram, gradients, hessians, 0, _histRows, from, to);
            });
        }

        var level = new List<GrowingNode> { root };
        while (level.Count > 0)
        {
            ForEachNode(level, node => SplitNode(node, gradients, hessians));
            (_histSamples, _histScratch) = (_histScratch, _histSamples);

            // Children in tree order; the histograms they will need are rented here, on one thread.
            var next = new List<GrowingNode>();
            var split = new List<GrowingNode>();
            foreach (var node in level)
            {
                if (!node.IsSplit)
                {
                    ReturnHistogram(node.Histogram);
                    node.Histogram = null;
                    continue;
                }

                var left = new GrowingNode(node.Start, node.LeftCount, node.Depth + 1)
                {
                    SumGrad = node.LeftSumGrad, SumHess = node.LeftSumHess
                };
                var right = new GrowingNode(node.Start + node.LeftCount, node.Count - node.LeftCount, node.Depth + 1)
                {
                    SumGrad = node.RightSumGrad, SumHess = node.RightSumHess
                };
                node.Left = left;
                node.Right = right;
                node.Tree.LeftChild = left.Tree;
                node.Tree.RightChild = right.Tree;
                next.Add(left);
                next.Add(right);

                bool leftNeeds = NeedsHistogram(left.Count, left.Depth);
                bool rightNeeds = NeedsHistogram(right.Count, right.Depth);
                if (leftNeeds || rightNeeds)
                {
                    // The smaller child is always accumulated (the larger one's histogram is derived from it).
                    var (small, large, largeNeeds) = left.Count <= right.Count
                        ? (left, right, rightNeeds)
                        : (right, left, leftNeeds);
                    small.Histogram = RentHistogram();
                    if (largeNeeds) large.Histogram = RentHistogram();
                    split.Add(node);
                }
                else
                {
                    ReturnHistogram(node.Histogram);
                    node.Histogram = null;
                }
            }

            int splitBlocks = FeatureBlocks(split.Count);
            ForEachTask(split.Count * splitBlocks, t =>
            {
                var (from, to) = FeatureRange(t % splitBlocks, splitBlocks);
                BuildChildHistograms(split[t / splitBlocks], gradients, hessians, from, to);
            });

            foreach (var node in split)
            {
                ReturnHistogram(node.Histogram);
                node.Histogram = null;
            }

            // A child that cannot split does not keep the histogram its sibling's subtraction needed.
            foreach (var child in next)
            {
                if (child.Histogram is not null && !NeedsHistogram(child.Count, child.Depth))
                {
                    ReturnHistogram(child.Histogram);
                    child.Histogram = null;
                }
            }

            level = next;
        }

        return root.Tree;
    }

    /// <summary>
    /// Makes <paramref name="node"/> a leaf, or finds its best split and partitions its samples for its children,
    /// summing each child's gradients and hessians in its sample order on the way.
    /// </summary>
    private void SplitNode(GrowingNode node, double[] gradients, double[] hessians)
    {
        int count = node.Count;
        double sumGrad = node.SumGrad;
        double sumHess = node.SumHess;
        var histogram = node.Histogram;
        if (histogram is null || !NeedsHistogram(count, node.Depth))
        {
            MakeLeaf(node, sumGrad, sumHess);
            return;
        }

        int bestFeature = -1;
        int bestBin = -1;
        int bestLeftCount = 0;
        double bestGain = 0;
        for (int j = 0; j < NumFeatures; j++)
        {
            var (gain, bin, leftCount) = ScanFeature(histogram, j, count, sumGrad, sumHess);
            if (gain > bestGain)
            {
                bestGain = gain;
                bestFeature = j;
                bestBin = bin;
                bestLeftCount = leftCount;
            }
        }

        if (bestFeature < 0)
        {
            MakeLeaf(node, sumGrad, sumHess);
            return;
        }

        if (_binBytes.Length > 0)
            Partition(node, gradients, hessians, bestFeature, bestBin, bestLeftCount, new ReadOnlySpan<byte>(_binBytes));
        else
            Partition(node, gradients, hessians, bestFeature, bestBin, bestLeftCount, new ReadOnlySpan<ushort>(_binShorts));

        node.IsSplit = true;
        node.Tree.FeatureIndex = bestFeature;
        node.Tree.BinThreshold = bestBin;
    }

    /// <summary>
    /// Stable partition into the next level's samples: the left child keeps samples in bins &lt;=
    /// <paramref name="bestBin"/>, in their original order, and the right child the rest. The histogram already
    /// counted the left side (<paramref name="leftCount"/>), so the right side's range is known from the start.
    /// </summary>
    private void Partition<TBin>(GrowingNode node, double[] gradients, double[] hessians, int bestFeature, int bestBin,
        int leftCount, ReadOnlySpan<TBin> bins) where TBin : struct
    {
        int start = node.Start;
        int count = node.Count;
        int features = NumFeatures;
        ref TBin column = ref Unsafe.Add(ref MemoryMarshal.GetReference(bins), bestFeature);
        ref int samples = ref MemoryMarshal.GetReference(new ReadOnlySpan<int>(_histSamples));
        ref int next = ref MemoryMarshal.GetReference(new Span<int>(_histScratch));
        ref double grad = ref MemoryMarshal.GetReference(new ReadOnlySpan<double>(gradients));
        ref double hess = ref MemoryMarshal.GetReference(new ReadOnlySpan<double>(hessians));
        int leftTotal = 0;
        int rightTotal = 0;
        double leftGrad = 0, leftHess = 0, rightGrad = 0, rightHess = 0;
        for (int r = start; r < start + count; r++)
        {
            int i = Unsafe.Add(ref samples, r);
            TBin raw = Unsafe.Add(ref column, i * features);
            int bin = typeof(TBin) == typeof(byte) ? Unsafe.As<TBin, byte>(ref raw) : Unsafe.As<TBin, ushort>(ref raw);
            if (bin <= bestBin)
            {
                Unsafe.Add(ref next, start + leftTotal++) = i;
                leftGrad += Unsafe.Add(ref grad, i);
                leftHess += Unsafe.Add(ref hess, i);
            }
            else
            {
                Unsafe.Add(ref next, start + leftCount + rightTotal++) = i;
                rightGrad += Unsafe.Add(ref grad, i);
                rightHess += Unsafe.Add(ref hess, i);
            }
        }

        if (leftTotal != leftCount)
        {
            throw new InvalidOperationException(
                $"The histogram counted {leftCount} samples left of the split but {leftTotal} were partitioned there.");
        }

        node.LeftCount = leftTotal;
        node.LeftSumGrad = leftGrad;
        node.LeftSumHess = leftHess;
        node.RightSumGrad = rightGrad;
        node.RightSumHess = rightHess;
    }

    private void MakeLeaf(GrowingNode node, double sumGrad, double sumHess)
    {
        double leafValue = -sumGrad / (sumHess + _l2Regularization);
        node.Tree.LeafValue = leafValue;
        for (int r = node.Start; r < node.Start + node.Count; r++)
            _sampleLeafValue[_histSamples[r]] = leafValue;
    }

    /// <summary>
    /// The best split of one feature's bins: its gain (0 when no candidate improves on the parent), the last bin of
    /// its left side, and how many samples fall on that side.
    /// </summary>
    private (double Gain, int Bin, int LeftCount) ScanFeature(double[] histogram, int feature, int count, double sumGrad,
        double sumHess)
    {
        double lambda = _l2Regularization;
        int minLeaf = _minSamplesLeaf;
        double parentVal = sumGrad * sumGrad / (sumHess + lambda);
        int offset = _featureBinOffsets[feature];
        int numBins = _featureBinOffsets[feature + 1] - offset;
        double bestGain = 0;
        int bestBin = -1;
        int bestLeftCount = 0;
        double leftGradSum = 0;
        double leftHessSum = 0;
        int leftCount = 0;
        ref double cell = ref Unsafe.Add(ref MemoryMarshal.GetReference(new ReadOnlySpan<double>(histogram)),
            offset * BinStride);
        for (int b = 0; b < numBins - 1; b++, cell = ref Unsafe.Add(ref cell, BinStride))
        {
            int binCount = (int)Unsafe.Add(ref cell, 2);

            // An empty bin leaves the left side as it was: its candidate repeats the previous one's gain, which the
            // strict comparison below never takes.
            if (binCount == 0)
                continue;

            leftGradSum += cell;
            leftHessSum += Unsafe.Add(ref cell, 1);
            leftCount += binCount;

            int rightCount = count - leftCount;
            if (leftCount < minLeaf || rightCount < minLeaf)
                continue;

            double rightGradSum = sumGrad - leftGradSum;
            double rightHessSum = sumHess - leftHessSum;
            double leftVal = leftGradSum * leftGradSum / (leftHessSum + lambda);
            double rightVal = rightGradSum * rightGradSum / (rightHessSum + lambda);
            double gain = 0.5 * (leftVal + rightVal - parentVal);

            if (gain > bestGain)
            {
                bestGain = gain;
                bestBin = b;
                bestLeftCount = leftCount;
            }
        }

        return (bestGain, bestBin, bestLeftCount);
    }

    /// <summary>
    /// Accumulates features <c>[featureFrom, featureTo)</c> of the smaller child's histogram of a split node and, when
    /// the larger child may split too, derives the same features of its histogram as the parent's minus the smaller one's.
    /// </summary>
    private void BuildChildHistograms(GrowingNode node, double[] gradients, double[] hessians, int featureFrom,
        int featureTo)
    {
        var left = node.Left ?? throw new InvalidOperationException("A split node has no left child.");
        var right = node.Right ?? throw new InvalidOperationException("A split node has no right child.");
        var parent = node.Histogram ?? throw new InvalidOperationException("A split node has no histogram.");
        var (small, large) = left.Count <= right.Count ? (left, right) : (right, left);
        var smallHistogram = small.Histogram ?? throw new InvalidOperationException("The smaller child has no histogram.");

        AccumulateRows(smallHistogram, gradients, hessians, small.Start, small.Count, featureFrom, featureTo);
        if (large.Histogram is { } largeHistogram)
        {
            int from = _featureBinOffsets[featureFrom] * BinStride;
            int to = _featureBinOffsets[featureTo] * BinStride;
            for (int k = from; k < to; k++) largeHistogram[k] = parent[k] - smallHistogram[k];
        }
    }

    /// <summary>
    /// How many feature blocks each of <paramref name="nodes"/> nodes' histograms is split into, so a level has about
    /// one task per thread: one block when there is a thread per node already, at most one per feature.
    /// </summary>
    private int FeatureBlocks(int nodes) =>
        _histJobs <= 1 || nodes <= 0 ? 1 : Math.Max(1, Math.Min(NumFeatures, (_histJobs + nodes - 1) / nodes));

    /// <summary>The features <c>[from, to)</c> of block <paramref name="block"/> of <paramref name="blocks"/>.</summary>
    private (int From, int To) FeatureRange(int block, int blocks) =>
        (block * NumFeatures / blocks, (block + 1) * NumFeatures / blocks);

    /// <summary>Runs <paramref name="task"/> for <c>[0, tasks)</c>, in parallel when there are several and jobs allow.</summary>
    private void ForEachTask(int tasks, Action<int> task)
    {
        if (_histJobs > 1 && tasks > 1)
            Parallel.For(0, tasks, new ParallelOptions { MaxDegreeOfParallelism = _histJobs }, task);
        else
            for (int t = 0; t < tasks; t++) task(t);
    }

    /// <summary>
    /// Fills features <c>[featureFrom, featureTo)</c> of <paramref name="histogram"/> from the samples in
    /// <c>[start, start + count)</c>, in their order.
    /// </summary>
    private void AccumulateRows(double[] histogram, double[] gradients, double[] hessians, int start, int count,
        int featureFrom, int featureTo)
    {
        int cellsFrom = _featureBinOffsets[featureFrom] * BinStride;
        Array.Clear(histogram, cellsFrom, _featureBinOffsets[featureTo] * BinStride - cellsFrom);
        if (_binBytes.Length > 0)
            AccumulateRows(histogram, gradients, hessians, start, count, featureFrom, featureTo,
                new ReadOnlySpan<byte>(_binBytes));
        else
            AccumulateRows(histogram, gradients, hessians, start, count, featureFrom, featureTo,
                new ReadOnlySpan<ushort>(_binShorts));
    }

    /// <summary>
    /// The accumulation loop, through refs: the indices are in range by construction (bins below each feature's bin
    /// count, samples below the row count), and span bounds checks cost a third of the loop.
    /// </summary>
    private void AccumulateRows<TBin>(double[] histogram, double[] gradients, double[] hessians, int start, int count,
        int featureFrom, int featureTo, ReadOnlySpan<TBin> bins) where TBin : struct
    {
        int features = NumFeatures;
        ref double hist = ref MemoryMarshal.GetReference(new Span<double>(histogram));
        ref double grad = ref MemoryMarshal.GetReference(new ReadOnlySpan<double>(gradients));
        ref double hess = ref MemoryMarshal.GetReference(new ReadOnlySpan<double>(hessians));
        ref int samples = ref MemoryMarshal.GetReference(new ReadOnlySpan<int>(_histSamples));
        ref int offsets = ref MemoryMarshal.GetReference(new ReadOnlySpan<int>(_featureBinOffsets));
        ref TBin bin0 = ref MemoryMarshal.GetReference(bins);
        for (int r = start; r < start + count; r++)
        {
            int i = Unsafe.Add(ref samples, r);
            double g = Unsafe.Add(ref grad, i);
            double h = Unsafe.Add(ref hess, i);
            ref TBin row = ref Unsafe.Add(ref bin0, i * features);
            for (int j = featureFrom; j < featureTo; j++)
            {
                TBin raw = Unsafe.Add(ref row, j);
                int bin = typeof(TBin) == typeof(byte) ? Unsafe.As<TBin, byte>(ref raw) : Unsafe.As<TBin, ushort>(ref raw);
                ref double cell = ref Unsafe.Add(ref hist, (Unsafe.Add(ref offsets, j) + bin) * BinStride);
                cell += g;
                Unsafe.Add(ref cell, 1) += h;
                Unsafe.Add(ref cell, 2) += 1.0;
            }
        }
    }
}
