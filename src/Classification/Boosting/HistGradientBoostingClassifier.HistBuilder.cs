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
/// parallel (<see cref="NJobs"/>). A large histogram is accumulated in fixed chunks of samples added together in chunk
/// order, in parallel when it is the only one of its level. Every node's work is independent of the others and of the
/// thread count, so the model is the same for any <see cref="NJobs"/>. Each leaf writes its value for its own samples,
/// which replaces walking every tree again to update the predictions.
/// </para>
/// </remarks>
public partial class HistGradientBoostingClassifier<T>
{
    /// <summary>Samples per histogram chunk; a node with more is accumulated chunk by chunk.</summary>
    private const int HistogramChunkRows = 2048;

    /// <summary>Doubles per histogram bin: gradient sum, hessian sum, count.</summary>
    private const int BinStride = 3;

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

    /// <summary>Sample indices; each node owns a contiguous range, partitioned in place for its children.</summary>
    [Scratch]
    private int[] _histSamples = new int[0];

    /// <summary>Partition scratch: a node uses the same range here as in <see cref="_histSamples"/>.</summary>
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

        /// <summary>Chunk scratch for building the smaller child's histogram.</summary>
        public double[]? Partial { get; set; }
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

        // Quantile boundaries over each feature's sorted values.
        var boundaries = new double[p][];
        var orders = new int[p][];
        _featureBinOffsets = new int[p + 1];
        int widest = 0;
        for (int j = 0; j < p; j++)
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
        for (int j = 0; j < p; j++)
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
        }

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
            root.Histogram = RentHistogram();
            AccumulateHistogramInParallel(root.Histogram, gradients, hessians, 0, _histRows);
        }

        var level = new List<GrowingNode> { root };
        while (level.Count > 0)
        {
            ForEachNode(level, node => SplitNode(node, gradients, hessians));

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
                    if (small.Count > HistogramChunkRows) node.Partial = RentHistogram();
                    split.Add(node);
                }
                else
                {
                    ReturnHistogram(node.Histogram);
                    node.Histogram = null;
                }
            }

            if (split.Count == 1 && _histJobs > 1)
            {
                BuildChildHistograms(split[0], gradients, hessians, parallelChunks: true);
            }
            else
            {
                ForEachNode(split, node => BuildChildHistograms(node, gradients, hessians, parallelChunks: false));
            }

            foreach (var node in split)
            {
                ReturnHistogram(node.Histogram);
                ReturnHistogram(node.Partial);
                node.Histogram = null;
                node.Partial = null;
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
        double bestGain = 0;
        for (int j = 0; j < NumFeatures; j++)
        {
            var (gain, bin) = ScanFeature(histogram, j, count, sumGrad, sumHess);
            if (gain > bestGain)
            {
                bestGain = gain;
                bestFeature = j;
                bestBin = bin;
            }
        }

        if (bestFeature < 0)
        {
            MakeLeaf(node, sumGrad, sumHess);
            return;
        }

        if (_binBytes.Length > 0)
            Partition(node, gradients, hessians, bestFeature, bestBin, new ReadOnlySpan<byte>(_binBytes));
        else
            Partition(node, gradients, hessians, bestFeature, bestBin, new ReadOnlySpan<ushort>(_binShorts));

        node.IsSplit = true;
        node.Tree.FeatureIndex = bestFeature;
        node.Tree.BinThreshold = bestBin;
    }

    /// <summary>
    /// Stable partition: the left child keeps samples in bins &lt;= <paramref name="bestBin"/>, in their original
    /// order, and the right child the rest.
    /// </summary>
    private void Partition<TBin>(GrowingNode node, double[] gradients, double[] hessians, int bestFeature, int bestBin,
        ReadOnlySpan<TBin> bins) where TBin : struct
    {
        int start = node.Start;
        int count = node.Count;
        int features = NumFeatures;
        ref TBin column = ref Unsafe.Add(ref MemoryMarshal.GetReference(bins), bestFeature);
        ref int samples = ref MemoryMarshal.GetReference(new Span<int>(_histSamples));
        ref int scratch = ref MemoryMarshal.GetReference(new Span<int>(_histScratch));
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
                Unsafe.Add(ref samples, start + leftTotal++) = i;
                leftGrad += Unsafe.Add(ref grad, i);
                leftHess += Unsafe.Add(ref hess, i);
            }
            else
            {
                Unsafe.Add(ref scratch, start + rightTotal++) = i;
                rightGrad += Unsafe.Add(ref grad, i);
                rightHess += Unsafe.Add(ref hess, i);
            }
        }

        Array.Copy(_histScratch, start, _histSamples, start + leftTotal, rightTotal);
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
    /// The best split of one feature's bins: its gain (0 when no candidate improves on the parent) and the last bin
    /// of its left side.
    /// </summary>
    private (double Gain, int Bin) ScanFeature(double[] histogram, int feature, int count, double sumGrad,
        double sumHess)
    {
        double lambda = _l2Regularization;
        int minLeaf = _minSamplesLeaf;
        double parentVal = sumGrad * sumGrad / (sumHess + lambda);
        int offset = _featureBinOffsets[feature];
        int numBins = _featureBinOffsets[feature + 1] - offset;
        double bestGain = 0;
        int bestBin = -1;
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
            }
        }

        return (bestGain, bestBin);
    }

    /// <summary>
    /// Accumulates the smaller child's histogram of a split node and, when the larger child may split too, derives
    /// its histogram as the parent's minus the smaller one's.
    /// </summary>
    private void BuildChildHistograms(GrowingNode node, double[] gradients, double[] hessians, bool parallelChunks)
    {
        var left = node.Left ?? throw new InvalidOperationException("A split node has no left child.");
        var right = node.Right ?? throw new InvalidOperationException("A split node has no right child.");
        var parent = node.Histogram ?? throw new InvalidOperationException("A split node has no histogram.");
        var (small, large) = left.Count <= right.Count ? (left, right) : (right, left);
        var smallHistogram = small.Histogram ?? throw new InvalidOperationException("The smaller child has no histogram.");

        if (parallelChunks)
            AccumulateHistogramInParallel(smallHistogram, gradients, hessians, small.Start, small.Count);
        else
            AccumulateHistogram(smallHistogram, node.Partial, gradients, hessians, small.Start, small.Count);

        if (large.Histogram is { } largeHistogram)
        {
            AddScaled(largeHistogram, parent, smallHistogram, -1.0);
        }
    }

    /// <summary>
    /// Accumulates the histogram of the samples in <c>[start, start + count)</c> one chunk after another: the first
    /// chunk into <paramref name="histogram"/>, each later one into <paramref name="partial"/> and then added.
    /// </summary>
    private void AccumulateHistogram(double[] histogram, double[]? partial, double[] gradients, double[] hessians,
        int start, int count)
    {
        AccumulateRows(histogram, gradients, hessians, start, Math.Min(count, HistogramChunkRows));
        for (int chunkStart = start + HistogramChunkRows; chunkStart < start + count; chunkStart += HistogramChunkRows)
        {
            var scratch = partial ?? throw new InvalidOperationException("A chunked histogram needs a partial buffer.");
            AccumulateRows(scratch, gradients, hessians, chunkStart, Math.Min(HistogramChunkRows, start + count - chunkStart));
            AddScaled(histogram, histogram, scratch, 1.0);
        }
    }

    /// <summary>
    /// <see cref="AccumulateHistogram"/> with its chunks accumulated in parallel and then added in the same order, so
    /// the sums are the same.
    /// </summary>
    private void AccumulateHistogramInParallel(double[] histogram, double[] gradients, double[] hessians, int start,
        int count)
    {
        int chunks = (count + HistogramChunkRows - 1) / HistogramChunkRows;
        if (chunks <= 1 || _histJobs <= 1)
        {
            var partial = chunks > 1 ? RentHistogram() : null;
            AccumulateHistogram(histogram, partial, gradients, hessians, start, count);
            ReturnHistogram(partial);
            return;
        }

        var partials = new double[chunks - 1][];
        for (int c = 0; c < partials.Length; c++) partials[c] = RentHistogram();
        Parallel.For(0, chunks, new ParallelOptions { MaxDegreeOfParallelism = _histJobs }, c =>
        {
            int chunkStart = start + c * HistogramChunkRows;
            AccumulateRows(c == 0 ? histogram : partials[c - 1], gradients, hessians, chunkStart,
                Math.Min(HistogramChunkRows, start + count - chunkStart));
        });

        foreach (var partial in partials)
        {
            AddScaled(histogram, histogram, partial, 1.0);
            ReturnHistogram(partial);
        }
    }

    /// <summary><paramref name="target"/> = <paramref name="a"/> + <paramref name="sign"/> * <paramref name="b"/>, bin by bin.</summary>
    private void AddScaled(double[] target, double[] a, double[] b, double sign)
    {
        int length = HistogramLength;
        ref double t = ref MemoryMarshal.GetReference(new Span<double>(target, 0, length));
        ref double x = ref MemoryMarshal.GetReference(new ReadOnlySpan<double>(a, 0, length));
        ref double y = ref MemoryMarshal.GetReference(new ReadOnlySpan<double>(b, 0, length));
        if (sign > 0)
        {
            for (int k = 0; k < length; k++) Unsafe.Add(ref t, k) = Unsafe.Add(ref x, k) + Unsafe.Add(ref y, k);
        }
        else
        {
            for (int k = 0; k < length; k++) Unsafe.Add(ref t, k) = Unsafe.Add(ref x, k) - Unsafe.Add(ref y, k);
        }
    }

    private void AccumulateRows(double[] histogram, double[] gradients, double[] hessians, int start, int count)
    {
        Array.Clear(histogram, 0, HistogramLength);
        if (_binBytes.Length > 0)
            AccumulateRows(histogram, gradients, hessians, start, count, new ReadOnlySpan<byte>(_binBytes));
        else
            AccumulateRows(histogram, gradients, hessians, start, count, new ReadOnlySpan<ushort>(_binShorts));
    }

    /// <summary>
    /// The accumulation loop, through refs: the indices are in range by construction (bins below each feature's bin
    /// count, samples below the row count), and span bounds checks cost a third of the loop.
    /// </summary>
    private void AccumulateRows<TBin>(double[] histogram, double[] gradients, double[] hessians, int start, int count,
        ReadOnlySpan<TBin> bins) where TBin : struct
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
            for (int j = 0; j < features; j++)
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
