using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Helpers;

/// <summary>
/// Whole-array engine helpers for the information-theoretic feature selectors: equal-width and rank
/// discretization, contingency tables built with one bincount, and entropy / mutual information over them.
/// </summary>
internal static class FeatureSelectionInformationOps
{
    /// <summary>
    /// Equal-width bin index of every entry of a 2-D tensor ([n, p]), per column:
    /// <c>min(floor((x - min) / (max - min + 1e-10) * bins), bins - 1)</c>.
    /// </summary>
    public static Tensor<int> EqualWidthBins<T>(Tensor<T> x, int bins)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = x.Shape[1];
        var min = engine.Reshape(FeatureSelectionTensorOps.ColumnMin(x), new[] { 1, p });
        var max = engine.Reshape(FeatureSelectionTensorOps.ColumnMax(x), new[] { 1, p });
        var range = engine.TensorAddScalar(engine.TensorSubtract(max, min), ops.FromDouble(1e-10));
        var scaled = engine.TensorMultiplyScalar(engine.TensorDivide(engine.TensorSubtract(x, min), range), ops.FromDouble(bins));
        return engine.TensorMin(engine.TensorFloor(scaled), ops.FromDouble(bins - 1)).Cast<int>();
    }

    /// <summary>
    /// Equal-width bin index of every entry of a 2-D tensor ([n, p]), per column, over the exact range:
    /// <c>min(floor((x - min) / (max - min) * bins), bins - 1)</c>, and bin 0 for a column whose range is not above 1e-10.
    /// </summary>
    public static Tensor<int> EqualWidthBinsExactRange<T>(Tensor<T> x, int bins)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var min = engine.Reshape(FeatureSelectionTensorOps.ColumnMin(x), new[] { 1, p });
        var max = engine.Reshape(FeatureSelectionTensorOps.ColumnMax(x), new[] { 1, p });
        var range = engine.TensorSubtract(max, min);
        var spread = engine.TensorBroadcastTo(engine.TensorGreaterThan(range, ops.FromDouble(1e-10)), new[] { n, p });
        var safeRange = engine.TensorWhere(engine.TensorGreaterThan(range, ops.FromDouble(1e-10)), range, FeatureSelectionTensorOps.Filled(new[] { 1, p }, ops.One));
        var scaled = engine.TensorMultiplyScalar(engine.TensorDivide(engine.TensorSubtract(x, min), safeRange), ops.FromDouble(bins));
        var binned = engine.TensorMin(engine.TensorFloor(scaled), ops.FromDouble(bins - 1));
        return engine.TensorMultiply(binned, spread).Cast<int>();
    }

    /// <summary>
    /// Equal-width bin index of every entry of a 2-D tensor ([n, p]), per column, with a padded bin width:
    /// <c>min(floor((x - min) / ((max - min) / bins + 1e-10)), bins - 1)</c>.
    /// </summary>
    public static Tensor<int> EqualWidthBinsPaddedWidth<T>(Tensor<T> x, int bins)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = x.Shape[1];
        var min = engine.Reshape(FeatureSelectionTensorOps.ColumnMin(x), new[] { 1, p });
        var max = engine.Reshape(FeatureSelectionTensorOps.ColumnMax(x), new[] { 1, p });
        var width = engine.TensorAddScalar(
            engine.TensorDivideScalar(engine.TensorSubtract(max, min), ops.FromDouble(bins)), ops.FromDouble(1e-10));
        var scaled = engine.TensorDivide(engine.TensorSubtract(x, min), width);
        return engine.TensorMin(engine.TensorFloor(scaled), ops.FromDouble(bins - 1)).Cast<int>();
    }

    /// <summary>
    /// Mutual information in bits between every column of <paramref name="a"/> ([n, p], values in 0..aCard-1)
    /// and the discrete variable <paramref name="b"/> ([n] or [n, 1], values in 0..bCard-1), clamped at 0. Returns [p].
    /// </summary>
    public static Tensor<T> MutualInformationBits<T>(Tensor<int> a, int aCard, Tensor<int> b, int bCard)
    {
        var engine = AiDotNetEngine.Current;
        int n = a.Shape[0];
        var nats = MutualInformation(JointCounts<T>(a, aCard, b, bCard), n);
        return engine.TensorMax(ToBits(nats), MathHelper.GetNumericOperations<T>().Zero);
    }

    /// <summary>Equal-width bin index of every entry of a 1-D tensor ([n] to [n, 1]), see <see cref="EqualWidthBins{T}(Tensor{T}, int)"/>.</summary>
    public static Tensor<int> EqualWidthBins<T>(Vector<T> values, int bins) =>
        EqualWidthBins(AiDotNetEngine.Current.Reshape(Tensor<T>.FromVector(values), new[] { values.Length, 1 }), bins);

    /// <summary>
    /// Equal-frequency (rank) bin index of every entry of a 1-D tensor: the entry of ascending rank r (ties in index
    /// order) goes to bin <c>min(r * bins / n, bins - 1)</c>. Returns [n].
    /// </summary>
    public static Tensor<int> RankBins<T>(Tensor<T> values, int bins)
    {
        int n = values.Length;
        var order = FeatureSelectionTensorOps.StableOrder(values, descending: false);
        var ranks = AiDotNetEngine.Current.TensorArgsort(new Tensor<int>(order, new[] { n }), 0, false);
        var binOfRank = new int[n];
        for (int r = 0; r < n; r++)
            binOfRank[r] = Math.Min((int)((long)r * bins / n), bins - 1);
        return AiDotNetEngine.Current.TensorIndexSelect(new Tensor<int>(binOfRank, new[] { n }), ranks, 0);
    }

    /// <summary>
    /// Contingency tables of every column of <paramref name="a"/> ([n, p], values in 0..aCard-1) against the single
    /// discrete variable <paramref name="b"/> ([n] or [n, 1], values in 0..bCard-1), from one bincount.
    /// Returns the counts as [p, aCard, bCard].
    /// </summary>
    public static Tensor<T> JointCounts<T>(Tensor<int> a, int aCard, Tensor<int> b, int bCard)
    {
        var engine = AiDotNetEngine.Current;
        int n = a.Shape[0];
        int p = a.Shape[1];
        int cells = aCard * bCard;
        if (n == 0 || p == 0)
            return new Tensor<T>(new[] { p, aCard, bCard });

        var bColumn = engine.TensorBroadcastTo(engine.Reshape(b, new[] { n, 1 }), new[] { n, p });
        var offsets = new int[p];
        for (int j = 0; j < p; j++)
            offsets[j] = j * cells;
        var columnOffsets = engine.TensorBroadcastTo(new Tensor<int>(offsets, new[] { 1, p }), new[] { n, p });
        var keys = engine.TensorAdd(engine.TensorAdd(engine.TensorMultiplyScalar(a, bCard), bColumn), columnOffsets);
        var counts = engine.TensorBinCount(engine.Reshape(keys, new[] { n * p }), p * cells);
        return engine.Reshape(counts, new[] { p, aCard, bCard }).Cast<T>();
    }

    /// <summary>
    /// The joint discrete variable of two binned columns, <c>a * bCard + b</c> (cardinality aCard * bCard).
    /// <paramref name="a"/> is [n, 1] and <paramref name="b"/> is [n, q]; the result is [n, q].
    /// </summary>
    public static Tensor<int> Combine(Tensor<int> a, Tensor<int> b, int bCard)
    {
        var engine = AiDotNetEngine.Current;
        int n = b.Shape[0];
        int q = b.Shape[1];
        return engine.TensorAdd(
            engine.TensorMultiplyScalar(engine.TensorBroadcastTo(engine.Reshape(a, new[] { n, 1 }), new[] { n, q }), bCard), b);
    }

    /// <summary>
    /// Mutual information (natural log) of each [aCard, bCard] contingency table in <paramref name="counts"/>
    /// ([p, aCard, bCard]) over <paramref name="n"/> samples:
    /// <c>sum p(a,b) log(p(a,b) / (p(a) p(b)))</c> over the non-empty cells. Returns [p].
    /// </summary>
    public static Tensor<T> MutualInformation<T>(Tensor<T> counts, int n)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var pab = engine.TensorDivideScalar(counts, ops.FromDouble(n));
        var pa = engine.ReduceSum(pab, new[] { 2 }, true);
        var pb = engine.ReduceSum(pab, new[] { 1 }, true);
        var shape = counts.Shape.ToArray();
        var expected = engine.TensorMultiply(engine.TensorBroadcastTo(pa, shape), engine.TensorBroadcastTo(pb, shape));
        return SumNonEmpty(pab, engine.TensorLog(SafeRatio(pab, expected)));
    }

    /// <summary>
    /// Conditional entropy H(B | A) (natural log) of each [aCard, bCard] contingency table in
    /// <paramref name="counts"/> ([p, aCard, bCard]): <c>-sum p(a) sum p(b|a) log p(b|a)</c>. Returns [p].
    /// </summary>
    public static Tensor<T> ConditionalEntropy<T>(Tensor<T> counts, int n)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var aCounts = engine.ReduceSum(counts, new[] { 2 }, true);
        var pba = SafeRatio(counts, aCounts);
        var pa = engine.TensorDivideScalar(aCounts, ops.FromDouble(n));
        var weighted = engine.TensorMultiply(engine.TensorBroadcastTo(pa, counts.Shape.ToArray()), pba);
        return engine.TensorNegate(SumNonEmpty(weighted, engine.TensorLog(pba), counts));
    }

    /// <summary>
    /// Entropy (natural log) of each discrete distribution held as counts along the last axis of a [p, k] tensor
    /// over <paramref name="n"/> samples: <c>-sum p log p</c> over the non-empty cells. Returns [p].
    /// </summary>
    public static Tensor<T> Entropy<T>(Tensor<T> counts, int n)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var probabilities = engine.TensorDivideScalar(counts, ops.FromDouble(n));
        var safe = engine.TensorWhere(
            engine.TensorGreaterThan(counts, ops.Zero), probabilities, FeatureSelectionTensorOps.Filled(counts.Shape.ToArray(), ops.One));
        var terms = engine.TensorMultiply(probabilities, engine.TensorLog(safe));
        return engine.TensorNegate(engine.ReduceSum(terms, new[] { counts.Rank - 1 }, false));
    }

    /// <summary>Converts a natural-log information value to bits.</summary>
    public static Tensor<T> ToBits<T>(Tensor<T> nats) =>
        AiDotNetEngine.Current.TensorDivideScalar(nats, MathHelper.GetNumericOperations<T>().FromDouble(Math.Log(2)));

    /// <summary><c>numerator / denominator</c> where the denominator is positive, 1 elsewhere (so its log is 0).</summary>
    private static Tensor<T> SafeRatio<T>(Tensor<T> numerator, Tensor<T> denominator)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = numerator.Shape.ToArray();
        var den = engine.TensorBroadcastTo(denominator, shape);
        var positive = engine.TensorGreaterThan(den, ops.Zero);
        var safeDen = engine.TensorWhere(positive, den, FeatureSelectionTensorOps.Filled(shape, ops.One));
        var ratio = engine.TensorDivide(numerator, safeDen);
        var nonEmpty = engine.TensorGreaterThan(numerator, ops.Zero);
        return engine.TensorWhere(nonEmpty, ratio, FeatureSelectionTensorOps.Filled(shape, ops.One));
    }

    /// <summary>Sums <c>weight * logTerm</c> over the last two axes of [p, a, b], skipping the empty cells.</summary>
    private static Tensor<T> SumNonEmpty<T>(Tensor<T> weight, Tensor<T> logTerm, Tensor<T>? cells = null)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var nonEmpty = engine.TensorGreaterThan(cells ?? weight, ops.Zero);
        var terms = engine.TensorMultiply(nonEmpty, engine.TensorMultiply(weight, logTerm));
        return engine.ReduceSum(terms, new[] { 1, 2 }, false);
    }
}
