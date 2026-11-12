using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Whole-array engine helpers shared by the filter feature selectors on top of
/// <see cref="FilterStatisticTensorOps"/>: per-column scattering, equal-width discretisation, contingency tables,
/// entropies and tie-averaged ranks, all generic in T.
/// </summary>
internal static class FeatureSelectionTensorOps
{
    /// <summary>
    /// A [p] tensor holding <paramref name="values"/> at the positions <paramref name="columns"/> and
    /// <paramref name="fill"/> everywhere else.
    /// </summary>
    public static Tensor<T> ScatterToFeatures<T>(Tensor<T> values, int[] columns, int p, T fill)
    {
        var result = FilterStatisticTensorOps.Filled(new[] { p }, fill);
        if (columns.Length == 0)
            return result;
        return AiDotNetEngine.Current.TensorScatter(
            result, new Tensor<int>((int[])columns.Clone(), new[] { columns.Length }), values, 0);
    }

    /// <summary>
    /// The <paramref name="k"/> best of the given candidate columns by score (ties to the earlier candidate, like
    /// <c>columns.OrderByDescending(c =&gt; scores[c]).Take(k)</c>), returned in ascending column order.
    /// </summary>
    public static int[] TopKOfColumns<T>(Tensor<T> scores, int[] columns, int k)
    {
        if (columns.Length == 0 || k <= 0)
            return Array.Empty<int>();

        var candidateScores = AiDotNetEngine.Current.TensorIndexSelect(
            scores, new Tensor<int>((int[])columns.Clone(), new[] { columns.Length }), 0);
        var order = FilterStatisticTensorOps.StableOrder(candidateScores, descending: true);
        var selected = order.Take(Math.Min(k, columns.Length)).Select(position => columns[position]).ToArray();
        Array.Sort(selected);
        return selected;
    }

    /// <summary>
    /// scikit-learn's <c>SelectPercentile</c> rule, ascending positions: every feature scoring above the
    /// <c>(100 - percentile)</c>-th percentile of the scores (numpy.percentile, linear interpolation), then features
    /// tied at that threshold in index order until <c>(int)(p * percentile / 100)</c> are kept. NaN scores count as
    /// the lowest value; percentile 100 keeps every feature and 0 keeps none.
    /// </summary>
    public static int[] PercentileSupport<T>(Tensor<T> scores, double percentile)
    {
        var engine = AiDotNetEngine.Current;
        int p = scores.Length;
        if (p == 0 || percentile <= 0)
            return Array.Empty<int>();
        if (percentile >= 100)
            return Enumerable.Range(0, p).ToArray();

        var clean = FilterStatisticTensorOps.NaNAsLowest(scores);
        var threshold = engine.TensorQuantile(clean, 1.0 - percentile / 100.0, 0, false, QuantileInterpolation.Linear);
        var thresholdGrid = engine.TensorBroadcastTo(engine.Reshape(threshold, new[] { 1 }), new[] { p });
        var above = FilterStatisticTensorOps.MaskPositions(engine.TensorGreaterThan(clean, thresholdGrid));
        var ties = FilterStatisticTensorOps.MaskPositions(engine.TensorEquals(clean, thresholdGrid));
        int maxFeatures = (int)(p * percentile / 100.0);
        int keptTies = Math.Max(0, Math.Min(ties.Length, maxFeatures - above.Length));
        var selected = above.Concat(ties.Take(keptTies)).ToArray();
        Array.Sort(selected);
        return selected;
    }

    /// <summary>The 0/1 mask of <c>values &lt;= limit</c> (NaN is never at most the limit).</summary>
    public static Tensor<T> AtMost<T>(Tensor<T> values, T limit)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorAdd(engine.TensorLessThan(values, limit), engine.TensorEquals(values, limit));
    }

    /// <summary>A support mask of <paramref name="p"/> features with the given (distinct) positions set.</summary>
    public static bool[] SupportMask(int[] selected, int p)
    {
        var mask = new bool[p];
        foreach (int index in selected)
            mask[index] = true;
        return mask;
    }

    /// <summary>One column of an [n, p] tensor as an [n] tensor.</summary>
    public static Tensor<T> Column<T>(Tensor<T> x, int column)
    {
        var engine = AiDotNetEngine.Current;
        return engine.Reshape(
            engine.TensorIndexSelect(x, new Tensor<int>(new[] { column }, new[] { 1 }), 1), new[] { x.Shape[0] });
    }

    /// <summary>Column minima of an [n, p] tensor ([p]).</summary>
    public static Tensor<T> ColumnMin<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(x), new[] { 0 }, false));
    }

    /// <summary>Column maxima of an [n, p] tensor ([p]).</summary>
    public static Tensor<T> ColumnMax<T>(Tensor<T> x) => AiDotNetEngine.Current.ReduceMax(x, new[] { 0 }, false);

    /// <summary>
    /// Equal-width bin of every entry of an [n, p] tensor, per column:
    /// <c>min((int)((x - min) / width), bins - 1)</c> with <c>width = (max - min) / bins</c>, or width 1 when the
    /// column range is below <paramref name="minimumWidth"/> times <paramref name="bins"/> (a constant column
    /// falls in bin 0). Returned as an [n, p] int tensor.
    /// </summary>
    public static Tensor<int> EqualWidthBins<T>(Tensor<T> x, int bins, double minimumWidth = 1e-10)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        if (n == 0)
            return new Tensor<int>(x.Shape.ToArray());

        var min = ColumnMin(x);
        var width = engine.TensorDivideScalar(engine.TensorSubtract(ColumnMax(x), min), ops.FromDouble(bins));
        var degenerate = engine.TensorLessThan(width, ops.FromDouble(minimumWidth));
        width = engine.TensorWhere(degenerate, FilterStatisticTensorOps.FilledLike(width, ops.One), width);
        return ToBinIndex(engine.TensorDivide(
            engine.TensorSubtract(x, FilterStatisticTensorOps.RowBroadcast(min, n)),
            FilterStatisticTensorOps.RowBroadcast(width, n)), bins);
    }

    /// <summary>
    /// Truncates non-negative bin positions to integers (C# <c>(int)</c> casts truncate toward zero) and clamps
    /// them to <c>[0, bins - 1]</c>.
    /// </summary>
    public static Tensor<int> ToBinIndex<T>(Tensor<T> position, int bins)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var truncated = engine.TensorTrunc(position);
        return engine.TensorClamp(truncated, ops.Zero, ops.FromDouble(bins - 1)).Cast<int>();
    }

    /// <summary>
    /// Joint counts of two discrete codes for every feature at once: <paramref name="a"/> is an [n, p] int tensor of
    /// codes in <c>[0, aLevels)</c>, <paramref name="bOneHot"/> an [n, bLevels] membership of a code shared by all
    /// features. Returns [p, aLevels, bLevels] counts in T.
    /// </summary>
    public static Tensor<T> JointCounts<T>(Tensor<int> a, int aLevels, Tensor<T> bOneHot)
    {
        var engine = AiDotNetEngine.Current;
        int n = a.Shape[0];
        int p = a.Shape[1];
        int bLevels = bOneHot.Shape[1];
        if (n == 0 || p == 0)
            return new Tensor<T>(new[] { p, aLevels, bLevels });

        // one-hot of (feature, code) in a single axis: [n, p * aLevels]
        var aOneHot = engine.Reshape(OneHot<T>(a, aLevels), new[] { n, p * aLevels });
        var counts = engine.TensorMatMul(engine.TensorTranspose(aOneHot), bOneHot);
        return engine.Reshape(counts, new[] { p, aLevels, bLevels });
    }

    /// <summary>
    /// Joint counts of two per-feature codes: <paramref name="a"/> and <paramref name="b"/> are [n, p] int tensors
    /// of codes in <c>[0, aLevels)</c> and <c>[0, bLevels)</c>. Returns [p, aLevels, bLevels] counts in T.
    /// </summary>
    public static Tensor<T> PairedJointCounts<T>(Tensor<int> a, int aLevels, Tensor<int> b, int bLevels)
    {
        var engine = AiDotNetEngine.Current;
        int n = a.Shape[0];
        int p = a.Shape[1];
        if (n == 0 || p == 0)
            return new Tensor<T>(new[] { p, aLevels, bLevels });

        // one code per (row, feature): a * bLevels + b, counted per feature with a feature-major one-hot
        var combined = engine.TensorAdd(engine.TensorMultiplyScalar(a, bLevels), b);
        var counts = engine.ReduceSum(OneHot<T>(combined, aLevels * bLevels), new[] { 0 }, false);
        return engine.Reshape(counts, new[] { p, aLevels, bLevels });
    }

    /// <summary>Counts of each code per feature: [n, p] int codes in <c>[0, levels)</c> to [p, levels] counts.</summary>
    public static Tensor<T> CodeCounts<T>(Tensor<int> codes, int levels)
    {
        var engine = AiDotNetEngine.Current;
        int n = codes.Shape[0];
        int p = codes.Shape[1];
        if (n == 0 || p == 0)
            return new Tensor<T>(new[] { p, levels });
        return engine.ReduceSum(OneHot<T>(codes, levels), new[] { 0 }, false);
    }

    /// <summary>One-hot of an [n, p] int code tensor: [n, p, levels] in T.</summary>
    public static Tensor<T> OneHot<T>(Tensor<int> codes, int levels)
    {
        var engine = AiDotNetEngine.Current;
        int n = codes.Shape[0];
        int p = codes.Shape[1];
        var flat = engine.TensorOneHot<T>(engine.Reshape(codes, new[] { n * p }), levels);
        return engine.Reshape(flat, new[] { n, p, levels });
    }

    /// <summary>
    /// Shannon entropy <c>-sum q log q</c> (natural log) of the distributions <c>q = counts / total</c> along the last
    /// axis, with <c>0 log 0 = 0</c>. <paramref name="total"/> broadcasts as a scalar.
    /// </summary>
    public static Tensor<T> Entropy<T>(Tensor<T> counts, T total, int axis)
    {
        var engine = AiDotNetEngine.Current;
        var q = engine.TensorDivideScalar(counts, total);
        return engine.TensorNegate(engine.ReduceSum(engine.TensorXlogy(q, q), new[] { axis }, false));
    }

    /// <summary>
    /// Mutual information <c>sum q_ab log(q_ab / (q_a q_b))</c> (natural log) of [p, A, B] joint counts over
    /// <paramref name="total"/> samples, per feature ([p]), from <c>H(A) + H(B) - H(A, B)</c>.
    /// </summary>
    public static Tensor<T> MutualInformation<T>(Tensor<T> jointCounts, T total)
    {
        var engine = AiDotNetEngine.Current;
        int p = jointCounts.Shape[0];
        int a = jointCounts.Shape[1];
        int b = jointCounts.Shape[2];
        var hA = Entropy(engine.ReduceSum(jointCounts, new[] { 2 }, false), total, 1);
        var hB = Entropy(engine.ReduceSum(jointCounts, new[] { 1 }, false), total, 1);
        var hAB = Entropy(engine.Reshape(jointCounts, new[] { p, a * b }), total, 1);
        return engine.TensorSubtract(engine.TensorAdd(hA, hB), hAB);
    }

    /// <summary>
    /// Tie-averaged ranks (1-based, scipy.stats.rankdata "average") of a 1-D tensor: tied values share the mean of
    /// the ranks they span.
    /// </summary>
    public static Tensor<T> AverageRanks<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        if (values.Length == 0)
            return values;

        // for value v: below = #{< v}, atMost = #{<= v}; the tied block spans ranks below + 1 .. atMost
        var sorted = engine.TensorSort(values, 0, false).Item1;
        var below = engine.TensorSearchSorted(sorted, values, false).Cast<T>();
        var atMost = engine.TensorSearchSorted(sorted, values, true).Cast<T>();
        return engine.TensorMultiplyScalar(engine.TensorAddScalar(engine.TensorAdd(below, atMost), ops.One), ops.FromDouble(0.5));
    }

    /// <summary>Tie-averaged ranks of every column of an [n, p] tensor, each column ranked on its own ([n, p]).</summary>
    public static Tensor<T> ColumnAverageRanks<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int p = x.Shape[1];
        if (n == 0 || p == 0)
            return new Tensor<T>(new[] { n, p });

        var columns = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
            columns[j] = AverageRanks(Column(x, j));
        return engine.TensorStack(columns, 1);
    }

    /// <summary>
    /// The tie correction term <c>sum (t^3 - t)</c> over the groups of tied values of a 1-D tensor (t = group size),
    /// as used by the Kruskal-Wallis and Mann-Whitney tie corrections.
    /// </summary>
    public static T TieCubeSum<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        if (values.Length == 0)
            return ops.Zero;

        var (_, _, counts) = engine.TensorUniqueWithInfo(values, true, false, true);
        if (counts is null)
            throw new InvalidOperationException("The engine returned no counts for the tied values.");
        var t = counts.Cast<T>();
        return engine.TensorSum(engine.TensorSubtract(engine.TensorMultiply(engine.TensorSquare(t), t), t));
    }
}
