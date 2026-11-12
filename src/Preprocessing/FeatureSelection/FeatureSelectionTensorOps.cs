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

    /// <summary>
    /// Every column of an [n, p] tensor scaled to <c>(x - min) / (max - min)</c>; a column whose range is below
    /// <paramref name="minimumRange"/> becomes 0 (it never differs between rows).
    /// </summary>
    public static Tensor<T> RangeScaled<T>(Tensor<T> x, double minimumRange = 1e-10)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        if (n == 0)
            return x;

        var min = ColumnMin(x);
        var range = engine.TensorSubtract(ColumnMax(x), min);
        var scale = FilterStatisticTensorOps.GuardedDivide(
            FilterStatisticTensorOps.FilledLike(range, ops.One), range, minimumRange, ops.Zero);
        return engine.TensorMultiply(
            engine.TensorSubtract(x, FilterStatisticTensorOps.RowBroadcast(min, n)),
            FilterStatisticTensorOps.RowBroadcast(scale, n));
    }

    /// <summary>The given rows of an [n, p] tensor ([rows.Length, p]).</summary>
    public static Tensor<T> Rows<T>(Tensor<T> x, int[] rows) =>
        AiDotNetEngine.Current.TensorIndexSelect(x, new Tensor<int>((int[])rows.Clone(), new[] { rows.Length }), 0);

    /// <summary>The given entries of a 1-D tensor.</summary>
    public static Tensor<T> Entries<T>(Tensor<T> x, int[] positions) =>
        AiDotNetEngine.Current.TensorIndexSelect(x, new Tensor<int>((int[])positions.Clone(), new[] { positions.Length }), 0);

    /// <summary>A 0/1 [rows.Length, n] mask with a 1 at (r, rows[r]): each sampled row's own column.</summary>
    public static Tensor<T> SelfMask<T>(int[] rows, int n)
    {
        if (rows.Length == 0)
            return new Tensor<T>(new[] { 0, n });
        return AiDotNetEngine.Current.TensorOneHot<T>(new Tensor<int>((int[])rows.Clone(), new[] { rows.Length }), n);
    }

    /// <summary>
    /// <paramref name="values"/> where <paramref name="keep"/> is non-zero, <paramref name="fill"/> elsewhere.
    /// </summary>
    public static Tensor<T> KeepOrFill<T>(Tensor<T> values, Tensor<T> keep, T fill) =>
        AiDotNetEngine.Current.TensorWhere(keep, values, FilterStatisticTensorOps.FilledLike(values, fill));

    /// <summary>
    /// Per row of an [m, n] tensor, the first (lowest) column holding the row minimum, and a 0/1 [m] flag that is 1
    /// where that minimum is finite (the row had a candidate at all).
    /// </summary>
    public static (int[] Index, Tensor<T> Found) FirstArgMinRows<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = values.Shape[0];
        int n = values.Shape[1];
        if (m == 0 || n == 0)
            return (new int[m], new Tensor<T>(new[] { m }));

        var rowMin = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(values), new[] { 1 }, true));
        var isMin = engine.TensorEquals(values, engine.TensorBroadcastTo(rowMin, new[] { m, n }));

        // n - column is distinct per column, so the largest masked key is the first minimum, whatever the argmax tie rule
        var key = engine.TensorMultiply(isMin, FilterStatisticTensorOps.RowBroadcast(
            engine.TensorArange<T>(n, 0, -1), m));
        var index = engine.TensorArgMax(key, 1).ToArray();
        var found = engine.TensorLessThan(engine.Reshape(rowMin, new[] { m }), ops.FromDouble(double.PositiveInfinity));
        return (index, found);
    }

    /// <summary>
    /// The <paramref name="k"/> smallest entries of every row of an [m, n] tensor, in ascending order with ties to the
    /// lower column (like a stable <c>OrderBy(distance).Take(k)</c>): per neighbour slot the column index and a 0/1
    /// [m] flag that is 1 where the row still had a finite candidate for that slot.
    /// </summary>
    public static List<(int[] Index, Tensor<T> Found)> NearestPerRow<T>(Tensor<T> values, int k)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = values.Shape.Length > 1 ? values.Shape[1] : 0;
        var slots = new List<(int[] Index, Tensor<T> Found)>();
        var remaining = values;
        var infinity = FilterStatisticTensorOps.FilledLike(values, ops.FromDouble(double.PositiveInfinity));
        for (int slot = 0; slot < Math.Min(k, n); slot++)
        {
            var (index, found) = FirstArgMinRows(remaining);
            slots.Add((index, found));
            var taken = engine.TensorOneHot<T>(new Tensor<int>(index, new[] { index.Length }), n);
            remaining = engine.TensorWhere(taken, infinity, remaining);
        }

        return slots;
    }

    /// <summary>
    /// ReliefF feature weights (Kononenko 1994; Robnik-Sikonja and Kononenko 2003, Fig. 3): for every sampled
    /// instance, the <paramref name="k"/> nearest hits lower each weight by their mean feature difference and the
    /// <paramref name="k"/> nearest misses of every other class C raise it by theirs, weighted by
    /// <c>P(C) / (1 - P(class))</c>; everything is divided by <c>m k</c>. Distances are the sum of the feature
    /// differences (Manhattan) on <paramref name="scaled"/>, which holds range-scaled features. Neighbours tie to
    /// the lower index.
    /// </summary>
    public static Tensor<T> ReliefFWeights<T>(Tensor<T> scaled, int[] classIndex, int classCount, int[] samples, int k)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = scaled.Shape[0];
        int p = scaled.Shape[1];
        int m = samples.Length;
        var total = new Tensor<T>(new[] { p });
        if (m == 0 || k <= 0)
            return total;

        T infinity = ops.FromDouble(double.PositiveInfinity);
        var sampledRows = Rows(scaled, samples);
        var distance = engine.TensorCDist(sampledRows, scaled, 1.0);
        var oneHot = FilterStatisticTensorOps.ClassOneHot<T>(classIndex, classCount);
        var sampleOneHot = Rows(oneHot, samples);
        var notSelf = FilterStatisticTensorOps.OneMinus(SelfMask<T>(samples, n));
        var priors = engine.TensorDivideScalar(engine.ReduceSum(oneHot, new[] { 0 }, false), ops.FromDouble(n));
        var otherMass = FilterStatisticTensorOps.OneMinus(
            engine.Reshape(engine.TensorMatMul(sampleOneHot, engine.Reshape(priors, new[] { classCount, 1 })), new[] { m }));

        // nearest hits: same class, not the instance itself
        var hitKeep = engine.TensorMultiply(engine.TensorMatMul(sampleOneHot, engine.TensorTranspose(oneHot)), notSelf);
        foreach (var (index, found) in NearestPerRow(KeepOrFill(distance, hitKeep, infinity), k))
            total = engine.TensorSubtract(total, WeightedRowSum(found, NeighbourDifference(sampledRows, scaled, index)));

        // nearest misses of every other class, weighted by P(C) / (1 - P(class of the instance))
        for (int c = 0; c < classCount; c++)
        {
            var inClass = Column(oneHot, c);
            var otherClass = FilterStatisticTensorOps.OneMinus(Column(sampleOneHot, c));
            var classPrior = engine.TensorBroadcastTo(Entries(priors, new[] { c }), new[] { m });
            var factor = engine.TensorMultiply(
                FilterStatisticTensorOps.GuardedDivide(classPrior, otherMass, 0.0, ops.Zero), otherClass);
            var missKeep = FilterStatisticTensorOps.RowBroadcast(inClass, m);
            foreach (var (index, found) in NearestPerRow(KeepOrFill(distance, missKeep, infinity), k))
                total = engine.TensorAdd(total, WeightedRowSum(
                    engine.TensorMultiply(found, factor), NeighbourDifference(sampledRows, scaled, index)));
        }

        return engine.TensorDivideScalar(total, ops.FromDouble((double)m * k));
    }

    /// <summary>
    /// <c>sum_{i, j} pairWeight[i, j] |x[i, a] - x[j, a]|</c> for every feature a of an [n, p] tensor, with an
    /// [n, n] pair weighting ([p]). Evaluated one feature at a time so only n x n values are ever held.
    /// </summary>
    public static Tensor<T> PairWeightedDifference<T>(Tensor<T> x, Tensor<T> pairWeight)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int p = x.Shape[1];
        if (n == 0 || p == 0)
            return new Tensor<T>(new[] { p });

        var perFeature = new Tensor<T>[p];
        for (int a = 0; a < p; a++)
        {
            var column = Column(x, a);
            var difference = engine.TensorAbs(engine.TensorSubtract(
                FilterStatisticTensorOps.ColumnBroadcast(column, n), FilterStatisticTensorOps.RowBroadcast(column, n)));
            perFeature[a] = engine.ReduceSum(engine.Reshape(engine.TensorMultiply(difference, pairWeight), new[] { 1, n * n }), new[] { 1 }, false);
        }

        return engine.TensorConcatenate(perFeature, 0);
    }

    /// <summary>
    /// The SURF-family pair weighting for near neighbours: <c>-hit[i, j] / (n max(1, hits_i)) +
    /// miss[i, j] / (n max(1, misses_i))</c>, from 0/1 [n, n] hit and miss masks.
    /// </summary>
    public static Tensor<T> NearNeighbourPairWeight<T>(Tensor<T> hits, Tensor<T> misses)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = hits.Shape[0];
        T scale = ops.FromDouble(n);
        var hitCount = engine.TensorMultiplyScalar(engine.TensorMax(engine.ReduceSum(hits, new[] { 1 }, false), ops.One), scale);
        var missCount = engine.TensorMultiplyScalar(engine.TensorMax(engine.ReduceSum(misses, new[] { 1 }, false), ops.One), scale);
        return engine.TensorSubtract(
            engine.TensorDivide(misses, FilterStatisticTensorOps.ColumnBroadcast(missCount, n)),
            engine.TensorDivide(hits, FilterStatisticTensorOps.ColumnBroadcast(hitCount, n)));
    }

    /// <summary>|rows - x[neighbour]| feature by feature ([m, p]).</summary>
    public static Tensor<T> NeighbourDifference<T>(Tensor<T> rows, Tensor<T> x, int[] neighbour)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorAbs(engine.TensorSubtract(rows, Rows(x, neighbour)));
    }

    /// <summary><c>sum_r weight[r] values[r, :]</c> of an [m, p] tensor ([p]).</summary>
    public static Tensor<T> WeightedRowSum<T>(Tensor<T> weight, Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        int m = values.Shape[0];
        int p = values.Shape[1];
        return engine.Reshape(engine.TensorMatMul(engine.Reshape(weight, new[] { 1, m }), values), new[] { p });
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
