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
        var order = PreprocessingTensorOps.StableOrder(candidateScores, descending: true);
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

        var min = PreprocessingTensorOps.ColumnMin(x);
        var range = engine.TensorSubtract(PreprocessingTensorOps.ColumnMax(x), min);
        // divided by the range (not multiplied by its reciprocal, which rounds differently and moves values that sit
        // exactly on an equal-width bin edge into the bin below)
        return FilterStatisticTensorOps.GuardedDivide(
            engine.TensorSubtract(x, FilterStatisticTensorOps.RowBroadcast(min, n)),
            FilterStatisticTensorOps.RowBroadcast(range, n), minimumRange, ops.Zero);
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

        // n - column is distinct per column, so the largest masked key is the first minimum, whatever the argmax tie
        // rule (kept in int: in Half the keys past 2048 are not distinct and the argmax would pick a later column)
        var key = engine.TensorMultiply(isMin.Cast<int>(), FilterStatisticTensorOps.RowBroadcast(
            engine.TensorArange<int>(n, 0, -1), m));
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
        var notSelf = PreprocessingTensorOps.OneMinus(SelfMask<T>(samples, n));
        var priors = engine.TensorDivideScalar(Counts(oneHot, 0), ops.FromDouble(n));
        var otherMass = PreprocessingTensorOps.OneMinus(
            engine.Reshape(engine.TensorMatMul(sampleOneHot, engine.Reshape(priors, new[] { classCount, 1 })), new[] { m }));

        // nearest hits: same class, not the instance itself
        var hitKeep = engine.TensorMultiply(engine.TensorMatMul(sampleOneHot, engine.TensorTranspose(oneHot)), notSelf);
        foreach (var (index, found) in NearestPerRow(KeepOrFill(distance, hitKeep, infinity), k))
            total = engine.TensorSubtract(total, WeightedRowSum(found, NeighbourDifference(sampledRows, scaled, index)));

        // nearest misses of every other class, weighted by P(C) / (1 - P(class of the instance))
        T zero = ops.Zero;
        for (int c = 0; c < classCount; c++)
        {
            var inClass = PreprocessingTensorOps.ColumnVector(oneHot, c);
            var otherClass = PreprocessingTensorOps.OneMinus(PreprocessingTensorOps.ColumnVector(sampleOneHot, c));
            var classPrior = engine.TensorBroadcastTo(Entries(priors, new[] { c }), new[] { m });
            var factor = engine.TensorMultiply(
                FilterStatisticTensorOps.GuardedDivide(classPrior, otherMass, 0.0, zero), otherClass);
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
            var column = PreprocessingTensorOps.ColumnVector(x, a);
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
        // divided by the count and then by n: the product n * count overflows Half past 65504
        var hitCount = engine.TensorMax(Counts(hits, 1), ops.One);
        var missCount = engine.TensorMax(Counts(misses, 1), ops.One);
        return engine.TensorDivideScalar(engine.TensorSubtract(
            engine.TensorDivide(misses, FilterStatisticTensorOps.ColumnBroadcast(missCount, n)),
            engine.TensorDivide(hits, FilterStatisticTensorOps.ColumnBroadcast(hitCount, n))), scale);
    }

    /// <summary>
    /// <paramref name="count"/> power-iteration starts of shape [n, 1], each entry <c>random.NextDouble() - 0.5</c>,
    /// drawn start by start and row by row (the order of a scalar loop, so a seeded fit draws what it always did).
    /// </summary>
    public static List<Tensor<T>> CenteredUniformStarts<T>(Random random, int count, int n)
    {
        var engine = AiDotNetEngine.Current;
        T shift = MathHelper.GetNumericOperations<T>().FromDouble(-0.5);
        var starts = new List<Tensor<T>>(Math.Max(count, 0));
        for (int s = 0; s < count; s++)
            starts.Add(engine.TensorAddScalar(Tensor<T>.CreateRandom(random, n, 1), shift));
        return starts;
    }

    /// <summary>Squared Euclidean distances from every row of an [m, d] tensor to every row of a [k, d] one ([m, k]).</summary>
    public static Tensor<T> SquaredDistancesBetween<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorSquare(engine.TensorCDist(a, b, 2.0));
    }

    /// <summary>Squared Euclidean distances between the rows of an [n, p] tensor ([n, n]).</summary>
    public static Tensor<T> SquaredDistances<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorSquare(engine.TensorCDist(x, x, 2.0));
    }

    /// <summary>
    /// A symmetric k-nearest-neighbour graph with Gaussian weights: <c>W[i, j] = exp(-coefficient d_ij^2)</c>
    /// (squared Euclidean d) when j is among the <paramref name="k"/> nearest neighbours of i or i among those of j
    /// (ties to the lower index), 0 otherwise and on the diagonal ([n, n]).
    /// </summary>
    public static Tensor<T> KnnGaussianAffinity<T>(Tensor<T> x, int k, double coefficient)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        if (n == 0)
            return new Tensor<T>(new[] { 0, 0 });

        var squared = SquaredDistances(x);
        var eye = engine.TensorEye<T>(n);
        var candidates = engine.TensorWhere(eye, FilterStatisticTensorOps.FilledLike(squared, ops.FromDouble(double.PositiveInfinity)), squared);
        var neighbour = new Tensor<T>(new[] { n, n });
        foreach (var (index, found) in NearestPerRow(candidates, k))
        {
            var oneHot = engine.TensorOneHot<T>(new Tensor<int>(index, new[] { n }), n);
            neighbour = engine.TensorMax(neighbour, engine.TensorMultiply(oneHot, FilterStatisticTensorOps.ColumnBroadcast(found, n)));
        }

        var linked = engine.TensorMax(neighbour, engine.TensorTranspose(neighbour));
        var kernel = engine.TensorExp(engine.TensorMultiplyScalar(squared, ops.FromDouble(-coefficient)));
        return engine.TensorMultiply(kernel, linked);
    }

    /// <summary>
    /// The 0/1 [n, n] mask of each row's <paramref name="k"/> nearest other rows by the given distances (ties to the
    /// lower index); the diagonal is never set.
    /// </summary>
    public static Tensor<T> KnnMask<T>(Tensor<T> distances, int k)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = distances.Shape[0];
        var candidates = engine.TensorWhere(
            engine.TensorEye<T>(n), FilterStatisticTensorOps.FilledLike(distances, ops.FromDouble(double.PositiveInfinity)), distances);
        var mask = new Tensor<T>(new[] { n, n });
        foreach (var (index, found) in NearestPerRow(candidates, k))
        {
            var oneHot = engine.TensorOneHot<T>(new Tensor<int>(index, new[] { n }), n);
            mask = engine.TensorMax(mask, engine.TensorMultiply(oneHot, FilterStatisticTensorOps.ColumnBroadcast(found, n)));
        }

        return mask;
    }

    /// <summary>
    /// Power iteration with deflation: for every start vector ([n, 1]) in turn, <paramref name="iterations"/> steps
    /// of <c>v = M v</c>, made orthogonal to the vectors already found and normalised. A step whose norm is not above
    /// 1e-10 keeps the previous vector when <paramref name="keepPreviousOnZeroNorm"/> is set, otherwise takes the
    /// unnormalised product. Returns the [n, 1] vectors.
    /// </summary>
    public static List<Tensor<T>> PowerIteration<T>(
        Tensor<T> matrix, IReadOnlyList<Tensor<T>> starts, int iterations, bool keepPreviousOnZeroNorm)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = matrix.Shape[0];
        T tolerance = ops.FromDouble(1e-10);
        T one = ops.One;
        var found = new List<Tensor<T>>();
        foreach (var start in starts)
        {
            var v = start;
            for (int iter = 0; iter < iterations; iter++)
            {
                var product = engine.TensorMatMul(matrix, v);
                foreach (var previous in found)
                {
                    var dot = engine.TensorMatMul(engine.TensorTranspose(previous), product);
                    product = engine.TensorSubtract(product, engine.TensorMultiply(previous, engine.TensorBroadcastTo(dot, new[] { n, 1 })));
                }

                var norm = engine.TensorBroadcastTo(
                    engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(product), new[] { 0 }, true)), new[] { n, 1 });
                var moved = engine.TensorGreaterThan(norm, tolerance);
                var safeNorm = engine.TensorWhere(moved, norm, FilterStatisticTensorOps.FilledLike(norm, one));
                v = engine.TensorWhere(moved, engine.TensorDivide(product, safeNorm), keepPreviousOnZeroNorm ? v : product);
            }

            found.Add(v);
        }

        return found;
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

        var min = PreprocessingTensorOps.ColumnMin(x);
        var width = engine.TensorDivideScalar(engine.TensorSubtract(PreprocessingTensorOps.ColumnMax(x), min), ops.FromDouble(bins));
        var degenerate = engine.TensorLessThan(width, ops.FromDouble(minimumWidth));
        width = engine.TensorWhere(degenerate, FilterStatisticTensorOps.FilledLike(width, ops.One), width);
        return ToBinIndex(engine.TensorDivide(
            engine.TensorSubtract(x, FilterStatisticTensorOps.RowBroadcast(min, n)),
            FilterStatisticTensorOps.RowBroadcast(width, n)), bins);
    }

    /// <summary>
    /// Equal-width bin of every entry of an [n, p] tensor on <c>bins - 1</c> steps, per column:
    /// <c>min(bins - 1, (int)((x - min) / (max - min) * (bins - 1)))</c>, or bin 0 for a column whose range is not
    /// above 1e-10. Returned as an [n, p] int tensor.
    /// </summary>
    public static Tensor<int> RangeBins<T>(Tensor<T> x, int bins)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return ToBinIndex(engine.TensorMultiplyScalar(RangeScaled(x), ops.FromDouble(bins - 1)), bins);
    }

    /// <summary>
    /// Equal-width bin of every entry of an [n, p] tensor over the column range, per column:
    /// <c>min(bins - 1, (int)((x - min) / (max - min) * bins))</c>, or bin 0 for a column whose range is below 1e-10.
    /// Returned as an [n, p] int tensor.
    /// </summary>
    public static Tensor<int> ScaledBins<T>(Tensor<T> x, int bins)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return ToBinIndex(engine.TensorMultiplyScalar(RangeScaled(x), ops.FromDouble(bins)), bins);
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
        return engine.Reshape(CountProduct(aOneHot, bOneHot), new[] { p, aLevels, bLevels });
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
        return engine.Reshape(Counts(OneHot<T>(combined, aLevels * bLevels), 0), new[] { p, aLevels, bLevels });
    }

    /// <summary>Counts of each code per feature: [n, p] int codes in <c>[0, levels)</c> to [p, levels] counts.</summary>
    public static Tensor<T> CodeCounts<T>(Tensor<int> codes, int levels)
    {
        var engine = AiDotNetEngine.Current;
        int n = codes.Shape[0];
        int p = codes.Shape[1];
        if (n == 0 || p == 0)
            return new Tensor<T>(new[] { p, levels });
        return Counts(OneHot<T>(codes, levels), 0);
    }

    /// <summary>
    /// The number of nonzero entries of a 0/1 tensor along <paramref name="axis"/>, counted in int and returned in T
    /// (a 0/1 sum in T stops counting at 2048 for Half).
    /// </summary>
    public static Tensor<T> Counts<T>(Tensor<T> mask, int axis) =>
        AiDotNetEngine.Current.ReduceSum(mask.Cast<int>(), new[] { axis }, false).Cast<T>();

    /// <summary>
    /// <c>a^T b</c> of two 0/1 tensors [n, r] and [n, c] ([r, c]): how many rows have both a[., i] and b[., j] set,
    /// counted in int and returned in T (a matmul in T stops counting at 2048 for Half).
    /// </summary>
    public static Tensor<T> CountProduct<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        int n = a.Shape[0];
        int r = a.Shape[1];
        int c = b.Shape[1];
        if (c == 0)
            return new Tensor<T>(new[] { r, 0 });

        var aCounts = a.Cast<int>();
        var bCounts = b.Cast<int>();
        var columns = new Tensor<int>[c];
        for (int j = 0; j < c; j++)
        {
            var bColumn = engine.TensorBroadcastTo(engine.TensorSlice(bCounts, new[] { 0, j }, new[] { n, 1 }), new[] { n, r });
            columns[j] = engine.Reshape(
                engine.ReduceSum(engine.TensorMultiply(aCounts, bColumn), new[] { 0 }, false), new[] { r, 1 });
        }

        return engine.TensorConcatenate(columns, 1).Cast<T>();
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
            columns[j] = AverageRanks(PreprocessingTensorOps.ColumnVector(x, j));
        return engine.TensorStack(columns, 1);
    }

    /// <summary>The tie term <c>sum (t^3 - t)</c> of every column of an [n, p] tensor ([p]).</summary>
    public static Tensor<T> ColumnTieCubeSums<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int p = x.Shape[1];
        if (n == 0 || p == 0)
            return new Tensor<T>(new[] { p });

        var sums = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            var (_, _, counts) = engine.TensorUniqueWithInfo(PreprocessingTensorOps.ColumnVector(x, j), true, false, true);
            if (counts is null)
                throw new InvalidOperationException("The engine returned no counts for the tied values.");
            var t = counts.Cast<T>();
            sums[j] = engine.ReduceSum(engine.TensorSubtract(engine.TensorMultiply(engine.TensorSquare(t), t), t), new[] { 0 }, true);
        }

        return engine.TensorConcatenate(sums, 0);
    }

    /// <summary>
    /// The Kruskal-Wallis H test of every column of <paramref name="x"/> ([n, p]) across the classes of
    /// <paramref name="classOneHot"/> ([n, k]), as scipy.stats.kruskal: tie-averaged ranks,
    /// <c>H = 12 / (n (n + 1)) sum_c R_c^2 / n_c - 3 (n + 1)</c> divided by the tie correction
    /// <c>1 - sum (t^3 - t) / (n^3 - n)</c>, p-value from the chi-squared distribution with k - 1 degrees of freedom.
    /// Fewer than two classes give H = 0 and p = 1; a column of equal values gives H = 0.
    /// </summary>
    public static (Tensor<T> H, Tensor<T> PValues) KruskalWallis<T>(Tensor<T> x, Tensor<T> classOneHot)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        int k = classOneHot.Shape[1];
        if (k < 2 || n < 2)
            return (new Tensor<T>(new[] { p }), FilterStatisticTensorOps.Filled(new[] { p }, ops.One));

        var ranks = ColumnAverageRanks(x);
        var rankSums = engine.TensorMatMul(engine.TensorTranspose(classOneHot), ranks);
        var counts = Counts(classOneHot, 0);
        var weighted = engine.ReduceSum(
            engine.TensorDivide(engine.TensorSquare(rankSums), FilterStatisticTensorOps.ColumnBroadcast(counts, p)),
            new[] { 0 }, false);
        double nd = n;
        var h = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(weighted, ops.FromDouble(12.0 / (nd * (nd + 1)))), ops.FromDouble(-3.0 * (nd + 1)));

        var correction = PreprocessingTensorOps.OneMinus(
            engine.TensorDivideScalar(ColumnTieCubeSums(x), ops.FromDouble(nd * nd * nd - nd)));
        // rounding can leave H a hair below 0 when the classes do not differ at all
        h = engine.TensorMax(FilterStatisticTensorOps.GuardedDivide(h, correction, 0.0, ops.Zero), ops.Zero);
        return (h, FilterStatisticTensorOps.ChiSquareSurvival(h, k - 1));
    }

    /// <summary>
    /// The Mann-Whitney U of the rows in <paramref name="groupMask"/> (0/1 [n], <paramref name="n0"/> rows) against
    /// the other <paramref name="n1"/> rows, for every column of <paramref name="x"/>: <c>U0 = R0 - n0 (n0 + 1) / 2</c>
    /// from tie-averaged ranks ([p]).
    /// </summary>
    public static Tensor<T> RankSumU<T>(Tensor<T> x, Tensor<T> groupMask, int n0)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var rankSum = WeightedRowSum(groupMask, ColumnAverageRanks(x));
        return engine.TensorAddScalar(rankSum, ops.FromDouble(-(double)n0 * (n0 + 1) / 2.0));
    }

    /// <summary>
    /// The two-sided Mann-Whitney U test of every column (scipy.stats.mannwhitneyu, method="asymptotic",
    /// use_continuity=True): the smaller of U0 and U1 = n0 n1 - U0, and the p-value
    /// <c>2 P(Z &gt; (max(U0, U1) - n0 n1 / 2 - 1/2) / s)</c> with the tie-corrected
    /// <c>s^2 = n0 n1 / 12 ((n + 1) - sum (t^3 - t) / (n (n - 1)))</c>, at most 1. Without spread (every value tied)
    /// the p-value is 1.
    /// </summary>
    public static (Tensor<T> U, Tensor<T> PValues) MannWhitney<T>(Tensor<T> x, Tensor<T> groupMask, int n0, int n1)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = x.Shape[1];
        double n = n0 + n1;
        double product = (double)n0 * n1;
        var u0 = RankSumU(x, groupMask, n0);
        var u1 = engine.TensorAddScalar(engine.TensorNegate(u0), ops.FromDouble(product));
        var uMin = engine.TensorMin(u0, u1);
        var uMax = engine.TensorMax(u0, u1);

        var variance = engine.TensorMultiplyScalar(
            engine.TensorAddScalar(
                engine.TensorDivideScalar(engine.TensorNegate(ColumnTieCubeSums(x)), ops.FromDouble(n * (n - 1))),
                ops.FromDouble(n + 1)),
            ops.FromDouble(product / 12.0));
        var z = FilterStatisticTensorOps.GuardedDivide(
            engine.TensorAddScalar(uMax, ops.FromDouble(-product / 2.0 - 0.5)), engine.TensorSqrt(variance), 0.0, ops.Zero);
        var spread = engine.TensorGreaterThan(variance, ops.Zero);
        var pValues = engine.TensorMin(
            engine.TensorMultiplyScalar(FilterStatisticTensorOps.NormalSurvival(z), ops.FromDouble(2.0)), ops.One);
        pValues = engine.TensorWhere(spread, pValues, FilterStatisticTensorOps.Filled(new[] { p }, ops.One));
        return (uMin, pValues);
    }

    /// <summary>
    /// The Wilcoxon signed-rank test of every column of <paramref name="x"/> ([n, p]) against the paired values
    /// <paramref name="y"/> ([n]) (scipy.stats.wilcoxon, zero_method="wilcox", method="approx", no continuity
    /// correction): differences within <paramref name="zeroTolerance"/> of 0 are dropped, the rest ranked by
    /// magnitude with ties averaged; <c>W = min(W+, W-)</c>, <c>z = (W - m (m + 1) / 4) / s</c> with
    /// <c>s^2 = m (m + 1)(2m + 1) / 24 - sum (t^3 - t) / 48</c> over the m non-zero differences, and the two-sided
    /// normal p-value. Returns W, the p-values and m, all [p]; a column with no spread has p = 1.
    /// </summary>
    public static (Tensor<T> W, Tensor<T> PValues, Tensor<T> NonZero) WilcoxonSignedRank<T>(
        Tensor<T> x, Tensor<T> y, double zeroTolerance)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        if (n == 0 || p == 0)
            return (new Tensor<T>(new[] { p }), FilterStatisticTensorOps.Filled(new[] { p }, ops.One), new Tensor<T>(new[] { p }));

        var difference = engine.TensorSubtract(x, FilterStatisticTensorOps.ColumnBroadcast(y, p));
        var nonZero = engine.TensorGreaterThan(engine.TensorAbs(difference), ops.FromDouble(zeroTolerance));
        // the dropped differences become distinct negative placeholders -(i + 1): they still take the lowest ranks
        // 1..zeros, but form no tie group, so the tie term needs no (zeros^3 - zeros) taken back out (that
        // subtraction of two huge, rounded numbers loses the real tie term in float once ~250 differences are zero)
        var placeholder = FilterStatisticTensorOps.ColumnBroadcast(engine.TensorArange<T>(-1, -n - 1, -1), p);
        var magnitude = engine.TensorWhere(nonZero, engine.TensorAbs(difference), placeholder);
        var zeros = Counts(PreprocessingTensorOps.OneMinus(nonZero), 0);
        var m = engine.TensorAddScalar(engine.TensorNegate(zeros), ops.FromDouble(n));

        // the dropped zeros take the lowest ranks 1..zeros, so a non-zero difference ranks (overall rank - zeros)
        var ranks = engine.TensorSubtract(ColumnAverageRanks(magnitude), FilterStatisticTensorOps.RowBroadcast(zeros, n));
        var positive = engine.TensorGreaterThan(difference, ops.FromDouble(zeroTolerance));
        var negative = engine.TensorLessThan(difference, ops.FromDouble(-zeroTolerance));
        var wPlus = engine.ReduceSum(engine.TensorMultiply(ranks, positive), new[] { 0 }, false);
        var wMinus = engine.ReduceSum(engine.TensorMultiply(ranks, negative), new[] { 0 }, false);
        var w = engine.TensorMin(wPlus, wMinus);

        // tie term over the non-zero magnitudes (the placeholders are all distinct and add nothing)
        var ties = ColumnTieCubeSums(magnitude);
        var mean = engine.TensorMultiplyScalar(engine.TensorMultiply(m, engine.TensorAddScalar(m, ops.One)), ops.FromDouble(0.25));
        var variance = engine.TensorSubtract(
            engine.TensorDivideScalar(
                engine.TensorMultiply(engine.TensorMultiply(m, engine.TensorAddScalar(m, ops.One)),
                    engine.TensorAddScalar(engine.TensorMultiplyScalar(m, ops.FromDouble(2.0)), ops.One)),
                ops.FromDouble(24.0)),
            engine.TensorDivideScalar(ties, ops.FromDouble(48.0)));
        var z = FilterStatisticTensorOps.GuardedDivide(
            engine.TensorSubtract(w, mean), engine.TensorSqrt(engine.TensorMax(variance, ops.Zero)), 0.0, ops.Zero);
        var pValues = engine.TensorWhere(
            engine.TensorGreaterThan(variance, ops.Zero),
            FilterStatisticTensorOps.NormalTwoSided(z),
            FilterStatisticTensorOps.Filled(new[] { p }, ops.One));
        return (w, pValues, m);
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

    /// <summary>
    /// Pearson chi-square statistic <c>sum (O - E)^2 / E</c> of [p, A, B] contingency tables, with
    /// <c>E = row total * column total / n</c>; cells with no expected count add nothing. Returns [p].
    /// </summary>
    public static Tensor<T> ContingencyChiSquare<T>(Tensor<T> observed)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = observed.Shape[0];
        int a = observed.Shape[1];
        int b = observed.Shape[2];
        var shape = new[] { p, a, b };
        var rows = engine.TensorBroadcastTo(engine.ReduceSum(observed, new[] { 2 }, true), shape);
        var columns = engine.TensorBroadcastTo(engine.ReduceSum(observed, new[] { 1 }, true), shape);
        var total = engine.TensorBroadcastTo(engine.ReduceSum(observed, new[] { 1, 2 }, true), shape);
        // row * (column / n): the product row * column overflows Half past 65504
        var expected = engine.TensorMultiply(rows, FilterStatisticTensorOps.GuardedDivide(columns, total, 0, ops.Zero));
        var cells = FilterStatisticTensorOps.GuardedDivide(
            engine.TensorSquare(engine.TensorSubtract(observed, expected)), expected, 0, ops.Zero);
        return engine.ReduceSum(cells, new[] { 1, 2 }, false);
    }

    /// <summary>
    /// Codes of the distinct values of every column of an [n, p] tensor (the position of each value among its
    /// column's sorted distinct values), and the largest number of distinct values in any column.
    /// </summary>
    public static Tensor<int> DistinctValueCodes<T>(Tensor<T> x, out int levels)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int p = x.Shape[1];
        levels = 1;
        if (p == 0)
            return new Tensor<int>(new[] { n, 0 });

        var columns = new Tensor<int>[p];
        for (int j = 0; j < p; j++)
        {
            var (values, inverse, _) = engine.TensorUniqueWithInfo(PreprocessingTensorOps.ColumnVector(x, j), true, true, false);
            if (inverse is null)
                throw new InvalidOperationException("The engine returned no inverse map for the distinct values.");
            levels = Math.Max(levels, values.Length);
            columns[j] = engine.Reshape(inverse, new[] { n, 1 });
        }

        return engine.TensorConcatenate(columns, 1);
    }

    /// <summary>
    /// Two-sided Fisher exact test of 2x2 tables <c>[[a, b], [c, d]]</c> given per feature ([p] each), as
    /// <c>scipy.stats.fisher_exact</c>: the p-value sums the hypergeometric probabilities of every table with the
    /// same margins that is no more likely than the observed one (relative tolerance 1e-7), and the odds ratio is
    /// <c>ad / bc</c> (infinite when <c>bc = 0</c>). A table with an empty row or column has odds ratio NaN and
    /// p-value 1.
    /// </summary>
    public static (Tensor<T> PValues, Tensor<T> OddsRatios) FisherExact<T>(
        Tensor<T> a, Tensor<T> b, Tensor<T> c, Tensor<T> d, int n)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = a.Length;
        var grid = new[] { p, n + 1 };
        T total = ops.FromDouble(n);
        var row1 = engine.TensorAdd(a, b);
        var column1 = engine.TensorAdd(a, c);
        var column2 = engine.TensorAdd(b, d);
        var k = FilterStatisticTensorOps.RowBroadcast(engine.TensorArange<T>(0, n + 1, 1), p);
        var r1 = engine.TensorBroadcastTo(engine.Reshape(row1, new[] { p, 1 }), grid);
        var c1 = engine.TensorBroadcastTo(engine.Reshape(column1, new[] { p, 1 }), grid);
        var c2 = engine.TensorBroadcastTo(engine.Reshape(column2, new[] { p, 1 }), grid);

        // P(X = k) = C(c1, k) C(c2, r1 - k) / C(n, r1) on the support max(0, r1 - c2) <= k <= min(r1, c1)
        var rest = engine.TensorSubtract(r1, k);
        var outside = engine.TensorAdd(
            engine.TensorAdd(engine.TensorGreaterThan(k, c1), engine.TensorGreaterThan(k, r1)),
            engine.TensorGreaterThan(rest, c2));
        var inSupport = engine.TensorEquals(outside, ops.Zero);
        var logPmf = engine.TensorSubtract(
            engine.TensorAdd(
                LogChoose(c1, engine.TensorMin(k, c1)),
                LogChoose(c2, engine.TensorMin(engine.TensorMax(rest, ops.Zero), c2))),
            LogChoose(FilterStatisticTensorOps.FilledLike(r1, total), r1));
        var pmf = engine.TensorMultiply(engine.TensorExp(logPmf), inSupport);

        var observedLog = engine.TensorSubtract(
            engine.TensorAdd(LogChoose(column1, a), LogChoose(column2, b)),
            LogChoose(FilterStatisticTensorOps.FilledLike(row1, total), row1));
        var threshold = FilterStatisticTensorOps.ColumnBroadcast(
            engine.TensorMultiplyScalar(engine.TensorExp(observedLog), ops.FromDouble(1 + 1e-7)), n + 1);
        var asExtreme = PreprocessingTensorOps.OneMinus(engine.TensorGreaterThan(pmf, threshold));
        var pValues = engine.TensorMin(
            engine.ReduceSum(engine.TensorMultiply(pmf, asExtreme), new[] { 1 }, false), ops.One);

        var odds = FilterStatisticTensorOps.GuardedDivide(
            engine.TensorMultiply(a, d), engine.TensorMultiply(b, c), 0, ops.FromDouble(double.PositiveInfinity));
        var row2 = engine.TensorAdd(c, d);
        var emptyMargins = engine.TensorAdd(
            engine.TensorAdd(engine.TensorEquals(row1, ops.Zero), engine.TensorEquals(row2, ops.Zero)),
            engine.TensorAdd(engine.TensorEquals(column1, ops.Zero), engine.TensorEquals(column2, ops.Zero)));
        var degenerate = engine.TensorGreaterThan(emptyMargins, ops.Zero);
        return (
            engine.TensorWhere(degenerate, FilterStatisticTensorOps.FilledLike(pValues, ops.One), pValues),
            engine.TensorWhere(degenerate, FilterStatisticTensorOps.FilledLike(odds, ops.FromDouble(double.NaN)), odds));
    }

    // log C(m, k) = lgamma(m + 1) - lgamma(k + 1) - lgamma(m - k + 1) for 0 <= k <= m
    private static Tensor<T> LogChoose<T>(Tensor<T> m, Tensor<T> k)
    {
        var engine = AiDotNetEngine.Current;
        T one = MathHelper.GetNumericOperations<T>().One;
        return engine.TensorSubtract(
            engine.TensorLgamma(engine.TensorAddScalar(m, one)),
            engine.TensorAdd(
                engine.TensorLgamma(engine.TensorAddScalar(k, one)),
                engine.TensorLgamma(engine.TensorAddScalar(engine.TensorSubtract(m, k), one))));
    }

    /// <summary>
    /// Best Gini impurity reduction of a single threshold split per column of an [n, p] tensor: the thresholds are
    /// <c>min + range * b / bins</c> for <c>b = 1 .. bins - 1</c> (range 1 for a column whose range is below
    /// 1e-10), a row goes left when its value is at or below the threshold, splits leaving a side empty are skipped,
    /// and a column with no useful split scores 0. <paramref name="classOneHot"/> is the [n, k] class membership.
    /// Returns [p].
    /// </summary>
    public static Tensor<T> BestGiniReduction<T>(Tensor<T> x, Tensor<T> classOneHot, int bins)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        T total = ops.FromDouble(n);
        T binCount = ops.FromDouble(bins);
        T zero = ops.Zero;
        var best = new Tensor<T>(new[] { p });
        if (n == 0)
            return best;

        // parent Gini impurity 1 - sum q_c^2
        var classCounts = Counts(classOneHot, 0);
        T parentGini = ops.Subtract(ops.One, engine.TensorSumOfSquares(engine.TensorDivideScalar(classCounts, total)));
        var allCounts = FilterStatisticTensorOps.RowBroadcast(classCounts, p);

        var min = PreprocessingTensorOps.ColumnMin(x);
        var range = engine.TensorSubtract(PreprocessingTensorOps.ColumnMax(x), min);
        range = engine.TensorWhere(engine.TensorLessThan(range, ops.FromDouble(1e-10)),
            FilterStatisticTensorOps.FilledLike(range, ops.One), range);
        foreach (T step in engine.TensorArange<T>(1, bins, 1).ToArray())
        {
            var threshold = engine.TensorAdd(min, engine.TensorDivideScalar(engine.TensorMultiplyScalar(range, step), binCount));
            var left = PreprocessingTensorOps.OneMinus(
                engine.TensorGreaterThan(x, FilterStatisticTensorOps.RowBroadcast(threshold, n)));
            var leftCounts = CountProduct(left, classOneHot);
            var rightCounts = engine.TensorSubtract(allCounts, leftCounts);
            var leftTotal = engine.ReduceSum(leftCounts, new[] { 1 }, false);
            var rightTotal = engine.ReduceSum(rightCounts, new[] { 1 }, false);

            // weighted child impurity: sum over sides of (side total - sum count^2 / side total) / n
            var weighted = engine.TensorDivideScalar(engine.TensorAdd(
                SideImpurity(leftCounts, leftTotal), SideImpurity(rightCounts, rightTotal)), total);
            var reduction = engine.TensorNegate(engine.TensorSubtractScalar(weighted, parentGini));
            var bothSides = engine.TensorMultiply(
                engine.TensorGreaterThan(leftTotal, zero), engine.TensorGreaterThan(rightTotal, zero));
            best = engine.TensorMax(best, KeepOrFill(reduction, bothSides, zero));
        }

        return best;
    }

    // side total * Gini impurity = total - sum count^2 / total for every row of [p, k] class counts
    private static Tensor<T> SideImpurity<T>(Tensor<T> counts, Tensor<T> totals)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorSubtract(totals, FilterStatisticTensorOps.GuardedDivide(
            engine.ReduceSum(engine.TensorSquare(counts), new[] { 1 }, false), totals, 0,
            MathHelper.GetNumericOperations<T>().Zero));
    }

    /// <summary>
    /// |Pearson r| of every column of an [n, p] tensor with a target ([p]); 0 for a column whose centred sum of
    /// squares is not above 1e-10, and everywhere when the target's is not.
    /// </summary>
    public static Tensor<T> AbsoluteCorrelation<T>(Tensor<T> x, Tensor<T> y)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var (sxx, sxy, syy) = FilterStatisticTensorOps.CenteredCrossSums(x, y);
        T tolerance = ops.FromDouble(1e-10);
        if (!ops.GreaterThan(syy, tolerance))
            return FilterStatisticTensorOps.FilledLike(sxx, ops.Zero);
        var correlation = engine.TensorAbs(FilterStatisticTensorOps.GuardedDivide(
            sxy, engine.TensorSqrt(engine.TensorMultiplyScalar(sxx, syy)), 0, ops.Zero));
        return KeepOrFill(correlation, engine.TensorGreaterThan(sxx, tolerance), ops.Zero);
    }

    /// <summary>
    /// Mean of <paramref name="values"/> ([p]) over the features each row of a 0/1 [m, p] mask keeps ([m]); 0 for a
    /// row that keeps none.
    /// </summary>
    public static Tensor<T> MeanOverMask<T>(Tensor<T> masks, Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = masks.Shape[0];
        int p = masks.Shape[1];
        var sums = engine.Reshape(engine.TensorMatMul(masks, engine.Reshape(values, new[] { p, 1 })), new[] { m });
        var counts = Counts(masks, 1);
        return FilterStatisticTensorOps.GuardedDivide(sums, counts, 0.5, ops.Zero);
    }

    /// <summary>
    /// 0/1 mask ([p]) of the <paramref name="k"/> largest entries of a [p] tensor, ties to the lower index (as
    /// <c>OrderByDescending(...).Take(k)</c>).
    /// </summary>
    public static Tensor<T> TopKMask<T>(Tensor<T> values, int k)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int p = values.Length;
        var top = PreprocessingTensorOps.StableOrder(values, descending: true).Take(k).ToArray();
        return ScatterToFeatures(FilterStatisticTensorOps.Filled(new[] { top.Length }, ops.One), top, p, ops.Zero);
    }

    /// <summary>
    /// For every entry of a 1-D tensor, how many entries are strictly smaller. Comparing these counts orders the
    /// entries exactly as comparing the values does (<c>a &gt; b</c> exactly when <c>less[a] &gt; less[b]</c>).
    /// </summary>
    public static int[] LessCounts<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        if (values.Length == 0)
            return Array.Empty<int>();
        var sorted = engine.TensorSort(values, 0, false).Item1;
        return engine.TensorSearchSorted(sorted, values, false).ToArray();
    }
}
