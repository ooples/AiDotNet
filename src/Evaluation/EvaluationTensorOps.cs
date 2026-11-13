using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation;

/// <summary>
/// Whole-array engine building blocks shared by the evaluation metrics and statistical tests.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Every helper works on whole tensors through <see cref="AiDotNetEngine.Current"/> and suppresses gradient
/// recording, because metrics are evaluated on plain data and must never add nodes to an active tape.
/// </para>
/// <para>
/// Label comparisons follow <c>INumericOperations.Compare(a, b) == 0</c>, the rule the metrics always used:
/// NaN compares equal to NaN and -0 equals +0.
/// </para>
/// </remarks>
internal static class EvaluationTensorOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Copies a span into a new rank-1 tensor.</summary>
    internal static Tensor<T> ToTensor(ReadOnlySpan<T> values)
        => new Tensor<T>(values.ToArray(), new[] { values.Length });

    /// <summary>A 1/0 mask of the NaN entries of <paramref name="values"/>.</summary>
    internal static Tensor<T> NanMask(Tensor<T> values)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        return engine.TensorWhere(engine.TensorIsNan(values), engine.TensorOnesLike(values), engine.TensorZerosLike(values));
    }

    /// <summary>A 1/0 mask of the entries that compare equal to <paramref name="label"/>.</summary>
    internal static Tensor<T> EqualMask(Tensor<T> values, T label)
    {
        if (NumOps.IsNaN(label))
        {
            return NanMask(values);
        }

        using var noGrad = new NoGradScope<T>();
        return AiDotNetEngine.Current.TensorEquals(values, label);
    }

    /// <summary>A 1/0 mask of the positions where <paramref name="a"/> and <paramref name="b"/> compare equal.</summary>
    internal static Tensor<T> PairEqualMask(Tensor<T> a, Tensor<T> b)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var equal = engine.TensorEquals(a, b);
        var bothNaN = engine.TensorMultiply(NanMask(a), NanMask(b));
        return engine.TensorAdd(equal, bothNaN);
    }

    /// <summary>A 1/0 mask of the entries that are greater than or equal to <paramref name="threshold"/>.</summary>
    internal static Tensor<T> GreaterOrEqualMask(Tensor<T> values, T threshold)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        return engine.TensorGreaterEqual(values, engine.TensorFullLike(values, threshold));
    }

    /// <summary>The complement (1 - mask) of a 1/0 mask.</summary>
    internal static Tensor<T> Not(Tensor<T> mask)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        return engine.TensorSubtract(engine.TensorOnesLike(mask), mask);
    }

    /// <summary>The element-wise product (logical and) of two 1/0 masks.</summary>
    internal static Tensor<T> And(Tensor<T> a, Tensor<T> b)
    {
        using var noGrad = new NoGradScope<T>();
        return AiDotNetEngine.Current.TensorMultiply(a, b);
    }

    /// <summary>
    /// Clamps to [<paramref name="min"/>, <paramref name="max"/>] as <c>Math.Max(min, Math.Min(max, x))</c> does:
    /// NaN stays NaN.
    /// </summary>
    internal static Tensor<T> ClampPreservingNaN(Tensor<T> values, T min, T max)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        return engine.TensorWhere(engine.TensorIsNan(values), values, engine.TensorClamp(values, min, max));
    }

    /// <summary>log(p / (1 - p)).</summary>
    internal static Tensor<T> Logit(Tensor<T> p)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        return engine.TensorLog(engine.TensorDivide(p, engine.TensorSubtract(engine.TensorOnesLike(p), p)));
    }

    /// <summary>Elementwise OR of two 1/0 masks.</summary>
    internal static Tensor<T> Or(Tensor<T> a, Tensor<T> b)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        return engine.TensorSubtract(engine.TensorAdd(a, b), engine.TensorMultiply(a, b));
    }

    /// <summary>
    /// The [numBins, n] 1/0 mask of reliability bins: value j is in bin i when i/numBins &lt;= p_j &lt; (i+1)/numBins,
    /// the last bin also taking p_j == 1; NaN is in no bin.
    /// </summary>
    internal static Tensor<T> UniformBinMask(Tensor<T> probabilities, int numBins)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        int n = probabilities.Length;
        var shape = new[] { numBins, n };
        var column = new[] { numBins, 1 };
        var binCount = NumOps.FromDouble(numBins);
        var binIndex = engine.TensorArange<T>(0, numBins);
        var p = engine.TensorBroadcastTo(engine.Reshape(probabilities, new[] { 1, n }), shape);
        var starts = engine.TensorBroadcastTo(engine.Reshape(engine.TensorDivideScalar(binIndex, binCount), column), shape);
        var ends = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorDivideScalar(engine.TensorAddScalar(binIndex, NumOps.One), binCount), column), shape);
        var isLast = engine.TensorBroadcastTo(engine.Reshape(EqualMask(binIndex, NumOps.FromDouble(numBins - 1)), column), shape);
        var valid = engine.TensorBroadcastTo(engine.Reshape(Not(NanMask(probabilities)), new[] { 1, n }), shape);

        var atLeastStart = Not(engine.TensorLessThan(p, starts));
        var belowEnd = engine.TensorLessThan(p, ends);
        var atMostEnd = Not(engine.TensorGreaterThan(p, ends));
        return And(And(valid, atLeastStart), Or(belowEnd, And(isLast, atMostEnd)));
    }

    /// <summary>The number of non-zero entries of a mask.</summary>
    internal static int Count(Tensor<T> mask)
    {
        using var noGrad = new NoGradScope<T>();
        return AiDotNetEngine.Current.TensorCountNonzero(mask);
    }

    /// <summary>
    /// Counts, for one label, the positions where both sides equal it, where the predictions equal it and where the
    /// actuals equal it.
    /// </summary>
    internal static (int TruePositives, int PredictedPositives, int ActualPositives) LabelCounts(
        Tensor<T> predictions, Tensor<T> actuals, T label)
    {
        var predicted = EqualMask(predictions, label);
        var actual = EqualMask(actuals, label);
        return (Count(And(predicted, actual)), Count(predicted), Count(actual));
    }

    /// <summary>
    /// The binary confusion counts when "positive" means comparing equal to <paramref name="positiveLabel"/>.
    /// </summary>
    internal static (int Tp, int Tn, int Fp, int Fn) BinaryConfusion(
        ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, T positiveLabel)
    {
        var counts = LabelCounts(ToTensor(predictions), ToTensor(actuals), positiveLabel);
        return FromCounts(predictions.Length, counts.TruePositives, counts.PredictedPositives, counts.ActualPositives);
    }

    /// <summary>
    /// The binary confusion counts when "positive" means being greater than or equal to
    /// <paramref name="threshold"/>.
    /// </summary>
    internal static (int Tp, int Tn, int Fp, int Fn) ThresholdConfusion(
        ReadOnlySpan<T> predictions, ReadOnlySpan<T> actuals, T threshold)
    {
        var predicted = GreaterOrEqualMask(ToTensor(predictions), threshold);
        var actual = GreaterOrEqualMask(ToTensor(actuals), threshold);
        return FromCounts(predictions.Length, Count(And(predicted, actual)), Count(predicted), Count(actual));
    }

    private static (int Tp, int Tn, int Fp, int Fn) FromCounts(int n, int tp, int predictedPositives, int actualPositives)
    {
        int fp = predictedPositives - tp;
        int fn = actualPositives - tp;
        return (tp, n - tp - fp - fn, fp, fn);
    }

    /// <summary>
    /// An [n, numClasses] 1/0 mask that marks, in row i, the class equal to labels[i] rounded half to even (as
    /// <c>(int)Math.Round(label)</c>); a row whose rounded label is not a class index stays all zero.
    /// </summary>
    internal static Tensor<T> RoundedOneHot(Tensor<T> labels, int numClasses)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        int n = labels.Length;
        var shape = new[] { n, numClasses };
        var rounded = engine.TensorBroadcastTo(engine.Reshape(engine.TensorRound(labels.Cast<double>()), new[] { n, 1 }), shape);
        var classIndex = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorArange<double>(0, numClasses, 1), new[] { 1, numClasses }), shape);
        return engine.TensorEquals(rounded, classIndex).Cast<T>();
    }

    /// <summary>
    /// Element-wise <c>Math.Max(value, floor)</c>, which returns NaN for a NaN value (the engine max returns the
    /// floor).
    /// </summary>
    internal static Tensor<T> MaxPreservingNaN(Tensor<T> values, T floor)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        return engine.TensorWhere(engine.TensorIsNan(values), values, engine.TensorMax(values, floor));
    }

    /// <summary>The sum and the number of the finite entries (neither NaN nor infinite).</summary>
    internal static (T Sum, int Count) FiniteSumAndCount(Tensor<T> values)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var finite = engine.TensorIsFinite(values);
        var finiteMask = engine.TensorWhere(finite, engine.TensorOnesLike(values), engine.TensorZerosLike(values));
        var kept = engine.TensorWhere(finite, values, engine.TensorZerosLike(values));
        return (engine.TensorSum(kept), Count(finiteMask));
    }

    /// <summary>
    /// 1-based ranks with tied values sharing the mean of their positions (scipy's <c>rankdata</c>, method
    /// "average").
    /// </summary>
    internal static Tensor<double> MidRanks(Tensor<T> values) => MidRanksWithTies(values).Ranks;

    internal static (Tensor<double> Ranks, Tensor<int> TieSizes) MidRanksWithTies(Tensor<T> values)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var sorted = engine.TensorSort(values).Values;
        var below = engine.TensorSearchSorted(sorted, values, false);
        var atOrBelow = engine.TensorSearchSorted(sorted, values, true);
        var doubled = engine.TensorAddScalar(engine.TensorAdd(below.Cast<double>(), atOrBelow.Cast<double>()), 1.0);
        return (engine.TensorMultiplyScalar(doubled, 0.5), engine.TensorSubtract(atOrBelow, below));
    }


    /// <summary>The entries of <paramref name="values"/> where the 1/0 <paramref name="mask"/> is 1, in order.</summary>
    internal static Tensor<T> Select(Tensor<T> values, Tensor<T> mask)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        return engine.TensorMaskedSelect(values, engine.TensorEqScalar(mask, NumOps.One));
    }

    /// <summary>
    /// Sorts <paramref name="values"/> ascending and gives every sorted position the mean 1-based rank of its tie
    /// run, a run being consecutive sorted values whose neighbouring gaps are below <paramref name="tolerance"/>.
    /// </summary>
    /// <returns>The sorted values, the source index of each sorted position, and the rank of each sorted position.</returns>
    internal static (Tensor<T> Sorted, Tensor<int> Order, Tensor<double> Ranks) SortedRunRanks(Tensor<T> values, double tolerance)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var (sorted, order) = engine.TensorSort(values);
        var runStart = RankRunStarts(sorted, 0, tolerance);

        // Run ids are non-decreasing, so a run occupies sorted positions [below, atOrBelow) of its id.
        var runId = engine.TensorCumSum(runStart, 0);
        var below = engine.TensorSearchSorted(runId, runId, false);
        var atOrBelow = engine.TensorSearchSorted(runId, runId, true);
        var doubled = engine.TensorAddScalar(engine.TensorAdd(below.Cast<double>(), atOrBelow.Cast<double>()), 1.0);
        return (sorted, order, engine.TensorMultiplyScalar(doubled, 0.5));
    }

    private static Tensor<double> RankRunStarts(Tensor<T> sorted, int axis, double tolerance)
    {
        int columns = sorted.Shape[axis];
        int rows = axis == 0 ? 1 : sorted.Shape[0];
        var type = Type.GetTypeCode(typeof(T));
        if (type is TypeCode.Decimal or TypeCode.Byte or TypeCode.SByte or TypeCode.Int16 or
            TypeCode.UInt16 or TypeCode.Int32 or TypeCode.UInt32 or TypeCode.Int64 or TypeCode.UInt64)
        {
            // Decimal arithmetic retains adjacent large labels; it also holds every Int64/UInt64 exactly.
            // A decimal gap that overflows is necessarily larger than the ranking tolerance.
            var values = sorted.AsSpan();
            var starts = new double[sorted.Length];
            decimal threshold = (decimal)tolerance;
            for (int row = 0; row < rows; row++)
            {
                for (int column = 0; column < columns; column++)
                {
                    int index = row * columns + column;
                    starts[index] = 1;
                    if (column == 0) continue;
                    try
                    {
                        decimal gap = Math.Abs(Convert.ToDecimal(values[index]) - Convert.ToDecimal(values[index - 1]));
                        if (gap < threshold) starts[index] = 0;
                    }
                    catch (OverflowException) { /* Distinct extreme values start a new run. */ }
                }
            }
            return new Tensor<double>(starts, axis == 0 ? new[] { columns } : new[] { rows, columns });
        }

        var engine = AiDotNetEngine.Current;
        var first = Tensor<double>.CreateDefault(axis == 0 ? new[] { 1 } : new[] { rows, 1 }, 1.0);
        if (columns <= 1) return first;
        // Casting floating-point values to double is exact; NaN gaps continue to start new runs.
        var gaps = engine.TensorAbs(engine.TensorDiff(sorted.Cast<double>(), 1, axis));
        var startsAfterFirst = EvaluationTensorOps<double>.Not(engine.TensorLessThan(gaps, tolerance));
        return engine.TensorConcatenate(new[] { first, startsAfterFirst }, axis);
    }

    /// <summary>Stacks equal-length rows into a [rows.Length, length] tensor.</summary>
    internal static Tensor<T> StackRows(T[][] rows, int length)
    {
        using var noGrad = new NoGradScope<T>();
        var tensors = new Tensor<T>[rows.Length];
        for (int r = 0; r < rows.Length; r++)
        {
            tensors[r] = new Tensor<T>(rows[r], new[] { length });
        }

        return AiDotNetEngine.Current.TensorStack(tensors, 0);
    }

    /// <summary>
    /// Ranks every row of an [n, k] matrix in descending order (rank 1 is the largest), a tie run of consecutive
    /// sorted values whose neighbouring gaps are below <paramref name="tolerance"/> sharing its mean rank.
    /// </summary>
    /// <returns>The [n, k] ranks in the matrix's own positions.</returns>
    internal static Tensor<double> RowRanksDescending(Tensor<T> matrix, double tolerance)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var (sorted, order) = engine.TensorSort(matrix, 1, true);
        var runStart = RankRunStarts(sorted, 1, tolerance);

        // Per row, run ids are non-decreasing, so a run occupies sorted positions [below, atOrBelow) of its id.
        var runId = engine.TensorCumSum(runStart, 1);
        var below = engine.TensorSearchSorted(runId, runId, false);
        var atOrBelow = engine.TensorSearchSorted(runId, runId, true);
        var doubled = engine.TensorAddScalar(engine.TensorAdd(below.Cast<double>(), atOrBelow.Cast<double>()), 1.0);
        var sortedRanks = engine.TensorMultiplyScalar(doubled, 0.5);

        // Back to the matrix positions: the inverse of each row's sort permutation.
        var inverse = engine.TensorArgsort(order, 1, false);
        return engine.TensorTakeAlongDim(sortedRanks, inverse, 1);
    }

    /// <summary>
    /// The distinct labels of the given tensors in ascending order, with a single NaN last when any is present
    /// (one class, as a <c>HashSet&lt;double&gt;</c> of the labels had).
    /// </summary>
    internal static Tensor<T> UniqueLabels(params Tensor<T>[] sources)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var all = sources.Length == 1 ? sources[0] : engine.TensorConcatenate(sources, 0);
        if (all.Length == 0)
        {
            return all;
        }

        var isNaN = engine.TensorIsNan(all);
        if (Count(NanMask(all)) == 0)
        {
            return engine.TensorUnique(all, true);
        }

        var nan = new Tensor<T>(new[] { NumOps.FromDouble(double.NaN) }, new[] { 1 });
        var labels = engine.TensorMaskedSelect(all, engine.TensorLogicalNot(isNaN));
        return labels.Length == 0 ? nan : engine.TensorConcatenate(new[] { engine.TensorUnique(labels, true), nan }, 0);
    }

    /// <summary>
    /// The area under the ROC curve of <paramref name="scores"/> for the 1/0 <paramref name="positiveMask"/>, as
    /// the Mann-Whitney statistic: the fraction of (positive, negative) pairs the scores order correctly, ties
    /// counting one half (scikit-learn's <c>roc_auc_score</c>). The caller guarantees both classes are present.
    /// </summary>
    internal static T RocAuc(Tensor<T> scores, Tensor<T> positiveMask)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        int positives = Count(positiveMask);
        int negatives = scores.Length - positives;
        var (negativesBelow, negativesAtOrBelow) = WeightBelow(scores, Not(positiveMask));
        var pairCredit = engine.TensorSum(engine.TensorMultiply(positiveMask.Cast<double>(), engine.TensorAdd(negativesBelow, negativesAtOrBelow)));
        return NumOps.FromDouble(pairCredit / (2.0 * positives * negatives));
    }

    /// <summary>
    /// Average precision of <paramref name="scores"/> for the 1/0 <paramref name="positiveMask"/>: the sum over
    /// distinct thresholds of precision times the recall gained there (scikit-learn's
    /// <c>average_precision_score</c>). Returns zero when there is no positive.
    /// </summary>
    internal static T AveragePrecision(Tensor<T> scores, Tensor<T> positiveMask)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        int positives = Count(positiveMask);
        if (positives == 0)
        {
            return NumOps.Zero;
        }

        // At the threshold of each sample's own score: predicted positives are the samples scoring at least as high.
        var (positivesBelow, _) = WeightBelow(scores, positiveMask);
        var (samplesBelow, _) = WeightBelow(scores, engine.TensorOnesLike(scores));
        var truePositives = engine.TensorSubtract(engine.TensorFullLike(positivesBelow, (double)positives), positivesBelow);
        var predictedPositives = engine.TensorSubtract(engine.TensorFullLike(samplesBelow, (double)scores.Length), samplesBelow);
        var precision = engine.TensorDivide(truePositives, predictedPositives);

        // Each positive adds 1 / positives of recall at its threshold.
        var gained = engine.TensorSum(engine.TensorMultiply(positiveMask.Cast<double>(), precision));
        return NumOps.FromDouble(gained / positives);
    }

    /// <summary>
    /// For every sample, the total <paramref name="weights"/> of the samples scoring strictly below it and of
    /// those scoring at or below it.
    /// </summary>
    private static (Tensor<double> Below, Tensor<double> AtOrBelow) WeightBelow(Tensor<T> scores, Tensor<T> weights)
    {
        var engine = AiDotNetEngine.Current;
        var (sorted, order) = engine.TensorSort(scores);
        var cumulative = engine.TensorCumSum(engine.TensorGather(weights.Cast<double>(), order, 0), 0);
        var padded = engine.TensorConcatenate(new[] { new Tensor<double>(new[] { 1 }), cumulative }, 0);
        var below = engine.TensorSearchSorted(sorted, scores, false);
        var atOrBelow = engine.TensorSearchSorted(sorted, scores, true);
        return (engine.TensorGather(padded, below, 0), engine.TensorGather(padded, atOrBelow, 0));
    }

    /// <summary>
    /// Returns two order statistics of bootstrap values, ordered as <c>Array.Sort</c> orders doubles (NaN first,
    /// then ascending).
    /// </summary>
    internal static (T Lower, T Upper) OrderStatistics(List<T> values, int lowerIndex, int upperIndex)
        => OrderStatistics(new Tensor<T>(values.ToArray(), new[] { values.Count }), lowerIndex, upperIndex);

    /// <summary>
    /// Returns two order statistics of a rank-1 tensor, ordered as <c>Array.Sort</c> orders doubles (NaN first,
    /// then ascending).
    /// </summary>
    internal static (T Lower, T Upper) OrderStatistics(Tensor<T> tensor, int lowerIndex, int upperIndex)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var sorted = engine.TensorSort(tensor).Values;
        int nanCount = Count(NanMask(tensor));
        return (SortedAt(sorted, nanCount, lowerIndex), SortedAt(sorted, nanCount, upperIndex));
    }

    // The engine sort puts NaN last; Array.Sort puts it first, so positions shift by the NaN count.
    private static T SortedAt(Tensor<T> sortedNaNLast, int nanCount, int index)
        => index < nanCount ? sortedNaNLast[sortedNaNLast.Length - 1] : sortedNaNLast[index - nanCount];

    /// <summary>
    /// Sorts a rank-1 tensor the way <c>Array.Sort</c> / <c>OrderBy</c> order doubles: NaN first, then ascending.
    /// </summary>
    internal static Tensor<T> SortNaNFirst(Tensor<T> values)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var sorted = engine.TensorSort(values).Values;
        int n = values.Length;
        int nanCount = Count(NanMask(values));
        if (nanCount == 0 || nanCount == n) return sorted;
        return engine.TensorConcatenate(
            new[] { engine.TensorNarrow(sorted, 0, n - nanCount, nanCount), engine.TensorNarrow(sorted, 0, 0, n - nanCount) }, 0);
    }

    /// <summary>
    /// The stable sort permutation for the <c>OrderBy</c> order of doubles: NaN first (in input order), then
    /// ascending with ties in input order.
    /// </summary>
    internal static Tensor<int> ArgsortNaNFirst(Tensor<T> values)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var order = engine.TensorSort(values).Indices;
        int n = values.Length;
        int nanCount = Count(NanMask(values));
        if (nanCount == 0 || nanCount == n) return order;
        return engine.TensorConcatenate(
            new[] { engine.TensorNarrow(order, 0, n - nanCount, nanCount), engine.TensorNarrow(order, 0, 0, n - nanCount) }, 0);
    }

    /// <summary>
    /// Concatenates groups into one rank-1 tensor and returns, for each pooled element, the index of its group.
    /// </summary>
    internal static (Tensor<double> Pooled, Tensor<int> GroupIndex, Tensor<double> Sizes) PoolGroups(T[][] groups)
    {
        using var noGrad = new NoGradScope<double>();
        var (pooled, groupIndex, sizes) = PoolNativeGroups(groups);
        return (pooled.Cast<double>(), groupIndex, sizes);
    }

    internal static (Tensor<T> Pooled, Tensor<int> GroupIndex, Tensor<double> Sizes) PoolNativeGroups(T[][] groups)
    {
        using var wideNoGrad = new NoGradScope<double>();

        var parts = new Tensor<T>[groups.Length];
        var sizes = new int[groups.Length];
        int total = 0;
        for (int g = 0; g < groups.Length; g++)
        {
            parts[g] = new Tensor<T>(groups[g], new[] { groups[g].Length });
            sizes[g] = groups[g].Length;
            total += sizes[g];
        }

        var groupIndex = new int[total];
        int position = 0;
        for (int g = 0; g < groups.Length; g++)
        {
            for (int i = 0; i < sizes[g]; i++)
                groupIndex[position++] = g;
        }

        using var noGrad = new NoGradScope<T>();
        return (AiDotNetEngine.Current.TensorConcatenate(parts, 0), new Tensor<int>(groupIndex, new[] { total }),
            new Tensor<int>(sizes, new[] { groups.Length }).Cast<double>());
    }

    /// <summary>
    /// Index of the first element equal to <paramref name="target"/> under <c>Compare</c> semantics (NaN equals
    /// NaN), or -1 when none matches.
    /// </summary>
    internal static int FirstIndexOf(Tensor<T> values, T target)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var matches = EqualMask(values, target);
        // Elements before the first match have a running match count of zero.
        int leading = Count(EqualMask(engine.TensorCumSum(matches, 0), MathHelper.GetNumericOperations<T>().Zero));
        return leading == values.Length ? -1 : leading;
    }

    /// <summary>
    /// The index a left-to-right scan keeps when it replaces the best value only on a strict
    /// <c>GreaterThan</c> (or <c>LessThan</c>) comparison: NaN never wins and never loses, so a NaN first element
    /// stays best.
    /// </summary>
    internal static int StrictScanBestIndex(Tensor<T> values, bool highest)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (numOps.IsNaN(values[0])) return 0;
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var finite = Select(values, Not(NanMask(values)));
        return FirstIndexOf(values, highest ? engine.TensorMaxValue(finite) : engine.TensorMinValue(finite));
    }

    /// <summary>
    /// The index a left-to-right scan keeps when it replaces the best value only when <c>Compare</c> (CompareTo
    /// order, NaN lowest) is strictly greater (or strictly less).
    /// </summary>
    internal static int CompareScanBestIndex(Tensor<T> values, bool highest)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var finite = Select(values, Not(NanMask(values)));
        if (finite.Length == 0) return 0;
        if (highest) return FirstIndexOf(values, engine.TensorMaxValue(finite));
        if (finite.Length < values.Length)
            return engine.TensorArgMax(NanMask(values), 0)[0];
        return FirstIndexOf(values, engine.TensorMinValue(finite));
    }

    /// <summary>
    /// Sums pooled values per group (torch <c>index_add</c> into a zero vector of length <paramref name="groupCount"/>).
    /// </summary>
    internal static Tensor<T> GroupSums(Tensor<T> pooled, Tensor<int> groupIndex, int groupCount)
    {
        using var noGrad = new NoGradScope<T>();
        return AiDotNetEngine.Current.TensorIndexAdd(new Tensor<T>(new[] { groupCount }), 0, groupIndex, pooled);
    }
}
