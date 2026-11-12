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
        var rounded = engine.TensorBroadcastTo(engine.Reshape(engine.TensorRound(labels), new[] { n, 1 }), shape);
        var classIndex = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorArange<T>(0, numClasses, 1), new[] { 1, numClasses }), shape);
        return engine.TensorEquals(rounded, classIndex);
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
    internal static Tensor<T> MidRanks(Tensor<T> values)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var sorted = engine.TensorSort(values).Values;
        // A value's ties occupy sorted positions [below, atOrBelow); their mean 1-based rank is (below + atOrBelow + 1) / 2.
        var below = engine.TensorSearchSorted(sorted, values, false);
        var atOrBelow = engine.TensorSearchSorted(sorted, values, true);
        var doubled = engine.TensorAddScalar(engine.TensorAdd(below.Cast<T>(), atOrBelow.Cast<T>()), NumOps.One);
        return engine.TensorMultiplyScalar(doubled, NumOps.FromDouble(0.5));
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
    internal static (Tensor<T> Sorted, Tensor<int> Order, Tensor<T> Ranks) SortedRunRanks(Tensor<T> values, T tolerance)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var (sorted, order) = engine.TensorSort(values);
        var runStart = new Tensor<T>(new[] { 1 });
        if (values.Length > 1)
        {
            // A gap that is not below the tolerance (NaN included) starts a new run.
            var gaps = engine.TensorAbs(engine.TensorDiff(sorted, 1, 0));
            runStart = engine.TensorConcatenate(new[] { runStart, Not(engine.TensorLessThan(gaps, tolerance)) }, 0);
        }

        // Run ids are non-decreasing, so a run occupies sorted positions [below, atOrBelow) of its id.
        var runId = engine.TensorCumSum(runStart, 0);
        var below = engine.TensorSearchSorted(runId, runId, false);
        var atOrBelow = engine.TensorSearchSorted(runId, runId, true);
        var doubled = engine.TensorAddScalar(engine.TensorAdd(below.Cast<T>(), atOrBelow.Cast<T>()), NumOps.One);
        return (sorted, order, engine.TensorMultiplyScalar(doubled, NumOps.FromDouble(0.5)));
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
    internal static Tensor<T> RowRanksDescending(Tensor<T> matrix, T tolerance)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        int rows = matrix.Shape[0];
        int columns = matrix.Shape[1];
        var (sorted, order) = engine.TensorSort(matrix, 1, true);
        var runStart = new Tensor<T>(new[] { rows, 1 });
        if (columns > 1)
        {
            // A gap that is not below the tolerance (NaN included) starts a new run.
            var gaps = engine.TensorAbs(engine.TensorDiff(sorted, 1, 1));
            runStart = engine.TensorConcatenate(new[] { runStart, Not(engine.TensorLessThan(gaps, tolerance)) }, 1);
        }

        // Per row, run ids are non-decreasing, so a run occupies sorted positions [below, atOrBelow) of its id.
        var runId = engine.TensorCumSum(runStart, 1);
        var below = engine.TensorSearchSorted(runId, runId, false);
        var atOrBelow = engine.TensorSearchSorted(runId, runId, true);
        var doubled = engine.TensorAddScalar(engine.TensorAdd(below.Cast<T>(), atOrBelow.Cast<T>()), NumOps.One);
        var sortedRanks = engine.TensorMultiplyScalar(doubled, NumOps.FromDouble(0.5));

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
        var pairCredit = engine.TensorSum(engine.TensorMultiply(positiveMask, engine.TensorAdd(negativesBelow, negativesAtOrBelow)));
        return NumOps.Divide(pairCredit, NumOps.FromDouble(2.0 * positives * negatives));
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
        var truePositives = engine.TensorSubtract(engine.TensorFullLike(scores, NumOps.FromDouble(positives)), positivesBelow);
        var predictedPositives = engine.TensorSubtract(engine.TensorFullLike(scores, NumOps.FromDouble(scores.Length)), samplesBelow);
        var precision = engine.TensorDivide(truePositives, predictedPositives);

        // Each positive adds 1 / positives of recall at its threshold.
        var gained = engine.TensorSum(engine.TensorMultiply(positiveMask, precision));
        return NumOps.Divide(gained, NumOps.FromDouble(positives));
    }

    /// <summary>
    /// For every sample, the total <paramref name="weights"/> of the samples scoring strictly below it and of
    /// those scoring at or below it.
    /// </summary>
    private static (Tensor<T> Below, Tensor<T> AtOrBelow) WeightBelow(Tensor<T> scores, Tensor<T> weights)
    {
        var engine = AiDotNetEngine.Current;
        var (sorted, order) = engine.TensorSort(scores);
        var cumulative = engine.TensorCumSum(engine.TensorGather(weights, order, 0), 0);
        var padded = engine.TensorConcatenate(new[] { new Tensor<T>(new[] { 1 }), cumulative }, 0);
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
}
