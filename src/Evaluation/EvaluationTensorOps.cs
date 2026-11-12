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
    /// Returns two order statistics of bootstrap values, ordered as <c>Array.Sort</c> orders doubles (NaN first,
    /// then ascending).
    /// </summary>
    internal static (T Lower, T Upper) OrderStatistics(List<T> values, int lowerIndex, int upperIndex)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var tensor = new Tensor<T>(values.ToArray(), new[] { values.Count });
        var sorted = engine.TensorSort(tensor).Values;
        int nanCount = Count(NanMask(tensor));
        return (SortedAt(sorted, nanCount, lowerIndex), SortedAt(sorted, nanCount, upperIndex));
    }

    // The engine sort puts NaN last; Array.Sort puts it first, so positions shift by the NaN count.
    private static T SortedAt(Tensor<T> sortedNaNLast, int nanCount, int index)
        => index < nanCount ? sortedNaNLast[sortedNaNLast.Length - 1] : sortedNaNLast[index - nanCount];
}
