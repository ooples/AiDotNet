using AiDotNet.Helpers;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Row-wise arg-max with first-index tie breaking, as whole-tensor engine ops: the result a left-to-right scan
/// that only moves on a strictly greater value would pick.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal static class FirstMax<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// A [rows, cols] one-hot that marks, in each row of <paramref name="scores"/>, the first column holding the
    /// row maximum. A row without a comparable maximum (all NaN) is all zero.
    /// </summary>
    internal static Tensor<T> OneHot(Tensor<T> scores)
    {
        var engine = AiDotNetEngine.Current;
        var shape = new[] { scores.Shape[0], scores.Shape[1] };
        var isMax = engine.TensorEquals(scores, engine.TensorBroadcastTo(engine.ReduceMax(scores, new[] { 1 }, true), shape));
        var firstOfThem = engine.TensorEquals(engine.TensorCumSum(isMax, 1), Ops.One);
        return engine.TensorMultiply(isMax, firstOfThem);
    }

    /// <summary>
    /// Per row of <paramref name="scores"/>, the entry of <paramref name="values"/> (one per column) at the row's
    /// first maximum; a row without one falls back to <c>values[0]</c>, where such a scan starts.
    /// </summary>
    internal static Tensor<T> Select(Tensor<T> scores, Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        int rows = scores.Shape[0];
        int cols = scores.Shape[1];
        var oneHot = OneHot(scores);
        var picked = engine.TensorMatMul(oneHot, values.Reshape(cols, 1)).Reshape(rows);
        var unmatched = engine.ScalarMinusTensor(Ops.One, engine.ReduceSum(oneHot, new[] { 1 }, false));
        return engine.TensorAdd(picked, engine.TensorMultiplyScalar(unmatched, values[0]));
    }

    /// <summary>The first maximum's column index per row (0 for a row without one), as T.</summary>
    internal static Tensor<T> Index(Tensor<T> scores)
        => Select(scores, AiDotNetEngine.Current.TensorArange<T>(0, scores.Shape[1], 1));
}
