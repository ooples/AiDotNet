using AiDotNet.Tensors.Engines;

namespace AiDotNet.Helpers;

/// <summary>
/// Chooses the <c>k</c> largest entries of each row as one-hot rows, with whole-tensor engine ops so the choice stays
/// on the device of the scores (a compiled GPU plan replays it without reading the scores back to the host).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// The order is a stable descending sort: larger first, ties keep the earlier entry. The rank of entry <c>i</c> is
/// the number of entries that come before it in that order (a larger value, or an equal one at an earlier position),
/// so slot <c>u</c> of the result is the entry whose rank is <c>u</c>. The pairwise comparison costs
/// <c>rows x n x n</c>, intended for the short rows of a sequence model's query or lag selection. The result is a
/// selection, not differentiated: call it under <c>GradientTape&lt;T&gt;.NoGrad()</c>.
/// </remarks>
internal static class OrderedTopKSelection<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// One-hot rows <c>[rows, k, n]</c> of the <c>k</c> largest entries of each row of <paramref name="scores"/>
    /// (<c>[rows, n]</c>), in descending order with ties keeping the earlier entry.
    /// </summary>
    public static Tensor<T> Select(IEngine engine, Tensor<T> scores, int k)
    {
        if (scores.Rank != 2) throw new ArgumentException("Scores must be [rows, n].", nameof(scores));
        int rows = scores.Shape[0], n = scores.Shape[1];
        if (k < 1 || k > n) throw new ArgumentOutOfRangeException(nameof(k), $"k must be in [1, {n}].");

        // diff[r, i, j] = s_j − s_i; j comes before i when diff > 0, or diff == 0 and j < i.
        var others = engine.TensorTile(engine.Reshape(scores, new[] { rows, 1, n }), new[] { 1, n, 1 });
        var own = engine.TensorTile(engine.Reshape(scores, new[] { rows, n, 1 }), new[] { 1, 1, n });
        var diff = engine.TensorSubtract(others, own);
        var larger = engine.TensorGreaterThan(diff, NumOps.Zero);
        var smaller = engine.TensorLessThan(diff, NumOps.Zero);
        var equal = engine.TensorSubtract(engine.TensorSubtract(Filled(new[] { rows, n, n }, 1), larger), smaller);
        var earlier = engine.TensorTile(Constant(n, n, (i, j) => j < i ? 1 : 0), new[] { rows, 1, 1 });
        var before = engine.TensorAdd(larger, engine.TensorMultiply(equal, earlier));
        // Masks are exact 0/1 values in T, but low-precision counters are not (BF16 rounds 257 to 256).
        if (typeof(T) == typeof(Half) || typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16))
            return CastData<float, T>(engine, OrderedTopKSelection<float>.SelectByRank(engine,
                CastData<T, float>(engine, before), rows, n, k));
        return SelectByRank(engine, before, rows, n, k);
    }

    private static Tensor<T> SelectByRank(IEngine engine, Tensor<T> before, int rows, int n, int k)
    {
        var rank = engine.ReduceSum(before, new[] { 2 }, keepDims: false);                                    // [rows, n]

        // slot[r, u, i] = 1 when rank[r, i] == u (ranks are small whole numbers, so |rank − u| < ½ is equality).
        var ranks = engine.TensorTile(engine.Reshape(rank, new[] { rows, 1, n }), new[] { 1, k, 1 });
        var slots = engine.TensorTile(Constant(k, n, (u, _) => u), new[] { rows, 1, 1 });
        return engine.TensorLessThan(engine.TensorAbs(engine.TensorSubtract(ranks, slots)), NumOps.FromDouble(0.5));
    }

    // Selection is data, with no gradient across the dtype boundary. Record both casts so replay recomputes ranks.
    private static Tensor<TOut> CastData<TIn, TOut>(IEngine engine, Tensor<TIn> input)
    {
        if (typeof(TIn) == typeof(TOut)) return (Tensor<TOut>)(object)input;
        if (AiDotNet.Tensors.Engines.Compilation.GraphMode.IsActive
            && AiDotNet.Tensors.Engines.Compilation.GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(engine);
            return scope.RecordCrossType<TIn, TOut>(AiDotNet.Tensors.Engines.Compilation.LazyNodeType.Custom,
                "OrderedTopKSelection.Cast", input, input._shape, (e, output) =>
                {
                    var converted = CastData<TIn, TOut>(e, input);
                    DirectGpuTensorEngine.CopyResultInto(e, converted, output);
                });
        }
        return engine.TryGet<DirectGpuTensorEngine>(out var gpu) ? gpu.CastResidentDtype<TIn, TOut>(input) : input.Cast<TOut>();
    }

    /// <summary>A <c>[1, rows, columns]</c> constant with entry <c>value(i, j)</c>.</summary>
    public static Tensor<T> Constant(int rows, int columns, Func<int, int, double> value)
    {
        var shape = new[] { 1, rows, columns };
        var values = new double[rows * columns];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                values[i * columns + j] = value(i, j);
        using var _ = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
        var doubles = new Tensor<double>(values, shape);
        return typeof(T) == typeof(double) ? (Tensor<T>)(object)doubles : doubles.Cast<T>();
    }

    /// <summary>A constant of <paramref name="shape"/> with every entry <paramref name="value"/>.</summary>
    public static Tensor<T> Filled(int[] shape, double value)
    {
        var tensor = new Tensor<T>(shape);
        if (value != 0) tensor.AsWritableSpan().Fill(NumOps.FromDouble(value));
        return tensor;
    }
}
