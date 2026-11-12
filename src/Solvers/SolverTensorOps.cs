using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Solvers;

/// <summary>
/// Whole-tensor building blocks shared by the solvers' input validation: element-wise flags as 0/1 masks of
/// <typeparamref name="T"/> and the first flagged position in row-major order, which is the element a scalar
/// scan would have stopped at.
/// </summary>
/// <typeparam name="T">The numeric element type.</typeparam>
internal static class SolverTensorOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    internal static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>A rank-1 tensor holding a copy of <paramref name="vector"/>.</summary>
    internal static Tensor<T> AsTensor(Vector<T> vector) => Tensor<T>.FromVector(vector);

    /// <summary>A rank-2 tensor holding a copy of <paramref name="matrix"/>.</summary>
    internal static Tensor<T> AsTensor(Matrix<T> matrix) => Tensor<T>.FromMatrix(matrix);

    /// <summary>1 where <paramref name="x"/> is NaN or infinite, 0 elsewhere.</summary>
    internal static Tensor<T> NonFinite(Tensor<T> x) => Select(Engine.TensorIsFinite(x), 0.0, 1.0, x);

    /// <summary>1 where <paramref name="x"/> is NaN, 0 elsewhere.</summary>
    internal static Tensor<T> NaN(Tensor<T> x) => Select(Engine.TensorIsNan(x), 1.0, 0.0, x);

    /// <summary><c>v &gt; 0 ? v : 0</c> per element: NaN fails the test and becomes zero.</summary>
    internal static Tensor<T> PositivePart(Tensor<T> x)
        => Engine.TensorWhere(
            Engine.TensorGreaterThan(x, NumOps.Zero),
            x,
            Tensor<T>.CreateDefault(x.Shape.ToArray(), NumOps.Zero));

    /// <summary>
    /// The index the scan <c>if (v &gt; best) { best = v; index = i; }</c> from <c>best = floor</c> ends on over a
    /// rank-1 <paramref name="values"/>: NaN never wins and ties keep the lowest index. -1 when nothing beats
    /// <paramref name="floor"/>.
    /// </summary>
    internal static int FirstStrictMaxIndex(Tensor<T> values, T floor)
    {
        var eligible = Engine.TensorGreaterThan(values, floor);
        if (First(eligible) is null) return -1;
        var candidates = Engine.TensorWhere(eligible, values, Tensor<T>.CreateDefault(values.Shape.ToArray(), floor));
        T best = Engine.TensorMaxValue(candidates);
        var winners = Engine.TensorMultiply(eligible, Engine.TensorEquals(candidates, best));
        return First(winners)?[0] ?? -1;
    }

    /// <summary>0/1 values of <typeparamref name="T"/> for <paramref name="flags"/>.</summary>
    internal static Tensor<T> Mask(IReadOnlyList<bool> flags)
    {
        var ids = new int[flags.Count];
        for (int i = 0; i < ids.Length; i++) ids[i] = flags[i] ? 1 : 0;
        return Engine.TensorTake(Engine.TensorArange<T>(0, 2, 1), new Tensor<int>(new[] { ids.Length }, new Vector<int>(ids)));
    }

    /// <summary>The element-wise maximum of two 0/1 masks (logical or).</summary>
    internal static Tensor<T> Or(Tensor<T> a, Tensor<T> b) => Engine.TensorMax(a, b);

    /// <summary>
    /// The coordinates of the first non-zero entry of <paramref name="mask"/> in row-major order, or <c>null</c>
    /// when every entry is zero.
    /// </summary>
    internal static int[]? First(Tensor<T> mask)
    {
        var hits = Engine.TensorNonzero(mask);
        if (hits.Shape[0] == 0) return null;
        int rank = hits.Shape[1];
        var first = new int[rank];
        Array.Copy(hits.ToArray(), first, rank);
        return first;
    }

    /// <summary>
    /// The result of folding <c>acc = acc &gt; v ? acc : v</c> over <paramref name="values"/> from
    /// <paramref name="start"/>. A NaN element replaces the accumulator and the next element replaces the NaN, so
    /// the result is the largest element after the last NaN (NaN when the last element is NaN), or the largest of
    /// <paramref name="start"/> and every element when there is no NaN.
    /// </summary>
    internal static T FoldMax(T start, Tensor<T> values)
    {
        int n = values.Length;
        if (n == 0) return start;
        var flat = Engine.Reshape(values.Contiguous(), new[] { n });
        var nan = First(Engine.TensorFlip(NaN(flat), new[] { 0 }));
        if (nan is null)
        {
            T largest = Engine.TensorMaxValue(flat);
            return NumOps.GreaterThan(start, largest) ? start : largest;
        }

        int lastNaN = n - 1 - nan[0];
        if (lastNaN == n - 1) return flat.GetFlat(lastNaN);
        return Engine.TensorMaxValue(Engine.TensorNarrow(flat, 0, lastNaN + 1, n - lastNaN - 1));
    }

    /// <summary>
    /// The result of <c>if (v &gt; acc) acc = v</c> over <paramref name="values"/> from <paramref name="start"/>:
    /// NaN elements never win.
    /// </summary>
    internal static T FoldMaxSkippingNaN(T start, Tensor<T> values)
    {
        if (values.Length == 0) return start;
        var flat = Engine.Reshape(values.Contiguous(), new[] { values.Length });
        var clean = Engine.TensorWhere(Engine.TensorIsNan(flat), Tensor<T>.CreateDefault(new[] { flat.Length }, start), flat);
        T largest = Engine.TensorMaxValue(clean);
        return NumOps.GreaterThan(largest, start) ? largest : start;
    }

    /// <summary><c>matrix · vector</c> as a rank-1 tensor of <c>matrix.Rows</c> entries.</summary>
    internal static Tensor<T> MatVec(Tensor<T> matrix, Tensor<T> vector)
        => Engine.Reshape(Engine.TensorMatMul(matrix, Engine.Reshape(vector, new[] { vector.Length, 1 })), new[] { matrix.Shape[0] });

    /// <summary><c>matrixᵀ · vector</c> as a rank-1 tensor of <c>matrix.Columns</c> entries.</summary>
    internal static Tensor<T> TransposedMatVec(Tensor<T> matrix, Tensor<T> vector)
        => Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(vector, new[] { 1, vector.Length }), matrix), new[] { matrix.Shape[1] });

    private static Tensor<T> Select(Tensor<AiDotNet.Tensors.Bit> condition, double whenTrue, double whenFalse, Tensor<T> like)
    {
        var shape = like.Shape.ToArray();
        return Engine.TensorWhere(
            condition,
            Tensor<T>.CreateDefault(shape, NumOps.FromDouble(whenTrue)),
            Tensor<T>.CreateDefault(shape, NumOps.FromDouble(whenFalse)));
    }
}
