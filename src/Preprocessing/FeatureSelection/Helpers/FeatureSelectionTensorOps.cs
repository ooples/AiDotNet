using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Helpers;

/// <summary>
/// Whole-array engine helpers shared by the feature selectors: column gathers, filled tensors, per-column
/// moments and ranked index selection with LINQ-compatible tie order.
/// </summary>
internal static class FeatureSelectionTensorOps
{
    /// <summary>Gathers the given columns of <paramref name="data"/> into a new matrix, in the given order.</summary>
    public static Matrix<T> SelectColumns<T>(Matrix<T> data, int[] columns)
    {
        if (columns.Length == 0 || data.Rows == 0)
            return new Matrix<T>(data.Rows, columns.Length);

        var indices = new Tensor<int>((int[])columns.Clone(), new[] { columns.Length });
        return AiDotNetEngine.Current.TensorIndexSelect(Tensor<T>.FromMatrix(data), indices, 1).ToMatrix();
    }

    /// <summary>A tensor of the given shape with every element set to <paramref name="value"/>.</summary>
    public static Tensor<T> Filled<T>(int[] shape, T value)
    {
        var result = new Tensor<T>(shape);
        AiDotNetEngine.Current.TensorFill(result, value);
        return result;
    }

    /// <summary>Element-wise <c>1 - x</c> (logical not of a 0/1 mask).</summary>
    public static Tensor<T> OneMinus<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorAddScalar(engine.TensorNegate(x), MathHelper.GetNumericOperations<T>().One);
    }

    /// <summary>Column-wise minimum of a 2-D tensor ([n, p] to [p]).</summary>
    public static Tensor<T> ColumnMin<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(x), new[] { 0 }, false));
    }

    /// <summary>Column-wise maximum of a 2-D tensor ([n, p] to [p]).</summary>
    public static Tensor<T> ColumnMax<T>(Tensor<T> x) => AiDotNetEngine.Current.ReduceMax(x, new[] { 0 }, false);

    /// <summary>
    /// Per-column population moments of a 2-D tensor ([n, p]): the column means ([1, p]), the population
    /// standard deviations ([p]) and the centred data ([n, p]).
    /// </summary>
    public static Tensor<T> CenterColumns<T>(Tensor<T> x, out Tensor<T> mean, out Tensor<T> std)
    {
        var engine = AiDotNetEngine.Current;
        mean = engine.ReduceMean(x, new[] { 0 }, true);
        var centered = engine.TensorSubtract(x, mean);
        std = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(centered), new[] { 0 }, false));
        return centered;
    }

    /// <summary>
    /// The positions of the <paramref name="k"/> largest entries of a 1-D score tensor, returned in ascending
    /// index order; ties keep the lower index first (LINQ <c>OrderByDescending(...).Take(k).OrderBy(i)</c>).
    /// </summary>
    public static int[] TopKIndicesSorted<T>(Tensor<T> scores, int k)
    {
        int count = Math.Min(Math.Max(k, 0), scores.Length);
        if (count == 0)
            return Array.Empty<int>();

        var selected = StableOrder(scores, descending: true).AsSpan().Slice(0, count).ToArray();
        Array.Sort(selected);
        return selected;
    }

    /// <summary>
    /// The positions of the <paramref name="k"/> smallest entries of a 1-D score tensor, returned in ascending
    /// index order; ties keep the lower index first (LINQ <c>OrderBy(...).Take(k).OrderBy(i)</c>).
    /// </summary>
    public static int[] BottomKIndicesSorted<T>(Tensor<T> scores, int k)
    {
        int count = Math.Min(Math.Max(k, 0), scores.Length);
        if (count == 0)
            return Array.Empty<int>();

        var selected = StableOrder(scores, descending: false).AsSpan().Slice(0, count).ToArray();
        Array.Sort(selected);
        return selected;
    }

    /// <summary>
    /// The indices that sort a 1-D tensor ascending (or descending), with tied values kept in their original
    /// index order like LINQ <c>OrderBy</c>/<c>OrderByDescending</c>. The engine argsort is not documented as
    /// stable, so it sorts a distinct composite key (rank of the value, then the index).
    /// </summary>
    public static int[] StableOrder<T>(Tensor<T> values, bool descending)
    {
        int count = values.Length;
        if (count == 0)
            return Array.Empty<int>();

        var engine = AiDotNetEngine.Current;
        var flat = values.Rank == 1 ? values : engine.Reshape(values, new[] { count });
        var sorted = engine.TensorSort(flat, 0, false).Item1;

        // ascending rank: #{m : v[m] < v[j]}; descending rank: #{m : v[m] > v[j]} = count - #{m : v[m] <= v[j]}
        var less = engine.TensorSearchSorted(sorted, flat, false).AsSpan().ToArray();
        var atMost = engine.TensorSearchSorted(sorted, flat, true).AsSpan().ToArray();
        var keys = new long[count];
        for (int j = 0; j < count; j++)
        {
            long rank = descending ? count - atMost[j] : less[j];
            keys[j] = rank * count + j;
        }

        return engine.TensorArgsort(new Tensor<long>(keys, new[] { count }), 0, false).AsSpan().ToArray();
    }

    /// <summary>The positions where a 0/1 mask tensor is non-zero, in ascending order.</summary>
    public static int[] MaskPositions<T>(Tensor<T> mask)
    {
        var flat = mask.Rank == 1 ? mask : AiDotNetEngine.Current.Reshape(mask, new[] { mask.Length });
        return AiDotNetEngine.Current.TensorNonzero(flat).AsSpan().ToArray();
    }

    /// <summary>The number of non-zero entries of a 0/1 mask tensor.</summary>
    public static int CountNonZero<T>(Tensor<T> mask) => AiDotNetEngine.Current.TensorCountNonzero(mask);
}
