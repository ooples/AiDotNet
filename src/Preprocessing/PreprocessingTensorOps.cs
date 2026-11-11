using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing;

/// <summary>
/// Whole-array engine helpers shared by the preprocessing transformers (column/row gathers, filled tensors,
/// ranked index selection and the scalar decisions iterative fits make once per iteration).
/// </summary>
internal static class PreprocessingTensorOps
{
    /// <summary>Gathers the given columns of <paramref name="data"/> into a new matrix, in the given order.</summary>
    public static Matrix<T> SelectColumns<T>(Matrix<T> data, int[] columns)
    {
        if (columns.Length == 0 || data.Rows == 0)
            return new Matrix<T>(data.Rows, columns.Length);

        var indices = new Tensor<int>((int[])columns.Clone(), new[] { columns.Length });
        return AiDotNetEngine.Current.TensorIndexSelect(Tensor<T>.FromMatrix(data), indices, 1).ToMatrix();
    }

    /// <summary>Gathers the given rows of <paramref name="data"/> into a new matrix, in the given order.</summary>
    public static Matrix<T> SelectRows<T>(Matrix<T> data, int[] rows)
    {
        if (rows.Length == 0 || data.Columns == 0)
            return new Matrix<T>(rows.Length, data.Columns);

        var indices = new Tensor<int>((int[])rows.Clone(), new[] { rows.Length });
        return AiDotNetEngine.Current.TensorIndexSelect(Tensor<T>.FromMatrix(data), indices, 0).ToMatrix();
    }

    /// <summary>Gathers the given elements of <paramref name="values"/> into a new vector, in the given order.</summary>
    public static Vector<T> SelectElements<T>(Vector<T> values, int[] positions)
    {
        if (positions.Length == 0)
            return new Vector<T>(0);

        var indices = new Tensor<int>((int[])positions.Clone(), new[] { positions.Length });
        return AiDotNetEngine.Current.TensorIndexSelect(Tensor<T>.FromVector(values), indices, 0).ToVector();
    }

    /// <summary>A tensor of the given shape with every element set to <paramref name="value"/>.</summary>
    public static Tensor<T> Filled<T>(int[] shape, T value)
    {
        var result = new Tensor<T>(shape);
        AiDotNetEngine.Current.TensorFill(result, value);
        return result;
    }

    /// <summary>
    /// The positions of the <paramref name="k"/> largest entries of a 1-D score tensor, returned in ascending
    /// index order (the usual "select top-k features, keep column order" step).
    /// </summary>
    public static int[] TopKIndicesSorted<T>(Tensor<T> scores, int k)
    {
        int count = Math.Min(Math.Max(k, 0), scores.Length);
        if (count == 0)
            return Array.Empty<int>();

        var order = AiDotNetEngine.Current.TensorArgsort(scores, 0, true);
        var selected = order.AsSpan().Slice(0, count).ToArray();
        Array.Sort(selected);
        return selected;
    }

    /// <summary>
    /// The positions of the <paramref name="k"/> smallest entries of a 1-D score tensor, returned in ascending
    /// index order.
    /// </summary>
    public static int[] BottomKIndicesSorted<T>(Tensor<T> scores, int k)
    {
        int count = Math.Min(Math.Max(k, 0), scores.Length);
        if (count == 0)
            return Array.Empty<int>();

        var order = AiDotNetEngine.Current.TensorArgsort(scores, 0, false);
        var selected = order.AsSpan().Slice(0, count).ToArray();
        Array.Sort(selected);
        return selected;
    }

    /// <summary>The positions where a 0/1 mask tensor is non-zero, in ascending order.</summary>
    public static int[] MaskPositions<T>(Tensor<T> mask)
    {
        var flat = mask.Rank == 1 ? mask : AiDotNetEngine.Current.Reshape(mask, new[] { mask.Length });
        var coordinates = AiDotNetEngine.Current.TensorNonzero(flat);
        return coordinates.AsSpan().ToArray();
    }

    /// <summary>The number of non-zero entries of a 0/1 mask tensor.</summary>
    public static int CountNonZero<T>(Tensor<T> mask) => AiDotNetEngine.Current.TensorCountNonzero(mask);

    /// <summary>A per-iteration scalar decision: whether <paramref name="value"/> is below <paramref name="limit"/>.</summary>
    public static bool IsBelow<T>(T value, T limit) => MathHelper.GetNumericOperations<T>().LessThan(value, limit);

    /// <summary>A per-iteration scalar decision: whether <paramref name="value"/> is above <paramref name="limit"/>.</summary>
    public static bool IsAbove<T>(T value, T limit) => MathHelper.GetNumericOperations<T>().GreaterThan(value, limit);

    /// <summary><c>max(a, b)</c> of two scalars.</summary>
    public static T ScalarMax<T>(T a, T b)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.GreaterThan(a, b) ? a : b;
    }
}
