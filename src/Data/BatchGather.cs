namespace AiDotNet.Data;

/// <summary>
/// Gathers dataset rows by index for batching and subsetting with the engine's index-select
/// (<c>torch.index_select</c>) instead of per-element indexer copies.
/// </summary>
internal static class BatchGather
{
    /// <summary>Selects <paramref name="indices"/> along axis 0 of <paramref name="source"/>.</summary>
    internal static Tensor<T> Rows<T>(Tensor<T> source, int[] indices)
    {
        if (indices.Length == 0)
        {
            var shape = source.Shape.ToArray();
            shape[0] = 0;
            return new Tensor<T>(shape);
        }

        return AiDotNetEngine.Current.TensorIndexSelect(
            source, new Tensor<int>(indices, new[] { indices.Length }), 0);
    }

    /// <summary>Selects the elements at <paramref name="indices"/> of a vector.</summary>
    internal static Vector<T> Elements<T>(Vector<T> source, int[] indices)
    {
        if (indices.Length == 0) return new Vector<T>(0);
        return Rows(Tensor<T>.FromVector(source), indices).ToVector();
    }

    /// <summary>Selects the rows at <paramref name="indices"/> of a matrix (all columns).</summary>
    internal static Matrix<T> Rows<T>(Matrix<T> source, int[] indices)
    {
        if (indices.Length == 0) return new Matrix<T>(0, source.Columns);
        return Rows(Tensor<T>.FromMatrix(source), indices).ToMatrix();
    }
}
