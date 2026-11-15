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
        int cols = source.Columns;
        var result = new Matrix<T>(indices.Length, cols);
        if (indices.Length == 0 || cols == 0) return result;

        // One block copy per requested row, straight from the matrix storage. Converting the matrix to a
        // tensor first would copy the whole dataset on every call, so a batch would cost O(rows * cols)
        // time and memory instead of O(batch * cols).
        var sourceSpan = source.AsSpan();
        var resultSpan = result.AsWritableSpan();
        for (int i = 0; i < indices.Length; i++)
        {
            int row = indices[i];
            if ((uint)row >= (uint)source.Rows)
                throw new ArgumentOutOfRangeException(
                    nameof(indices), $"Row index {row} at position {i} is outside [0, {source.Rows}).");
            sourceSpan.Slice(row * cols, cols).CopyTo(resultSpan.Slice(i * cols, cols));
        }

        return result;
    }

    /// <summary>The integer labels 0..count-1 as a new <typeparamref name="T"/> array (one bulk cast).</summary>
    internal static T[] Ramp<T>(int count)
    {
        if (count <= 0) return Array.Empty<T>();
        return new Tensor<int>(Enumerable.Range(0, count).ToArray(), new[] { count }).Cast<T>().AsSpan().ToArray();
    }

    /// <summary>Rounds integral-valued labels to their int class ids with the engine (round half to even).</summary>
    internal static int[] RoundToInt32<T>(Vector<T> values)
    {
        if (values.Length == 0) return Array.Empty<int>();
        return AiDotNetEngine.Current.TensorRound(Tensor<T>.FromVector(values)).Cast<int>().AsSpan().ToArray();
    }
}
