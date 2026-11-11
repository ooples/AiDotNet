namespace AiDotNet.Data.Graph;

/// <summary>
/// Builds and reads the dense graph tensors shared by the graph loaders (edge index, node feature rows,
/// one-hot labels, adjacency) with bulk casts and engine ops instead of per-element tensor writes.
/// </summary>
internal static class GraphTensorHelper
{
    /// <summary>
    /// Builds a <c>[numEdges, 2]</c> COO edge index tensor from (source, target) pairs with one bulk cast.
    /// </summary>
    internal static Tensor<T> EdgeIndex<T>(IReadOnlyList<(int Source, int Target)> edges)
    {
        var flat = new int[edges.Count * 2];
        for (int i = 0; i < edges.Count; i++)
        {
            flat[2 * i] = edges[i].Source;
            flat[2 * i + 1] = edges[i].Target;
        }

        return new Tensor<int>(flat, new[] { edges.Count, 2 }).Cast<T>();
    }

    /// <summary>
    /// Reads a <c>[numEdges, 2]</c> edge index tensor as row-major integers
    /// (<c>result[2 * e]</c> = source, <c>result[2 * e + 1]</c> = target), rounding each value to the nearest
    /// integer exactly like <c>INumericOperations&lt;T&gt;.ToInt32</c>.
    /// </summary>
    internal static int[] EdgeIndexToInt32<T>(Tensor<T> edgeIndex)
    {
        if (edgeIndex.Length == 0) return Array.Empty<int>();
        return AiDotNetEngine.Current.TensorRound(edgeIndex).Cast<int>().AsSpan().ToArray();
    }

    /// <summary>
    /// Builds a <c>[rows.Count, width]</c> tensor from equally sized double rows with one bulk cast.
    /// </summary>
    internal static Tensor<T> Rows<T>(IReadOnlyList<double[]> rows, int width)
    {
        var flat = new double[rows.Count * width];
        for (int i = 0; i < rows.Count; i++)
        {
            double[] row = rows[i];
            if (row.Length < width)
                throw new ArgumentException(
                    $"Row {i} has {row.Length} values but {width} are required.");
            Array.Copy(row, 0, flat, i * width, width);
        }

        return new Tensor<double>(flat, new[] { rows.Count, width }).Cast<T>();
    }

    /// <summary>
    /// Builds a <c>[count, width]</c> tensor from rows <c>start .. start + count - 1</c> of a possibly ragged
    /// row list: each row contributes its first <c>min(width, row.Length)</c> values and rows past the end of
    /// the list stay zero. One bulk cast.
    /// </summary>
    internal static Tensor<T> PaddedRows<T>(IReadOnlyList<double[]> rows, int start, int count, int width)
    {
        var flat = new double[count * width];
        for (int i = 0; i < count && start + i < rows.Count; i++)
        {
            double[] row = rows[start + i];
            Array.Copy(row, 0, flat, i * width, Math.Min(width, row.Length));
        }

        return new Tensor<double>(flat, new[] { count, width }).Cast<T>();
    }

    /// <summary>
    /// One-hot encodes <paramref name="classIndices"/> into a <c>[count, depth]</c> tensor with the engine.
    /// Every index must lie in <c>[0, depth)</c>.
    /// </summary>
    internal static Tensor<T> OneHot<T>(int[] classIndices, int depth)
    {
        for (int i = 0; i < classIndices.Length; i++)
        {
            if ((uint)classIndices[i] >= (uint)depth)
                throw new ArgumentOutOfRangeException(
                    nameof(classIndices), $"Class index {classIndices[i]} at position {i} is outside [0, {depth}).");
        }

        if (classIndices.Length == 0) return new Tensor<T>(new[] { 0, depth });
        return AiDotNetEngine.Current.TensorOneHot<T>(
            new Tensor<int>(classIndices, new[] { classIndices.Length }), depth);
    }

    /// <summary>
    /// One-hot encodes the first <paramref name="classIndices"/>.Length rows of a <c>[rowCount, depth]</c>
    /// tensor; the remaining rows stay zero.
    /// </summary>
    internal static Tensor<T> OneHotPadded<T>(int[] classIndices, int rowCount, int depth)
    {
        if (classIndices.Length >= rowCount)
            return OneHot<T>(classIndices.Length == rowCount ? classIndices : classIndices.Take(rowCount).ToArray(), depth);

        if (classIndices.Length == 0) return new Tensor<T>(new[] { rowCount, depth });

        var encoded = OneHot<T>(classIndices, depth);
        var padding = new Tensor<T>(new[] { rowCount - classIndices.Length, depth });
        return AiDotNetEngine.Current.TensorConcatenate(new[] { encoded, padding }, 0);
    }

    /// <summary>
    /// Sets <c>tensor[i0[k], i1[k], ...] = value</c> for every k with one engine index-put.
    /// </summary>
    internal static Tensor<T> IndexPut<T>(Tensor<T> tensor, int[][] indices, T value)
    {
        int count = indices.Length == 0 ? 0 : indices[0].Length;
        if (count == 0) return tensor;

        var indexTensors = new Tensor<int>[indices.Length];
        for (int axis = 0; axis < indices.Length; axis++)
            indexTensors[axis] = new Tensor<int>(indices[axis], new[] { count });

        var values = new Tensor<T>(new[] { count });
        values.Fill(value);
        return AiDotNetEngine.Current.TensorIndexPut(tensor, indexTensors, values);
    }
}
