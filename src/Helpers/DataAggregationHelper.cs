using AiDotNet.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Helper class for aggregating data samples.
/// </summary>
/// <remarks>
/// <para>
/// DataAggregationHelper provides utility methods for combining multiple
/// data samples (Matrix, Vector, or Tensor) into a single aggregated structure.
/// </para>
/// <para><b>For Beginners:</b> When loading data in batches or streaming,
/// you often need to combine multiple smaller pieces into one larger structure.
/// This helper provides optimized methods for this common operation.
/// </para>
/// </remarks>
public static class DataAggregationHelper
{
    /// <summary>
    /// Aggregates a list of data samples into a single structure.
    /// </summary>
    /// <typeparam name="T">The numeric type of the elements.</typeparam>
    /// <typeparam name="TData">The data type (Matrix, Vector, or Tensor).</typeparam>
    /// <param name="items">The list of items to aggregate.</param>
    /// <param name="itemTypeName">The name used in error messages (e.g., "input" or "output").</param>
    /// <returns>An aggregated structure containing all samples.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the list is empty.</exception>
    /// <exception cref="NotSupportedException">Thrown when the data type is not supported.</exception>
    /// <remarks>
    /// <para>
    /// This method supports aggregation of Matrix&lt;T&gt;, Vector&lt;T&gt;, and Tensor&lt;T&gt;
    /// types. For matrices, rows are concatenated vertically. For vectors, elements are
    /// concatenated sequentially. For tensors, samples are concatenated along the first dimension.
    /// </para>
    /// </remarks>
    public static TData Aggregate<T, TData>(List<TData> items, string itemTypeName)
    {
        if (items.Count == 0)
        {
            throw new InvalidOperationException($"Cannot aggregate empty {itemTypeName} list.");
        }

        // If items are already in the right format (single item)
        if (items.Count == 1)
        {
            return items[0];
        }

        // Handle Matrix<T> aggregation
        if (items[0] is Matrix<T>)
        {
            return (TData)(object)AggregateMatrices<T>(items.Cast<Matrix<T>>().ToList());
        }

        // Handle Vector<T> aggregation
        if (items[0] is Vector<T>)
        {
            return (TData)(object)AggregateVectors<T>(items.Cast<Vector<T>>().ToList());
        }

        // Handle Tensor<T> aggregation
        if (items[0] is Tensor<T>)
        {
            return (TData)(object)AggregateTensors<T>(items.Cast<Tensor<T>>().ToList());
        }

        // Unsupported type - throw exception to prevent silent data loss
        throw new NotSupportedException(
            $"Cannot aggregate {items.Count} {itemTypeName}s of type {typeof(TData).Name}. " +
            $"Supported types are Matrix<T>, Vector<T>, and Tensor<T>. " +
            $"For other types, use a pre-batched data loader or implement custom aggregation.");
    }

    /// <summary>
    /// Aggregates multiple matrices by concatenating rows.
    /// </summary>
    private static Matrix<T> AggregateMatrices<T>(List<Matrix<T>> matrices)
    {
        int totalRows = matrices.Sum(m => m.Rows);
        int cols = matrices[0].Columns;

        var result = new Matrix<T>(totalRows, cols);
        int currentRow = 0;
        foreach (var matrix in matrices)
        {
            for (int r = 0; r < matrix.Rows; r++)
            {
                result.SetRow(currentRow++, matrix.GetRow(r));
            }
        }
        return result;
    }

    /// <summary>
    /// Aggregates multiple vectors by concatenating elements.
    /// </summary>
    private static Vector<T> AggregateVectors<T>(List<Vector<T>> vectors)
    {
        int totalLength = vectors.Sum(v => v.Length);

        var result = new Vector<T>(totalLength);
        int currentIdx = 0;
        foreach (var vector in vectors)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                result[currentIdx++] = vector[i];
            }
        }
        return result;
    }

    /// <summary>
    /// Aggregates multiple tensors by concatenating along the first dimension.
    /// </summary>
    private static Tensor<T> AggregateTensors<T>(List<Tensor<T>> tensors)
    {
        int totalSamples = tensors.Sum(t => t.Shape[0]);

        // Create new shape with updated first dimension
        var newShape = (int[])tensors[0].Shape.Clone();
        newShape[0] = totalSamples;

        var result = new Tensor<T>(newShape);
        int currentSample = 0;
        foreach (var tensor in tensors)
        {
            for (int s = 0; s < tensor.Shape[0]; s++)
            {
                TensorCopyHelper.CopySample(tensor, result, s, currentSample++);
            }
        }
        return result;
    }
}
