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

        // Validate all items are of the same type as the first item
        var firstItemType = items[0]?.GetType();
        for (int i = 1; i < items.Count; i++)
        {
            var currentType = items[i]?.GetType();
            if (currentType != firstItemType)
            {
                throw new ArgumentException(
                    $"Cannot aggregate {itemTypeName}s of mixed types. " +
                    $"Item 0 is {firstItemType?.Name ?? "null"}, but item {i} is {currentType?.Name ?? "null"}. " +
                    $"All items must be of the same type.");
            }
        }

        // Handle Matrix<T> aggregation
        if (items[0] is Matrix<T>)
        {
            return CastToDataType<Matrix<T>, TData>(AggregateMatrices<T>(items.Cast<Matrix<T>>().ToList()));
        }

        // Handle Vector<T> aggregation
        if (items[0] is Vector<T>)
        {
            return CastToDataType<Vector<T>, TData>(AggregateVectors<T>(items.Cast<Vector<T>>().ToList()));
        }

        // Handle Tensor<T> aggregation
        if (items[0] is Tensor<T>)
        {
            return CastToDataType<Tensor<T>, TData>(AggregateTensors<T>(items.Cast<Tensor<T>>().ToList()));
        }

        // Unsupported type - throw exception to prevent silent data loss
        throw new NotSupportedException(
            $"Cannot aggregate {items.Count} {itemTypeName}s of type {typeof(TData).Name}. " +
            $"Supported types are Matrix<T>, Vector<T>, and Tensor<T>. " +
            $"For other types, use a pre-batched data loader or implement custom aggregation.");
    }

    /// <summary>
    /// Casts a source type to a target type using implicit boxing.
    /// </summary>
    /// <remarks>
    /// This helper method avoids the explicit upcast to object that code analyzers
    /// flag as unnecessary. The boxing happens implicitly when assigning to object.
    /// </remarks>
    private static TTarget CastToDataType<TSource, TTarget>(TSource source) where TSource : class
    {
        // Boxing happens implicitly here, avoiding the explicit upcast warning
        object boxed = source;
        // Cast from boxed object to target type
        if (boxed is TTarget result)
        {
            return result;
        }
        throw new InvalidCastException($"Cannot cast {typeof(TSource).Name} to {typeof(TTarget).Name}");
    }

    /// <summary>
    /// Aggregates multiple matrices by concatenating rows.
    /// </summary>
    private static Matrix<T> AggregateMatrices<T>(List<Matrix<T>> matrices)
    {
        int cols = matrices[0].Columns;

        // Validate all matrices have the same column count
        for (int i = 1; i < matrices.Count; i++)
        {
            if (matrices[i].Columns != cols)
            {
                throw new ArgumentException(
                    $"Cannot aggregate matrices with different column counts. " +
                    $"Matrix 0 has {cols} columns, but matrix {i} has {matrices[i].Columns} columns.");
            }
        }

        int totalRows = matrices.Sum(m => m.Rows);
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
        var referenceShape = tensors[0].Shape;

        // Validate all tensors have compatible shapes (same dimensions except first)
        for (int i = 1; i < tensors.Count; i++)
        {
            var currentShape = tensors[i].Shape;
            if (currentShape.Length != referenceShape.Length)
            {
                throw new ArgumentException(
                    $"Cannot aggregate tensors with different ranks. " +
                    $"Tensor 0 has rank {referenceShape.Length}, but tensor {i} has rank {currentShape.Length}.");
            }

            for (int d = 1; d < referenceShape.Length; d++)
            {
                if (currentShape[d] != referenceShape[d])
                {
                    throw new ArgumentException(
                        $"Cannot aggregate tensors with different shapes. " +
                        $"Tensor 0 has shape [{string.Join(", ", referenceShape)}], " +
                        $"but tensor {i} has shape [{string.Join(", ", currentShape)}]. " +
                        $"Shapes must match for all dimensions except the first.");
                }
            }
        }

        int totalSamples = tensors.Sum(t => t.Shape[0]);

        // Create new shape with updated first dimension
        var newShape = (int[])referenceShape.Clone();
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
