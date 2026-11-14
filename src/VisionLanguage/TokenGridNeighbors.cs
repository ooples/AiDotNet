using AiDotNet.Tensors.Engines;

namespace AiDotNet.VisionLanguage;

/// <summary>
/// Neighbour lookups on a row-major square token grid that respect row boundaries: the neighbour of a
/// token at column 0 in direction "left" does not exist (it is not the last token of the previous row).
/// </summary>
/// <typeparam name="T">The numeric type of the token values.</typeparam>
internal static class TokenGridNeighbors<T>
{
    /// <summary>
    /// Returns, for every grid position (row, col), the value at (row + rowOffset, col + colOffset), or 0
    /// when that position lies outside the grid.
    /// </summary>
    /// <param name="engine">Engine to run on.</param>
    /// <param name="values">gridSize * gridSize values in row-major order (read flattened).</param>
    /// <param name="gridSize">Side length of the grid.</param>
    /// <param name="rowOffset">Row offset of the neighbour.</param>
    /// <param name="colOffset">Column offset of the neighbour.</param>
    /// <returns>A [gridSize * gridSize] tensor of neighbour values.</returns>
    internal static Tensor<T> Shift(IEngine engine, Tensor<T> values, int gridSize, int rowOffset, int colOffset)
    {
        int n = gridSize * gridSize;
        if (values.Length != n)
            throw new ArgumentException($"Expected {n} grid values but got {values.Length}.", nameof(values));
        var flat = values.Reshape([n]);
        int offset = rowOffset * gridSize + colOffset;
        Tensor<T> shifted;
        if (offset == 0)
            shifted = flat;
        else if (Math.Abs(offset) >= n)
            shifted = new Tensor<T>([n]);
        else if (offset > 0)
            shifted = engine.TensorConcatenate([engine.TensorSlice(flat, [offset], [n - offset]), new Tensor<T>([offset])], 0);
        else
            shifted = engine.TensorConcatenate([new Tensor<T>([-offset]), engine.TensorSlice(flat, [0], [n + offset])], 0);
        if (colOffset == 0)
            return shifted;

        // With the column in range, the flat index t + offset is in range exactly when the row is, so
        // masking the columns that would leave the row gives the 2-D neighbour rule.
        var targetColumn = engine.TensorAddScalar(
            engine.TensorRemainder(engine.TensorArange<int>(0, n, 1), Tensor<int>.CreateDefault([n], gridSize)), colOffset);
        var columnInRange = engine.TensorMultiply(
            engine.TensorGreaterThan(targetColumn, -1), engine.TensorLessThan(targetColumn, gridSize)).Cast<T>();
        return engine.TensorWhere(columnInRange, shifted, new Tensor<T>([n]));
    }
}
