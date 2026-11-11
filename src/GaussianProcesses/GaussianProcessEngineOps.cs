namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Whole-array engine helpers shared by the Gaussian process models (diagonal updates, finiteness checks).
/// </summary>
internal static class GaussianProcessEngineOps
{
    /// <summary>
    /// Returns <paramref name="matrix"/> + <paramref name="value"/> * I as a new matrix.
    /// </summary>
    internal static Matrix<T> AddToDiagonal<T>(Matrix<T> matrix, T value)
    {
        var engine = AiDotNetEngine.Current;
        var identity = engine.TensorEye<T>(matrix.Rows);
        if (matrix.Columns != matrix.Rows)
            identity = engine.TensorSlice(
                engine.TensorEye<T>(Math.Max(matrix.Rows, matrix.Columns)),
                new[] { 0, 0 },
                new[] { matrix.Rows, matrix.Columns });

        return engine.TensorAdd(Tensor<T>.FromMatrix(matrix), engine.TensorMultiplyScalar(identity, value)).ToMatrix();
    }

    /// <summary>
    /// Returns <paramref name="matrix"/> + diag(<paramref name="diagonal"/>) as a new square matrix.
    /// </summary>
    internal static Matrix<T> AddToDiagonal<T>(Matrix<T> matrix, Vector<T> diagonal)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorAdd(Tensor<T>.FromMatrix(matrix), engine.TensorDiag(Tensor<T>.FromVector(diagonal))).ToMatrix();
    }

    /// <summary>
    /// Adds <paramref name="value"/> to every diagonal entry of <paramref name="matrix"/> in place.
    /// </summary>
    internal static void AddToDiagonalInPlace<T>(Matrix<T> matrix, T value)
    {
        var updated = AddToDiagonal(matrix, value);
        updated.AsSpan().CopyTo(matrix.AsWritableSpan());
    }

    /// <summary>
    /// Adds diag(<paramref name="diagonal"/>) to <paramref name="matrix"/> in place.
    /// </summary>
    internal static void AddToDiagonalInPlace<T>(Matrix<T> matrix, Vector<T> diagonal)
    {
        var updated = AddToDiagonal(matrix, diagonal);
        updated.AsSpan().CopyTo(matrix.AsWritableSpan());
    }

    /// <summary>
    /// Main diagonal of a matrix as a vector.
    /// </summary>
    internal static Vector<T> Diagonal<T>(Matrix<T> matrix)
    {
        return AiDotNetEngine.Current.TensorDiagonal(Tensor<T>.FromMatrix(matrix)).ToVector();
    }

    /// <summary>
    /// True when every element is finite (no NaN and no infinity). An empty tensor is finite.
    /// </summary>
    internal static bool AllFinite<T>(Tensor<T> values)
    {
        if (values.Length == 0)
            return true;

        var engine = AiDotNetEngine.Current;
        return engine.TensorMinValue(engine.TensorIsFinite(values)) != AiDotNet.Tensors.Bit.False;
    }

    /// <summary>
    /// True when every element is finite (no NaN and no infinity). An empty vector is finite.
    /// </summary>
    internal static bool AllFinite<T>(Vector<T> values) => AllFinite(Tensor<T>.FromVector(values));

    /// <summary>
    /// True when every element is finite (no NaN and no infinity). An empty matrix is finite.
    /// </summary>
    internal static bool AllFinite<T>(Matrix<T> values) => AllFinite(Tensor<T>.FromMatrix(values));
}
