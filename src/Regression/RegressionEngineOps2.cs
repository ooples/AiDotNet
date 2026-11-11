namespace AiDotNet.Regression;

/// <summary>
/// Whole-array engine building blocks for the regression models (index tensors, scatter/gather,
/// row selection), so the models do not walk vectors and matrices one element at a time.
/// </summary>
internal static class RegressionEngineOps2
{
    /// <summary>
    /// Wraps integer positions as a rank-1 <see cref="Tensor{T}"/> of <see cref="int"/> for the
    /// engine's index operations.
    /// </summary>
    public static Tensor<int> IndexTensor(IReadOnlyList<int> indices)
    {
        var data = new int[indices.Count];
        if (indices is int[] array)
        {
            Array.Copy(array, data, data.Length);
        }
        else if (indices is List<int> list)
        {
            list.CopyTo(data);
        }
        else
        {
            data = indices.ToArray();
        }

        return new Tensor<int>(data, new[] { data.Length });
    }

    /// <summary>
    /// A length-<paramref name="length"/> vector of zeros with <c>values[k]</c> written at
    /// position <c>indices[k]</c> (later duplicates win).
    /// </summary>
    public static Vector<T> ScatterToLength<T>(int length, IReadOnlyList<int> indices, Vector<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var zeros = engine.FillZero<T>(length);
        if (indices.Count == 0)
        {
            return zeros;
        }

        return engine.TensorIndexCopy(zeros.AsTensor(), 0, IndexTensor(indices), values.AsTensor()).ToVector();
    }

    /// <summary>
    /// The elements of <paramref name="values"/> at <paramref name="indices"/>, in that order.
    /// </summary>
    public static Vector<T> Gather<T>(Vector<T> values, IReadOnlyList<int> indices)
    {
        if (indices.Count == 0)
        {
            return new Vector<T>(0);
        }

        return AiDotNetEngine.Current.TensorIndexSelect(values.AsTensor(), IndexTensor(indices), 0).ToVector();
    }

    /// <summary>
    /// The rows of <paramref name="matrix"/> at <paramref name="indices"/>, in that order.
    /// </summary>
    public static Matrix<T> GatherRows<T>(Matrix<T> matrix, IReadOnlyList<int> indices)
    {
        if (indices.Count == 0)
        {
            return new Matrix<T>(0, matrix.Columns);
        }

        return AiDotNetEngine.Current
            .TensorIndexSelect(Tensor<T>.FromMatrix(matrix), IndexTensor(indices), 0).ToMatrix();
    }

    /// <summary>
    /// The columns of <paramref name="matrix"/> at <paramref name="indices"/>, in that order.
    /// </summary>
    public static Matrix<T> GatherColumns<T>(Matrix<T> matrix, IReadOnlyList<int> indices)
    {
        if (indices.Count == 0)
        {
            return new Matrix<T>(matrix.Rows, 0);
        }

        return AiDotNetEngine.Current
            .TensorIndexSelect(Tensor<T>.FromMatrix(matrix), IndexTensor(indices), 1).ToMatrix();
    }

    /// <summary>
    /// True when any element of <paramref name="mask"/> is non-zero.
    /// </summary>
    public static bool Any<T>(Tensor<T> mask)
    {
        return mask.Length > 0 && AiDotNetEngine.Current.TensorCountNonzero(mask) > 0;
    }

    /// <summary>
    /// Index of the first non-zero element of <paramref name="mask"/>, or -1 when there is none.
    /// </summary>
    public static int FirstNonzero<T>(Tensor<T> mask)
    {
        if (mask.Length == 0)
        {
            return -1;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var flat = mask.Rank == 1 ? mask : mask.Reshape(new[] { mask.Length });
        var notZero = AiDotNetEngine.Current.TensorNotEqualMask(flat, numOps);
        int first = AiDotNetEngine.Current.TensorArgMax(notZero, 0)[0];
        return numOps.Equals(notZero[first], numOps.Zero) ? -1 : first;
    }

    private static Tensor<T> TensorNotEqualMask<T>(this IEngine engine, Tensor<T> tensor, INumericOperations<T> numOps)
    {
        // 1 where tensor != 0 (NaN counts as non-zero, like !Equals(x, 0)), else 0.
        return engine.ScalarMinusTensor(numOps.One, engine.TensorEquals(tensor, numOps.Zero));
    }
}
