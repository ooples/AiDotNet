namespace AiDotNet.Regression;

/// <summary>
/// Whole-array engine building blocks shared by the regression models (row partitioning, masks,
/// index extraction), so the models do not walk vectors one element at a time.
/// </summary>
internal static class RegressionEngineOps
{
    /// <summary>
    /// Returns the positions where <paramref name="mask"/> is non-zero, in ascending order.
    /// </summary>
    public static int[] NonzeroIndices<T>(Tensor<T> mask)
    {
        return AiDotNetEngine.Current.TensorNonzero(mask).ToArray();
    }

    /// <summary>
    /// A 0/1 mask that is 1 where <c>values[i] &lt;= threshold</c>. NaN compares false, so it is 0.
    /// </summary>
    public static Tensor<T> LessOrEqualMask<T>(Vector<T> values, T threshold)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorLessEqual(
            Tensor<T>.FromVector(values),
            Tensor<T>.FromVector(engine.Fill(values.Length, threshold)));
    }

    /// <summary>
    /// Splits sample positions by <c>values[i] &lt;= threshold</c> (left) versus everything else,
    /// NaN included (right). Both index arrays are ascending, matching a sequential scan.
    /// </summary>
    public static (int[] Left, int[] Right) PartitionByThreshold<T>(Vector<T> values, T threshold)
    {
        if (values.Length == 0)
        {
            return (Array.Empty<int>(), Array.Empty<int>());
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var leftMask = LessOrEqualMask(values, threshold);
        var rightMask = AiDotNetEngine.Current.ScalarMinusTensor(numOps.One, leftMask);
        return (NonzeroIndices(leftMask), NonzeroIndices(rightMask));
    }

    /// <summary>
    /// Gathers <paramref name="source"/> at <paramref name="indices"/> (in the given order).
    /// </summary>
    public static Vector<T> Gather<T>(Vector<T> source, int[] indices)
    {
        if (indices.Length == 0)
        {
            return new Vector<T>(0);
        }

        return AiDotNetEngine.Current.TensorGather(
            Tensor<T>.FromVector(source),
            new Tensor<int>(indices, new[] { indices.Length })).ToVector();
    }
}
