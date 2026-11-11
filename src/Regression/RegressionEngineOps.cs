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
    /// Index of the first largest value (ties keep the earliest index, like a sequential
    /// strictly-greater scan). The sequence must not be empty.
    /// </summary>
    public static int ArgMaxFirst<T>(IEnumerable<T> values)
    {
        var tensor = Tensor<T>.FromVector(new Vector<T>(values));
        return AiDotNetEngine.Current.TensorArgMax(tensor, 0).ToArray()[0];
    }

    /// <summary>
    /// Index of the first smallest value (ties keep the earliest index, like a sequential
    /// strictly-less scan). The sequence must not be empty.
    /// </summary>
    public static int ArgMinFirst<T>(IEnumerable<T> values)
    {
        var tensor = Tensor<T>.FromVector(new Vector<T>(values));
        return AiDotNetEngine.Current.TensorArgMin(tensor, 0).ToArray()[0];
    }

    /// <summary>
    /// The first largest value among the positions where <paramref name="allowed"/> is true,
    /// ignoring NaN. Returns index -1 when no allowed, non-NaN value exists.
    /// </summary>
    public static (int Index, T Value) MaskedArgMaxFirst<T>(Vector<T> values, bool[] allowed)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        if (values.Length == 0)
        {
            return (-1, numOps.MinValue);
        }

        var tensor = Tensor<T>.FromVector(values);
        var floor = Tensor<T>.FromVector(engine.Fill(values.Length, numOps.MinValue));
        var usable = engine.TensorWhere(allowed, engine.TensorWhere(engine.TensorIsNan(tensor), floor, tensor), floor);
        int index = engine.TensorArgMax(usable, 0).ToArray()[0];
        if (!allowed[index] || numOps.IsNaN(values[index]))
        {
            return (-1, numOps.MinValue);
        }

        return (index, usable[index]);
    }

    /// <summary>
    /// Squared Euclidean distance from <paramref name="query"/> to every row of <paramref name="x"/>,
    /// summed in feature order per row. <paramref name="x"/> must have at least one row.
    /// </summary>
    public static Vector<T> RowSquaredDistances<T>(Matrix<T> x, Vector<T> query)
    {
        var engine = AiDotNetEngine.Current;
        var broadcastQuery = engine.TensorBroadcastTo(
            new Tensor<T>(new[] { 1, query.Length }, query),
            new[] { x.Rows, x.Columns });
        var diff = engine.TensorSubtract(Tensor<T>.FromMatrix(x), broadcastQuery);
        return engine.ReduceSum(engine.TensorSquare(diff), new[] { 1 }, keepDims: false).ToVector();
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
