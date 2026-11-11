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
        => MaskedArgMaxFirst(values, null, allowed);

    /// <summary>
    /// The first largest value among the positions where <paramref name="mask"/> is non-zero and
    /// <paramref name="allowed"/> is true (either may be null for "everywhere"), ignoring NaN.
    /// Returns index -1 when no such value exists.
    /// </summary>
    public static (int Index, T Value) MaskedArgMaxFirst<T>(Vector<T> values, Tensor<T>? mask, bool[]? allowed)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        if (values.Length == 0)
        {
            return (-1, numOps.MinValue);
        }

        var tensor = Tensor<T>.FromVector(values);
        var floor = Tensor<T>.FromVector(engine.Fill(values.Length, numOps.MinValue));
        var usable = engine.TensorWhere(engine.TensorIsNan(tensor), floor, tensor);
        if (mask is not null)
        {
            usable = engine.TensorWhere(mask, usable, floor);
        }

        if (allowed is not null)
        {
            usable = engine.TensorWhere(allowed, usable, floor);
        }

        int index = engine.TensorArgMax(usable, 0).ToArray()[0];
        bool masked = mask is not null && numOps.Equals(mask[index], numOps.Zero);
        bool disallowed = allowed is not null && !allowed[index];
        if (masked || disallowed || numOps.IsNaN(values[index]))
        {
            return (-1, numOps.MinValue);
        }

        return (index, usable[index]);
    }

    /// <summary>
    /// Picks the best split position from per-position scores over a sorted feature: the first
    /// largest score where <paramref name="mask"/> is non-zero and <paramref name="allowed"/> is true
    /// (either may be null), ignoring NaN. The threshold is the midpoint of the two sorted values
    /// around the split. Returns index -1 when no position qualifies.
    /// </summary>
    public static (int Index, T Score, T Threshold) BestSortedSplit<T>(
        Vector<T> scores, Tensor<T>? mask, bool[]? allowed, Vector<T> sortedFeature)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var (index, score) = MaskedArgMaxFirst(scores, mask, allowed);
        if (index < 0)
        {
            return (-1, numOps.MinValue, numOps.Zero);
        }

        T threshold = numOps.Divide(
            numOps.Add(sortedFeature[index], sortedFeature[index + 1]), numOps.FromDouble(2));
        return (index, score, threshold);
    }

    /// <summary>
    /// Weighted sum of squared deviations from running moments, element-wise:
    /// sumWy2 - sumWy^2 / sumW where sumW &gt; 0, else 0.
    /// </summary>
    public static Vector<T> WeightedSseFromMoments<T>(Vector<T> sumW, Vector<T> sumWy, Vector<T> sumWy2)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var raw = engine.Subtract(sumWy2, engine.Divide(engine.Multiply(sumWy, sumWy), sumW));
        return engine.TensorWhere(
            engine.TensorGreaterThan(Tensor<T>.FromVector(sumW), numOps.Zero),
            Tensor<T>.FromVector(raw),
            Tensor<T>.FromVector(engine.FillZero(sumW.Length))).ToVector();
    }

    /// <summary>
    /// Sample positions ordered by ascending key; equal keys keep their original order (stable),
    /// exactly like LINQ OrderBy over the positions.
    /// </summary>
    public static int[] StableAscendingOrder<T>(Vector<T> keys)
    {
        return keys
            .Select((key, index) => (Key: key, Index: index))
            .OrderBy(p => p.Key, Comparer<T>.Default)
            .Select(p => p.Index)
            .ToArray();
    }

    /// <summary>
    /// Running sums of <paramref name="sortedValues"/> for every split position: element s-1 is the
    /// sum of the first s values, for s = 1..n-1 (n-1 elements; empty when n &lt; 2).
    /// </summary>
    public static Vector<T> SplitPrefixSums<T>(Vector<T> sortedValues)
    {
        int n = sortedValues.Length;
        if (n < 2)
        {
            return new Vector<T>(0);
        }

        return AiDotNetEngine.Current.TensorCumSum(Tensor<T>.FromVector(sortedValues), 0)
            .ToVector().GetSubVector(0, n - 1);
    }

    /// <summary>
    /// Midpoints (v[s-1] + v[s]) / 2 of adjacent sorted values, for s = 1..n-1.
    /// </summary>
    public static Vector<T> AdjacentMidpoints<T>(Vector<T> sortedValues)
    {
        int n = sortedValues.Length;
        if (n < 2)
        {
            return new Vector<T>(0);
        }

        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        return engine.Divide(
            engine.Add(sortedValues.GetSubVector(0, n - 1), sortedValues.GetSubVector(1, n - 1)),
            numOps.FromDouble(2));
    }

    /// <summary>
    /// A 0/1 mask over split positions s = 1..n-1 that is 1 where v[s] &gt; v[s-1] (a real threshold
    /// exists between the two sorted values).
    /// </summary>
    public static Tensor<T> AdjacentIncreaseMask<T>(Vector<T> sortedValues)
    {
        int n = sortedValues.Length;
        return AiDotNetEngine.Current.TensorGreaterThan(
            Tensor<T>.FromVector(sortedValues.GetSubVector(1, n - 1)),
            Tensor<T>.FromVector(sortedValues.GetSubVector(0, n - 1)));
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
    /// All row differences left[i] - right[j] as a [left.Rows, right.Rows, features] tensor.
    /// </summary>
    public static Tensor<T> PairwiseDifferences<T>(Matrix<T> left, Matrix<T> right)
    {
        var engine = AiDotNetEngine.Current;
        int m = left.Rows;
        int n = right.Rows;
        int d = left.Columns;
        var shape = new[] { m, n, d };
        var l = engine.TensorBroadcastTo(engine.Reshape(Tensor<T>.FromMatrix(left), new[] { m, 1, d }), shape);
        var r = engine.TensorBroadcastTo(engine.Reshape(Tensor<T>.FromMatrix(right), new[] { 1, n, d }), shape);
        return engine.TensorSubtract(l, r);
    }

    /// <summary>
    /// Squared Euclidean distances ||left[i] - right[j]||^2 as a [left.Rows, right.Rows] matrix,
    /// summed in feature order. Left rows are processed in blocks so the intermediate difference
    /// tensor stays bounded.
    /// </summary>
    public static Matrix<T> PairwiseSquaredDistances<T>(Matrix<T> left, Matrix<T> right)
    {
        if (left.Rows == 0 || right.Rows == 0)
        {
            return new Matrix<T>(left.Rows, right.Rows);
        }

        var engine = AiDotNetEngine.Current;
        long perRow = Math.Max(1L, (long)right.Rows * Math.Max(1, left.Columns));
        int blockRows = (int)Math.Max(1L, Math.Min(left.Rows, PairwiseBlockElements / perRow));
        var blocks = new List<Tensor<T>>();
        for (int start = 0; start < left.Rows; start += blockRows)
        {
            int count = Math.Min(blockRows, left.Rows - start);
            var block = count == left.Rows ? left : left.GetSubMatrix(start, 0, count, left.Columns);
            blocks.Add(engine.ReduceSum(
                engine.TensorSquare(PairwiseDifferences(block, right)), new[] { 2 }, keepDims: false));
        }

        var result = blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), 0);
        return result.ToMatrix();
    }

    private const long PairwiseBlockElements = 1L << 24;

    /// <summary>
    /// All row dot products left[i] . right[j] as a [left.Rows, right.Rows] tensor.
    /// </summary>
    public static Tensor<T> CrossDotProducts<T>(Matrix<T> left, Matrix<T> right)
    {
        return AiDotNetEngine.Current.TensorMatMul(
            Tensor<T>.FromMatrix(left), Tensor<T>.FromMatrix(right.Transpose()));
    }

    /// <summary>
    /// Copies <paramref name="matrix"/> into a vector in row-major order (bulk copy, no arithmetic).
    /// </summary>
    public static Vector<T> FlattenRowMajor<T>(Matrix<T> matrix)
    {
        var flat = new Vector<T>(matrix.Rows * matrix.Columns);
        matrix.AsSpan().CopyTo(flat.AsWritableSpan());
        return flat;
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
