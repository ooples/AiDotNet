namespace AiDotNet.Geometry;

/// <summary>
/// Whole-array mesh and point-cloud primitives shared by the geometry types, expressed as engine operations.
/// </summary>
/// <typeparam name="T">The numeric type of the coordinates.</typeparam>
internal static class GeometryEngineOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Returns one column of a [rows, columns] tensor as a rank-1 tensor.
    /// </summary>
    internal static Tensor<TValue> Column<TValue>(Tensor<TValue> table, int column)
    {
        int rows = table.Shape[0];
        return Engine.TensorSlice(table, new[] { 0, column }, new[] { rows, 1 }).Reshape(rows);
    }

    /// <summary>
    /// Returns the given corner of every face as a [faces, 3] tensor of positions.
    /// </summary>
    internal static Tensor<T> FaceCorner(Tensor<T> vertices, Tensor<int> faces, int corner)
    {
        return Engine.TensorIndexSelect(vertices, Column(faces, corner), 0);
    }

    /// <summary>
    /// Returns the unnormalized face normals (v1 - v0) x (v2 - v0) as a [faces, 3] tensor.
    /// </summary>
    internal static Tensor<T> FaceCrossProducts(Tensor<T> vertices, Tensor<int> faces)
    {
        var p0 = FaceCorner(vertices, faces, 0);
        var edge1 = Engine.TensorSubtract(FaceCorner(vertices, faces, 1), p0);
        var edge2 = Engine.TensorSubtract(FaceCorner(vertices, faces, 2), p0);
        return Engine.TensorCross(edge1, edge2, 1);
    }

    /// <summary>
    /// Returns the squared Euclidean length of every row of a [rows, 3] tensor as a [rows, 1] tensor.
    /// </summary>
    internal static Tensor<T> RowSquaredNorms(Tensor<T> rows)
    {
        return Engine.ReduceSum(Engine.TensorSquare(rows), new[] { 1 }, true);
    }

    /// <summary>
    /// Multiplies every row of a [rows, columns] tensor by the matching entry of a [rows, 1] tensor.
    /// </summary>
    internal static Tensor<T> ScaleRows(Tensor<T> rows, Tensor<T> factors)
    {
        return Engine.TensorMultiply(rows, Engine.TensorBroadcastTo(factors, rows.Shape.ToArray()));
    }

    /// <summary>
    /// Scales every row to unit length; rows whose squared length is not positive (zero or NaN) become zero.
    /// </summary>
    /// <remarks>
    /// The division uses a safe denominator of one on the rows that are zeroed, so decimal never divides by zero,
    /// and the rows are divided directly by their length (no reciprocal), so integral types keep their whole-number
    /// quotient. Half is normalized in float: its squared lengths overflow above 255.
    /// </remarks>
    internal static Tensor<T> NormalizeRowsOrZero(Tensor<T> rows)
    {
        if (rows.Shape[0] == 0)
        {
            return rows;
        }

        if (typeof(T) == typeof(Half))
        {
            return GeometryEngineOps<float>.NormalizeRowsOrZero(rows.Cast<float>()).Cast<T>();
        }

        var shape = rows.Shape.ToArray();
        var lengthSquared = RowSquaredNorms(rows);
        var positive = Engine.TensorGreaterThan(lengthSquared, NumOps.Zero);
        var safeLength = Engine.TensorWhere(positive, Engine.TensorSqrt(lengthSquared), Engine.TensorOnesLike(lengthSquared));
        var quotient = Engine.TensorDivide(rows, Engine.TensorBroadcastTo(safeLength, shape));
        return Engine.TensorWhere(Engine.TensorBroadcastTo(positive, shape), quotient, Engine.TensorZerosLike(rows));
    }

    /// <summary>
    /// Divides every row by its length when that length exceeds <paramref name="minLength"/>; shorter rows are
    /// returned unchanged.
    /// </summary>
    internal static Tensor<T> NormalizeRowsAbove(Tensor<T> rows, T minLength)
    {
        if (rows.Shape[0] == 0)
        {
            return rows;
        }

        if (typeof(T) == typeof(Half))
        {
            return GeometryEngineOps<float>.NormalizeRowsAbove(rows.Cast<float>(), NumOps.ToFloat(minLength)).Cast<T>();
        }

        var length = Engine.TensorBroadcastTo(Engine.TensorSqrt(RowSquaredNorms(rows)), rows.Shape.ToArray());
        var usable = Engine.TensorGreaterThan(length, minLength);
        var safeLength = Engine.TensorWhere(usable, length, Engine.TensorOnesLike(length));
        return Engine.TensorWhere(usable, Engine.TensorDivide(rows, safeLength), rows);
    }

    /// <summary>
    /// Sums each face's row of <paramref name="faceValues"/> into its three vertices, in face then corner order.
    /// </summary>
    internal static Tensor<T> AccumulateToVertices(Tensor<T> faceValues, Tensor<int> faces, int numVertices)
    {
        int numFaces = faces.Shape[0];
        int width = faceValues.Shape[1];
        var sums = new Tensor<T>(new[] { numVertices, width });
        if (numFaces == 0)
        {
            return sums;
        }

        var cornerVertices = faces.Reshape(numFaces * 3);
        var cornerValues = Engine.TensorRepeatInterleave(faceValues, 3, 0);
        return Engine.TensorIndexAdd(sums, 0, cornerVertices, cornerValues);
    }

    /// <summary>
    /// Returns the per-column minimum and maximum of a [rows, columns] tensor, ignoring NaN entries. The results
    /// start from <c>MaxValue</c> and <c>MinValue</c>, as a running comparison would, so they never exceed the
    /// type's finite range and an all-NaN column reports those start values.
    /// </summary>
    internal static (Tensor<T> min, Tensor<T> max) ColumnBounds(Tensor<T> rows)
    {
        var isNan = Engine.TensorIsNan(rows);
        var forMin = Engine.TensorWhere(isNan, Engine.TensorFullLike(rows, NumOps.MaxValue), rows);
        var forMax = Engine.TensorWhere(isNan, Engine.TensorFullLike(rows, NumOps.MinValue), rows);
        var min = Engine.TensorNegate(Engine.ReduceMax(Engine.TensorNegate(forMin), new[] { 0 }, false));
        var max = Engine.ReduceMax(forMax, new[] { 0 }, false);
        return (Engine.TensorMin(min, NumOps.MaxValue), Engine.TensorMax(max, NumOps.MinValue));
    }

    /// <summary>
    /// Returns a 1/0 mask over the columns of a [rows, columns] tensor marking the columns that contain NaN.
    /// </summary>
    internal static Tensor<T> ColumnHasNan(Tensor<T> rows)
    {
        var isNan = Engine.TensorIsNan(rows);
        var flags = Engine.TensorWhere(isNan, Engine.TensorOnesLike(rows), Engine.TensorZerosLike(rows));
        return Engine.TensorGreaterThan(Engine.ReduceMax(flags, new[] { 0 }, false), NumOps.Zero);
    }

    /// <summary>
    /// Returns the xyz columns of a [points, features] tensor whose first three features are the position.
    /// </summary>
    internal static Tensor<T> Positions(Tensor<T> points)
    {
        int count = points.Shape[0];
        return points.Shape[1] == 3 ? points : Engine.TensorSlice(points, new[] { 0, 0 }, new[] { count, 3 });
    }

    /// <summary>
    /// Returns the [from, to] matrix of squared Euclidean distances between the xyz positions of two point sets,
    /// computed from the coordinate differences (not the less accurate |a|^2 + |b|^2 - 2ab expansion).
    /// </summary>
    internal static Tensor<T> PairwiseSquaredDistances(Tensor<T> from, Tensor<T> to)
    {
        int fromCount = from.Shape[0];
        int toCount = to.Shape[0];
        var shape = new[] { fromCount, toCount };
        var fromPositions = Positions(from);
        var toPositions = Positions(to);

        // Accumulated one axis at a time in x, y, z order (the order the reduction over the last axis used), so
        // only [from, to] tensors are ever allocated rather than a [from, to, 3] difference block.
        Tensor<T>? total = null;
        for (int axis = 0; axis < 3; axis++)
        {
            var left = Engine.TensorBroadcastTo(Column(fromPositions, axis).Reshape(fromCount, 1), shape);
            var right = Engine.TensorBroadcastTo(Column(toPositions, axis).Reshape(1, toCount), shape);
            var squared = Engine.TensorSquare(Engine.TensorSubtract(left, right));
            total = total is null ? squared : Engine.TensorAdd(total, squared);
        }

        return total ?? new Tensor<T>(shape);
    }

    /// <summary>The most distance-matrix entries one block of rows may hold (about 32 MB of double).</summary>
    internal const int DistanceBlockElements = 1 << 22;

    /// <summary>
    /// Yields the [from, to] squared-distance matrix in blocks of consecutive <paramref name="from"/> rows, each
    /// with its first row index, so callers that only reduce over rows or columns never hold the whole matrix.
    /// </summary>
    internal static IEnumerable<(int Start, Tensor<T> Distances)> SquaredDistanceBlocks(Tensor<T> from, Tensor<T> to)
    {
        int fromCount = from.Shape[0];
        int toCount = to.Shape[0];
        int features = from.Shape[1];
        int block = Math.Max(1, DistanceBlockElements / Math.Max(1, toCount));
        for (int start = 0; start < fromCount; start += block)
        {
            int count = Math.Min(block, fromCount - start);
            var rows = count == fromCount
                ? from
                : Engine.TensorSlice(from, new[] { start, 0 }, new[] { count, features });
            yield return (start, PairwiseSquaredDistances(rows, to));
        }
    }

    /// <summary>
    /// Returns, for every <paramref name="from"/> point and every <paramref name="to"/> point, the smallest squared
    /// distance to the other set, skipping NaN distances as <see cref="RowMinSkippingNan"/> does (an all-NaN row or
    /// column reports <c>MaxValue</c>). The distance matrix is processed in row blocks.
    /// </summary>
    internal static (Tensor<T> FromMin, Tensor<T> ToMin) NearestSquaredDistances(Tensor<T> from, Tensor<T> to)
    {
        var fromMins = new List<Tensor<T>>();
        Tensor<T>? toMin = null;
        foreach (var (_, distances) in SquaredDistanceBlocks(from, to))
        {
            fromMins.Add(RowMinSkippingNan(distances));
            var blockMin = RowMinSkippingNan(Engine.TensorTranspose(distances));
            toMin = toMin is null ? blockMin : Engine.TensorWhere(Engine.TensorLessThan(blockMin, toMin), blockMin, toMin);
        }

        var fromMin = fromMins.Count == 1 ? fromMins[0] : Engine.TensorConcatenate(fromMins.ToArray(), 0);
        return (fromMin, toMin ?? new Tensor<T>(new[] { to.Shape[0] }));
    }

    /// <summary>
    /// Returns the minimum of every row of a [rows, columns] tensor the way a running "value &lt; best" scan
    /// starting from <c>MaxValue</c> finds it: NaN entries never win and an all-NaN row reports <c>MaxValue</c>.
    /// </summary>
    internal static Tensor<T> RowMinSkippingNan(Tensor<T> table)
    {
        var masked = Engine.TensorWhere(Engine.TensorIsNan(table), Engine.TensorFullLike(table, NumOps.MaxValue), table);
        var min = Engine.TensorNegate(Engine.ReduceMax(Engine.TensorNegate(masked), new[] { 1 }, false));
        return Engine.TensorMin(min, NumOps.MaxValue);
    }

    /// <summary>
    /// Returns order-preserving integer keys for every entry of a [rows, columns] tensor, row-major: within a row
    /// equal values share a key and smaller values get smaller keys; NaN entries get -1. Comparing the keys gives
    /// the same outcome as comparing the values, so scalar selection and sort logic can run on integers.
    /// </summary>
    internal static int[] DenseRowRanks(Tensor<T> table)
    {
        int rows = table.Shape[0];
        int columns = table.Shape[1];
        var ranks = new int[rows * columns];
        if (rows == 0 || columns == 0)
        {
            return ranks;
        }

        var (sorted, order) = Engine.TensorSort(table, 1, false);
        var opensGroup = Engine.TensorOnesLike(Engine.TensorSlice(sorted, new[] { 0, 0 }, new[] { rows, 1 }));
        if (columns > 1)
        {
            var right = Engine.TensorSlice(sorted, new[] { 0, 1 }, new[] { rows, columns - 1 });
            var left = Engine.TensorSlice(sorted, new[] { 0, 0 }, new[] { rows, columns - 1 });
            opensGroup = Engine.TensorConcatenate(new[] { opensGroup, Engine.TensorNotEquals(right, left) }, 1);
        }

        var isStart = new bool[rows * columns];
        var starts = Engine.TensorNonzero(opensGroup).ToArray();
        for (int s = 0; s + 1 < starts.Length; s += 2)
        {
            isStart[starts[s] * columns + starts[s + 1]] = true;
        }

        var positions = order.ToArray();
        for (int row = 0; row < rows; row++)
        {
            int rank = -1;
            int rowOffset = row * columns;
            for (int p = 0; p < columns; p++)
            {
                if (isStart[rowOffset + p])
                {
                    rank++;
                }

                ranks[rowOffset + positions[rowOffset + p]] = rank;
            }
        }

        var nanFlags = Engine.TensorWhere(Engine.TensorIsNan(table), Engine.TensorOnesLike(table), Engine.TensorZerosLike(table));
        var nanCoordinates = Engine.TensorNonzero(nanFlags).ToArray();
        for (int c = 0; c + 1 < nanCoordinates.Length; c += 2)
        {
            ranks[nanCoordinates[c] * columns + nanCoordinates[c + 1]] = -1;
        }

        return ranks;
    }

    /// <summary>
    /// Returns the (row, column) pairs, row-major, where <paramref name="mask"/> is nonzero, as a flat int array.
    /// </summary>
    internal static int[] NonzeroPairs(Tensor<T> mask)
    {
        return Engine.TensorNonzero(mask).ToArray();
    }

    /// <summary>
    /// Returns <paramref name="points"/> with its xyz columns replaced by <paramref name="positions"/>; any further
    /// feature columns are kept unchanged.
    /// </summary>
    internal static Tensor<T> WithPositions(Tensor<T> points, Tensor<T> positions)
    {
        int count = points.Shape[0];
        int features = points.Shape[1];
        if (features == 3)
        {
            return positions;
        }

        var rest = Engine.TensorSlice(points, new[] { 0, 3 }, new[] { count, features - 3 });
        return Engine.TensorConcatenate(new[] { positions, rest }, 1);
    }

    /// <summary>
    /// Returns the per-column minimum and maximum the way a running comparison seeded with the first row finds
    /// them: later NaN entries are skipped, and a column whose first entry is NaN reports NaN.
    /// </summary>
    internal static (Tensor<T> min, Tensor<T> max) RunningColumnBounds(Tensor<T> rows)
    {
        int columns = rows.Shape[1];
        var first = Engine.TensorSlice(rows, new[] { 0, 0 }, new[] { 1, columns });
        var masked = Engine.TensorWhere(Engine.TensorIsNan(rows), Engine.TensorBroadcastTo(first, rows.Shape.ToArray()), rows);
        var min = Engine.TensorNegate(Engine.ReduceMax(Engine.TensorNegate(masked), new[] { 0 }, false));
        var max = Engine.ReduceMax(masked, new[] { 0 }, false);
        var firstRow = first.Reshape(columns);
        var firstIsNan = Engine.TensorIsNan(firstRow);
        return (Engine.TensorWhere(firstIsNan, firstRow, min), Engine.TensorWhere(firstIsNan, firstRow, max));
    }

    /// <summary>
    /// Returns true when any entry of <paramref name="values"/> is NaN.
    /// </summary>
    internal static bool AnyNan(Tensor<T> values)
    {
        var flags = Engine.TensorWhere(Engine.TensorIsNan(values), Engine.TensorOnesLike(values), Engine.TensorZerosLike(values));
        return Engine.TensorCountNonzero(flags) > 0;
    }

    /// <summary>
    /// Returns the per-column minimum and maximum the way Math.Min/Math.Max folds seeded with
    /// <c>MaxValue</c>/<c>MinValue</c> find them: a column containing NaN reports NaN for both bounds.
    /// </summary>
    internal static (Tensor<T> min, Tensor<T> max) PropagatingColumnBounds(Tensor<T> rows)
    {
        var (min, max) = ColumnBounds(rows);
        // Only a type that can hold NaN reaches the replacement; decimal cannot convert NaN at all.
        if (!AnyNan(rows))
        {
            return (min, max);
        }

        var hasNan = ColumnHasNan(rows);
        var nan = Engine.TensorFullLike(min, NumOps.FromDouble(double.NaN));
        return (Engine.TensorWhere(hasNan, nan, min), Engine.TensorWhere(hasNan, nan, max));
    }

    /// <summary>
    /// Returns the largest entry the way a Math.Max fold does: NaN when any entry is NaN.
    /// </summary>
    internal static T PropagatingMax(Tensor<T> values)
    {
        return AnyNan(values) ? NumOps.FromDouble(double.NaN) : Engine.TensorMaxValue(values);
    }
}
