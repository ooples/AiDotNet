using AiDotNet.Clustering.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Clustering.Base;

/// <summary>
/// Whole-array engine building blocks shared by the clustering algorithms, metrics and evaluators.
/// </summary>
/// <remarks>
/// <para>
/// Every helper expresses its math as <see cref="IEngine"/> operations over whole matrices instead of per-element
/// loops. Integer label data (cluster ids, noise = -1) is stored by the clustering API as integral values of
/// <typeparamref name="T"/>; the helpers convert between that representation and <see cref="int"/> indices with
/// bulk search/gather operations.
/// </para>
/// </remarks>
internal static class ClusteringTensorOps
{
    /// <summary>Upper bound on the number of elements materialized per block by the pairwise helpers.</summary>
    private const int PairBlockElements = 1 << 22;

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Copies a matrix into a rank-2 tensor.</summary>
    internal static Tensor<T> ToTensor<T>(Matrix<T> matrix) => Tensor<T>.FromMatrix(matrix);

    /// <summary>Squared Euclidean distance between the paired rows of two [p, d] tensors, returned as [p].</summary>
    internal static Tensor<T> RowSquaredEuclidean<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = Engine;
        return engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(a, b)), new[] { 1 }, false);
    }

    /// <summary>Euclidean distance between the paired rows of two [p, d] tensors, returned as [p].</summary>
    internal static Tensor<T> RowEuclidean<T>(Tensor<T> a, Tensor<T> b)
        => Engine.TensorSqrt(RowSquaredEuclidean(a, b));

    /// <summary>Manhattan (L1) distance between the paired rows of two [p, d] tensors, returned as [p].</summary>
    internal static Tensor<T> RowManhattan<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = Engine;
        return engine.ReduceSum(engine.TensorAbs(engine.TensorSubtract(a, b)), new[] { 1 }, false);
    }

    /// <summary>Chebyshev (L-infinity) distance between the paired rows of two [p, d] tensors, returned as [p].</summary>
    internal static Tensor<T> RowChebyshev<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a.Shape[1] == 0)
        {
            return Tensor<T>.CreateZeros(new[] { a.Shape[0] });
        }

        var engine = Engine;
        return engine.ReduceMax(engine.TensorAbs(engine.TensorSubtract(a, b)), new[] { 1 }, false);
    }

    /// <summary>
    /// Applies a paired-row reducer to every (row of <paramref name="x"/>, row of <paramref name="y"/>) pair and
    /// returns the [x.Rows, y.Rows] result. Rows of x are processed in blocks so the replicated operands stay bounded.
    /// </summary>
    internal static Matrix<T> PairwiseRowReduce<T>(Matrix<T> x, Matrix<T> y, Func<Tensor<T>, Tensor<T>, Tensor<T>> rowPairs)
    {
        int n = x.Rows;
        int m = y.Rows;
        int d = x.Columns;
        if (n == 0 || m == 0)
        {
            return new Matrix<T>(n, m);
        }

        var engine = Engine;
        var xt = ToTensor(x);
        var yt = ToTensor(y);
        long perRow = (long)m * Math.Max(1, d);
        int rowsPerBlock = (int)Math.Max(1, Math.Min(n, PairBlockElements / perRow));
        if (rowsPerBlock >= n)
        {
            var a = engine.TensorRepeatInterleave(xt, m, 0);
            var b = engine.TensorTile(yt, new[] { n, 1 });
            return engine.Reshape(rowPairs(a, b), new[] { n, m }).ToMatrix();
        }

        var result = Tensor<T>.CreateZeros(new[] { n, m });
        for (int start = 0; start < n; start += rowsPerBlock)
        {
            int count = Math.Min(rowsPerBlock, n - start);
            var block = engine.TensorSlice(xt, new[] { start, 0 }, new[] { count, d });
            var a = engine.TensorRepeatInterleave(block, m, 0);
            var b = engine.TensorTile(yt, new[] { count, 1 });
            var values = engine.Reshape(rowPairs(a, b), new[] { count, m });
            result = engine.TensorSetSlice(result, values, new[] { start, 0 });
        }

        return result.ToMatrix();
    }

    /// <summary>Applies a paired-row reducer between one point and every row of <paramref name="data"/>.</summary>
    internal static Vector<T> PointRowReduce<T>(Vector<T> point, Matrix<T> data, Func<Tensor<T>, Tensor<T>, Tensor<T>> rowPairs)
    {
        int n = data.Rows;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        var tiled = Engine.TensorTile(Tensor<T>.FromVector(point, new[] { 1, point.Length }), new[] { n, 1 });
        return rowPairs(tiled, ToTensor(data)).ToVector();
    }

    /// <summary>Exact squared Euclidean distances between every row of x and every row of y, as [x.Rows, y.Rows].</summary>
    internal static Matrix<T> SquaredDistances<T>(Matrix<T> x, Matrix<T> y) => PairwiseRowReduce(x, y, RowSquaredEuclidean<T>);

    /// <summary>Exact Euclidean distances between every row of x and every row of y, as [x.Rows, y.Rows].</summary>
    internal static Matrix<T> Distances<T>(Matrix<T> x, Matrix<T> y) => PairwiseRowReduce(x, y, RowEuclidean<T>);

    /// <summary>Squared Euclidean distance from one point to every row of data.</summary>
    internal static Vector<T> SquaredDistancesToPoint<T>(Vector<T> point, Matrix<T> data)
        => PointRowReduce(point, data, RowSquaredEuclidean<T>);

    /// <summary>Euclidean distance from one point to every row of data.</summary>
    internal static Vector<T> DistancesToPoint<T>(Vector<T> point, Matrix<T> data)
        => PointRowReduce(point, data, RowEuclidean<T>);

    /// <summary>Index of the smallest entry in every row (first one on ties).</summary>
    internal static int[] ArgMinRows<T>(Matrix<T> values)
    {
        if (values.Rows == 0)
        {
            return Array.Empty<int>();
        }

        return Engine.TensorArgMin(ToTensor(values), 1).ToArray();
    }

    /// <summary>Index of the smallest entry of a non-empty vector (first one on ties).</summary>
    internal static int ArgMin<T>(Vector<T> values) => Engine.TensorArgMin(values.AsTensor(), 0).ToArray()[0];

    /// <summary>Index of the largest entry of a non-empty vector (first one on ties).</summary>
    internal static int ArgMax<T>(Vector<T> values) => Engine.TensorArgMax(values.AsTensor(), 0).ToArray()[0];

    /// <summary>Largest entry of the vector, or zero when it is empty or every entry is negative.</summary>
    internal static T MaxOrZero<T>(Vector<T> values)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (values.Length == 0)
        {
            return numOps.Zero;
        }

        T max = Engine.TensorMaxValue(values.AsTensor());
        return numOps.GreaterThan(max, numOps.Zero) ? max : numOps.Zero;
    }

    /// <summary>Smallest entry of every row.</summary>
    internal static Vector<T> MinRows<T>(Matrix<T> values)
    {
        if (values.Rows == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorAmin(ToTensor(values), new[] { 1 }, false).ToVector();
    }

    /// <summary>
    /// Converts integral values stored as <typeparamref name="T"/> (labels, counts, flags) to <see cref="int"/>
    /// with one bulk search against the integer range they span.
    /// </summary>
    internal static int[] ToIntArray<T>(Vector<T> integralValues)
    {
        int n = integralValues.Length;
        if (n == 0)
        {
            return Array.Empty<int>();
        }

        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var values = integralValues.AsTensor();
        int min = (int)Math.Floor(numOps.ToDouble(engine.TensorMinValue(values)));
        int max = (int)Math.Ceiling(numOps.ToDouble(engine.TensorMaxValue(values)));
        var table = engine.TensorArange<T>(min, (double)max + 1, 1);
        // With right = true the position of an integral value v is (v - min + 1); fractional values round down.
        var positions = engine.TensorSearchSorted(table, values, true);
        return (min == 1 ? positions : engine.TensorAddScalar(positions, min - 1)).ToArray();
    }

    /// <summary>Converts integer values (labels, indices) to a <typeparamref name="T"/> vector with one bulk gather.</summary>
    internal static Vector<T> FromIntArray<T>(int[] values)
    {
        if (values.Length == 0)
        {
            return new Vector<T>(0);
        }

        var engine = Engine;
        var indices = Vector<int>.Wrap(values).AsTensor();
        int min = engine.TensorMinValue(indices);
        int max = engine.TensorMaxValue(indices);
        var table = engine.TensorArange<T>(min, (double)max + 1, 1);
        var offsets = min == 0 ? indices : engine.TensorSubtractScalar(indices, min);
        return engine.TensorGather(table, offsets, 0).ToVector();
    }

    /// <summary>
    /// One-hot membership mask [n, k] in <typeparamref name="T"/>: entry (i, c) is one when labels[i] == c.
    /// Labels outside [0, k) (noise, unassigned) produce an all-zero row.
    /// </summary>
    internal static Tensor<T> OneHotMask<T>(Vector<T> labels, int k)
    {
        int n = labels.Length;
        var engine = Engine;
        var column = engine.TensorBroadcastTo(Tensor<T>.FromVector(labels, new[] { n, 1 }), new[] { n, k });
        var ids = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, k, 1), new[] { 1, k }), new[] { n, k });
        return engine.TensorEquals(column, ids);
    }

    /// <summary>Number of samples carrying each label 0..k-1 ([k], in T).</summary>
    internal static Vector<T> ClusterCounts<T>(Vector<T> labels, int k)
        => Engine.ReduceSum(OneHotMask(labels, k), new[] { 0 }, false).ToVector();

    /// <summary>Per-cluster sums of the rows of x [k, d] and member counts [k].</summary>
    internal static Matrix<T> ClusterSums<T>(Matrix<T> x, Vector<T> labels, int k, out Vector<T> counts)
    {
        var engine = Engine;
        var mask = OneHotMask(labels, k);
        counts = engine.ReduceSum(mask, new[] { 0 }, false).ToVector();
        return engine.TensorMatMul(engine.TensorTranspose(mask), ToTensor(x)).ToMatrix();
    }

    /// <summary>
    /// Per-cluster means of the rows of x [k, d]; clusters without members get a zero row. Also returns the counts.
    /// </summary>
    internal static Matrix<T> ClusterMeans<T>(Matrix<T> x, Vector<T> labels, int k, out Vector<T> counts)
    {
        var sums = ClusterSums(x, labels, k, out counts);
        return DivideRows(sums, counts);
    }

    /// <summary>Divides every row of m by the matching count, treating counts below one as one.</summary>
    internal static Matrix<T> DivideRows<T>(Matrix<T> m, Vector<T> counts)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var safe = engine.TensorMax(counts.AsTensor(), numOps.One);
        var divisor = engine.TensorBroadcastTo(engine.Reshape(safe, new[] { m.Rows, 1 }), new[] { m.Rows, m.Columns });
        return engine.TensorDivide(ToTensor(m), divisor).ToMatrix();
    }

    /// <summary>Gathers the row of <paramref name="centers"/> assigned to every sample ([n, d]); invalid labels give zero rows.</summary>
    internal static Tensor<T> AssignedRows<T>(Vector<T> labels, Matrix<T> centers, out Tensor<T> valid)
    {
        var engine = Engine;
        var mask = OneHotMask(labels, centers.Rows);
        valid = engine.ReduceSum(mask, new[] { 1 }, false);
        return engine.TensorMatMul(mask, ToTensor(centers));
    }

    /// <summary>Squared distance of every sample to its assigned center; zero for invalid labels.</summary>
    internal static Vector<T> AssignedSquaredDistances<T>(Matrix<T> x, Vector<T> labels, Matrix<T> centers)
    {
        var engine = Engine;
        var assigned = AssignedRows(labels, centers, out var valid);
        return engine.TensorMultiply(RowSquaredEuclidean(ToTensor(x), assigned), valid).ToVector();
    }

    /// <summary>Within-cluster sum of squared distances to the assigned centers (labels outside the centers are skipped).</summary>
    internal static T Inertia<T>(Matrix<T> x, Vector<T> labels, Matrix<T> centers)
    {
        if (x.Rows == 0 || centers.Rows == 0)
        {
            return MathHelper.GetNumericOperations<T>().Zero;
        }

        return Engine.TensorSum(AssignedSquaredDistances(x, labels, centers).AsTensor());
    }

    /// <summary>Row-major flattening of a matrix.</summary>
    internal static Vector<T> Flatten<T>(Matrix<T> m) => ToTensor(m).ToVector();

    /// <summary>Builds a [rows, columns] matrix from a row-major vector.</summary>
    internal static Matrix<T> Unflatten<T>(Vector<T> values, int rows, int columns)
        => Tensor<T>.FromVector(values, new[] { rows, columns }).ToMatrix();

    /// <summary>Selects rows of a matrix by index.</summary>
    internal static Matrix<T> SelectRows<T>(Matrix<T> m, int[] rows)
    {
        if (rows.Length == 0)
        {
            return new Matrix<T>(0, m.Columns);
        }

        return Engine.TensorIndexSelect(ToTensor(m), Vector<int>.Wrap(rows).AsTensor(), 0).ToMatrix();
    }

    /// <summary>Selects columns of a rank-2 tensor by index.</summary>
    internal static Tensor<T> SelectColumns<T>(Tensor<T> m, int[] columns)
        => Engine.TensorIndexSelect(m, Vector<int>.Wrap(columns).AsTensor(), 1);

    /// <summary>Selects elements of a vector by index.</summary>
    internal static Vector<T> SelectElements<T>(Vector<T> v, int[] indices)
    {
        if (indices.Length == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorGather(v.AsTensor(), Vector<int>.Wrap(indices).AsTensor(), 0).ToVector();
    }

    /// <summary>Column means of a matrix ([d]).</summary>
    internal static Vector<T> ColumnMeans<T>(Matrix<T> m) => Engine.ReduceMean(ToTensor(m), new[] { 0 }, false).ToVector();

    /// <summary>Subtracts a row vector from every row of a matrix.</summary>
    internal static Tensor<T> SubtractRow<T>(Matrix<T> m, Vector<T> row)
    {
        var engine = Engine;
        var tiled = engine.TensorTile(Tensor<T>.FromVector(row, new[] { 1, row.Length }), new[] { m.Rows, 1 });
        return engine.TensorSubtract(ToTensor(m), tiled);
    }

    /// <summary>Flags the entries of v that are below the threshold.</summary>
    internal static bool[] LessThanFlags<T>(Vector<T> v, T threshold)
        => ToBoolArray(Engine.TensorLessThan(v.AsTensor(), threshold).ToVector());

    /// <summary>Flags the entries of v that are at most the threshold.</summary>
    internal static bool[] LessOrEqualFlags<T>(Vector<T> v, T threshold)
        => ToBoolArray(Engine.TensorGreaterThan(v.AsTensor(), threshold).ToVector(), invert: true);

    /// <summary>Converts a 0/1 vector in T to bool flags.</summary>
    internal static bool[] ToBoolArray<T>(Vector<T> zeroOne, bool invert = false)
    {
        var ints = ToIntArray(zeroOne);
        var flags = new bool[ints.Length];
        for (int i = 0; i < ints.Length; i++)
        {
            flags[i] = (ints[i] != 0) != invert;
        }

        return flags;
    }

    /// <summary>True when any entry of the matrix is NaN or infinite.</summary>
    internal static bool ContainsNonFinite<T>(Matrix<T> m)
    {
        if (m.Rows == 0 || m.Columns == 0)
        {
            return false;
        }

        var engine = Engine;
        var t = ToTensor(m);
        return engine.TensorMaskedSelect(t, engine.TensorLogicalNot(engine.TensorIsFinite(t))).Length > 0;
    }

    /// <summary>
    /// Draws an index with probability proportional to the (non-negative) weights: the first index whose running
    /// sum reaches u * total for u = random.NextDouble(), or 0 when none does (same rule as a sequential scan).
    /// </summary>
    internal static int SampleProportional<T>(Vector<T> weights, Random random)
    {
        int position = SampleProportionalOrNone(weights, random);
        return position >= 0 ? position : 0;
    }

    /// <summary>As <see cref="SampleProportional"/>, but returns -1 when no running sum reaches the threshold.</summary>
    internal static int SampleProportionalOrNone<T>(Vector<T> weights, Random random)
    {
        if (weights.Length == 0)
        {
            random.NextDouble();
            return -1;
        }

        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var cumulative = engine.TensorCumSum(weights.AsTensor(), 0);

        // The weights are non-negative, so the running sum ends at its maximum.
        T total = engine.TensorMaxValue(cumulative);
        T threshold = numOps.Multiply(numOps.FromDouble(random.NextDouble()), total);
        int position = engine.TensorSearchSorted(cumulative, Vector<T>.FromArray(new[] { threshold }).AsTensor(), false).ToArray()[0];
        return position < weights.Length ? position : -1;
    }

    /// <summary>Distance (configured metric) from every row of x to one point, evaluated as metric(row, point).</summary>
    internal static Vector<T> DistancesToPoint<T>(IDistanceMetric<T> metric, Matrix<T> x, Vector<T> point)
        => Tensor<T>.FromMatrix(metric.ComputePairwise(x, Tensor<T>.FromVector(point, new[] { 1, point.Length }).ToMatrix())).ToVector();

    /// <summary>Distance (configured metric) between row k of a and row k of b, for every k.</summary>
    internal static Vector<T> RowwiseDistances<T>(IDistanceMetric<T> metric, Matrix<T> a, Matrix<T> b)
        => Engine.TensorDiagonal(Tensor<T>.FromMatrix(metric.ComputePairwise(a, b))).ToVector();

    /// <summary>Nearest-center label (first one on ties) for every row of x under the configured metric.</summary>
    internal static int[] NearestCenters<T>(IDistanceMetric<T> metric, Matrix<T> x, Matrix<T> centers)
        => ArgMinRows(metric.ComputePairwise(x, centers));

    /// <summary>Distance (configured metric) from every sample to the center of its label; zero for invalid labels.</summary>
    internal static Vector<T> AssignedDistances<T>(IDistanceMetric<T> metric, Matrix<T> x, Vector<T> labels, Matrix<T> centers)
    {
        var engine = Engine;
        var mask = OneHotMask(labels, centers.Rows);
        var distances = Tensor<T>.FromMatrix(metric.ComputePairwise(x, centers));
        return engine.ReduceSum(engine.TensorMultiply(distances, mask), new[] { 1 }, false).ToVector();
    }

    /// <summary>Rows of <paramref name="candidate"/> where the count is positive, rows of <paramref name="fallback"/> elsewhere.</summary>
    internal static Matrix<T> RowsWherePositive<T>(Vector<T> counts, Matrix<T> candidate, Matrix<T> fallback)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var positive = engine.TensorGreaterThan(counts.AsTensor(), numOps.Zero);
        var condition = engine.TensorBroadcastTo(engine.Reshape(positive, new[] { candidate.Rows, 1 }), new[] { candidate.Rows, candidate.Columns });
        return engine.TensorWhere(condition, ToTensor(candidate), ToTensor(fallback)).ToMatrix();
    }

    /// <summary>
    /// Column means and population standard deviations of x (deviations below 1e-10 become 1), and x standardized with them.
    /// </summary>
    internal static Matrix<T> StandardizeColumns<T>(Matrix<T> x, out Vector<T> means, out Vector<T> stds)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        means = ColumnMeans(x);
        var centered = SubtractRow(x, means);
        var std = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(centered), new[] { 0 }, false));
        var tiny = engine.TensorLessThan(std, numOps.FromDouble(1e-10));
        stds = engine.TensorWhere(tiny, engine.TensorOnesLike(std), std).ToVector();
        return engine.TensorDivide(centered, TileRow(stds, x.Rows)).ToMatrix();
    }

    /// <summary>(x - means) / stds applied to every row.</summary>
    internal static Matrix<T> Standardize<T>(Matrix<T> x, Vector<T> means, Vector<T> stds)
        => Engine.TensorDivide(SubtractRow(x, means), TileRow(stds, x.Rows)).ToMatrix();

    /// <summary>x * stds + means applied to every row (inverse of <see cref="Standardize{T}"/>).</summary>
    internal static Matrix<T> Destandardize<T>(Matrix<T> x, Vector<T> means, Vector<T> stds)
    {
        var engine = Engine;
        return engine.TensorAdd(engine.TensorMultiply(ToTensor(x), TileRow(stds, x.Rows)), TileRow(means, x.Rows)).ToMatrix();
    }

    /// <summary>A [rows, d] tensor whose every row is <paramref name="row"/>.</summary>
    internal static Tensor<T> TileRow<T>(Vector<T> row, int rows)
        => Engine.TensorTile(Tensor<T>.FromVector(row, new[] { 1, row.Length }), new[] { rows, 1 });

    /// <summary>
    /// Dense rank of every entry (equal values share a rank, ranks ascend with the values), so integer comparisons
    /// of ranks order the entries exactly like comparisons of the values.
    /// </summary>
    internal static int[] DenseRanks<T>(Vector<T> values)
    {
        int n = values.Length;
        if (n == 0)
        {
            return Array.Empty<int>();
        }

        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var (sorted, order) = engine.TensorSort(values.AsTensor(), 0, false);

        // 1 where a sorted entry differs from its predecessor; the running sum is the dense rank
        var first = Tensor<T>.CreateZeros(new[] { 1 });
        var changes = n == 1
            ? first
            : engine.TensorConcatenate(new[]
            {
                first,
                engine.ScalarMinusTensor(numOps.One, engine.TensorEquals(
                    engine.TensorSlice(sorted, new[] { 1 }, new[] { n - 1 }),
                    engine.TensorSlice(sorted, new[] { 0 }, new[] { n - 1 })))
            }, 0);
        var sortedRanks = engine.TensorCumSum(changes, 0);
        return ToIntArray(engine.Scatter(Tensor<T>.CreateZeros(new[] { n }), order, sortedRanks, 0).ToVector());
    }
}
