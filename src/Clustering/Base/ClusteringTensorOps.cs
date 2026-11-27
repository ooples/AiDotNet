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
    /// <remarks>
    /// A distance can be representable while its square is not (in half precision anything above ~256 squares past
    /// 65504). Rows whose direct sqrt(sum diff^2) overflows are recomputed as s * ||diff / s|| with s the row's largest
    /// |difference|, so every square stays at most one; all other rows keep the direct value.
    /// </remarks>
    internal static Tensor<T> RowEuclidean<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = Engine;
        var direct = engine.TensorSqrt(RowSquaredEuclidean(a, b));
        if (direct.Length == 0 || !HasNonFinite(direct))
        {
            return direct;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var diff = engine.TensorSubtract(a, b);
        int rows = diff.Shape[0];
        var scale = engine.ReduceMax(engine.TensorAbs(diff), new[] { 1 }, true);
        var safe = engine.TensorWhere(engine.TensorGreaterThan(scale, numOps.Zero), scale, engine.TensorOnesLike(scale));
        var scaled = engine.TensorDivide(diff, engine.TensorBroadcastTo(safe, new[] { rows, diff.Shape[1] }));
        var norms = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(scaled), new[] { 1 }, false));
        var rescaled = engine.TensorMultiply(norms, engine.Reshape(safe, new[] { rows }));
        return engine.TensorWhere(engine.TensorIsFinite(direct), direct, rescaled);
    }

    /// <summary>True when any entry of the tensor is NaN or infinite.</summary>
    private static bool HasNonFinite<T>(Tensor<T> t)
        => Engine.TensorMaskedSelect(t, Engine.TensorLogicalNot(Engine.TensorIsFinite(t))).Length > 0;

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

        // The blocks are concatenated once at the end: writing each one into the full [n, m] result with
        // TensorSetSlice would copy the whole result per block, which is quadratic in n.
        var blocks = new List<Tensor<T>>((n + rowsPerBlock - 1) / rowsPerBlock);
        for (int start = 0; start < n; start += rowsPerBlock)
        {
            int count = Math.Min(rowsPerBlock, n - start);
            var block = engine.TensorSlice(xt, new[] { start, 0 }, new[] { count, d });
            var a = engine.TensorRepeatInterleave(block, m, 0);
            var b = engine.TensorTile(yt, new[] { count, 1 });
            blocks.Add(engine.Reshape(rowPairs(a, b), new[] { count, m }));
        }

        return engine.TensorConcatenate(blocks.ToArray(), 0).ToMatrix();
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

    /// <summary>Index of the smallest entry in every row (first one on ties; NaN entries are never chosen).</summary>
    internal static int[] ArgMinRows<T>(Matrix<T> values)
    {
        if (values.Rows == 0)
        {
            return Array.Empty<int>();
        }

        return Engine.TensorArgMin(NanToInfinity(ToTensor(values)), 1).ToArray();
    }

    /// <summary>Index of the smallest entry of a non-empty vector (first one on ties; NaN entries are never chosen).</summary>
    internal static int ArgMin<T>(Vector<T> values) => Engine.TensorArgMin(NanToInfinity(values.AsTensor()), 0).ToArray()[0];

    /// <summary>
    /// Replaces NaN with +infinity. The loops the argmin helpers replaced compared with a strict <c>&lt;</c> from
    /// MaxValue, so a NaN was never selected; the engine argmin starts from element 0 and would keep a NaN there.
    /// </summary>
    private static Tensor<T> NanToInfinity<T>(Tensor<T> values)
    {
        var engine = Engine;
        var nan = engine.TensorIsNan(values);
        return engine.TensorMaskedSelect(values, nan).Length == 0
            ? values
            : engine.TensorMaskedFill(values, nan, MathHelper.GetNumericOperations<T>().FromDouble(double.PositiveInfinity));
    }

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
    /// with one bulk cast; fractional values round down.
    /// </summary>
    /// <remarks>
    /// The cost is proportional to the number of values, not to the range they span: label vectors are
    /// arbitrary ids (two clusters labelled 0 and 20 000 000 are valid input to every metric), so a lookup
    /// table over [min, max] is unbounded. The floored tensor is a fresh contiguous buffer, which
    /// <c>Cast</c> requires (it ignores the offset of a narrowed view).
    /// </remarks>
    internal static int[] ToIntArray<T>(Vector<T> integralValues)
    {
        int n = integralValues.Length;
        if (n == 0)
        {
            return Array.Empty<int>();
        }

        return Engine.TensorFloor(integralValues.AsTensor()).Contiguous().Cast<int>().ToArray();
    }

    /// <summary>
    /// Relabels cluster ids to consecutive slots 0..k-1 (ascending by id) so per-cluster tensors are sized by
    /// the number of clusters present instead of by the largest id. Negative ids (noise) become -1.
    /// <paramref name="slotLabels"/>[slot] is the original id; ids that are already 0..k-1 are unchanged.
    /// </summary>
    internal static Vector<T> CompactLabels<T>(int[] ids, out int[] slotLabels)
    {
        var present = new SortedSet<int>();
        foreach (int id in ids)
        {
            if (id >= 0)
            {
                present.Add(id);
            }
        }

        slotLabels = present.ToArray();
        var slotOf = new Dictionary<int, int>(slotLabels.Length);
        for (int slot = 0; slot < slotLabels.Length; slot++)
        {
            slotOf[slotLabels[slot]] = slot;
        }

        var slots = new int[ids.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            slots[i] = ids[i] >= 0 ? slotOf[ids[i]] : -1;
        }

        return FromIntArray<T>(slots);
    }

    /// <summary>Converts integer values (labels, indices) to a <typeparamref name="T"/> vector with one bulk gather.</summary>
    internal static Vector<T> FromIntArray<T>(int[] values)
    {
        if (values.Length == 0)
        {
            return new Vector<T>(0);
        }

        // A direct per-value conversion: a gather from a [min, max] lookup table would be sized by the range of
        // the ids (unbounded for sparse labels), and a ramp in T cannot represent integers above 2048 in half
        // precision.
        return new Tensor<int>((int[])values.Clone(), new[] { values.Length }).Cast<T>().ToVector();
    }

    /// <summary>Number of ids equal to each slot 0..k-1; ids outside [0, k) are not counted.</summary>
    /// <remarks>Counted in <see cref="int"/>: a running count in half precision stops at 2048.</remarks>
    internal static int[] CountIds(int[] ids, int k)
    {
        var counts = new int[Math.Max(0, k)];
        foreach (int id in ids)
        {
            if (id >= 0 && id < k)
            {
                counts[id]++;
            }
        }

        return counts;
    }

    /// <summary>
    /// One-hot membership mask [n, k] in <typeparamref name="T"/>: entry (i, c) is one when labels[i] == c.
    /// Labels outside [0, k) (noise, unassigned) produce an all-zero row.
    /// </summary>
    internal static Tensor<T> OneHotMask<T>(Vector<T> labels, int k) => OneHotRows<T>(ToIntArray(labels), k);

    /// <summary>
    /// One-hot mask [ids.Length, k] in <typeparamref name="T"/> built from <see cref="int"/> ids: entry (i, c) is one
    /// when ids[i] == c; ids outside [0, k) give an all-zero row.
    /// </summary>
    /// <remarks>
    /// The comparison is made on the int ids. Comparing against a ramp 0..k-1 built in <typeparamref name="T"/> would
    /// merge columns above 2048 in half precision, where consecutive integers are no longer representable.
    /// </remarks>
    internal static Tensor<T> OneHotRows<T>(int[] ids, int k)
    {
        int n = ids.Length;
        if (n == 0 || k <= 0)
        {
            return Tensor<T>.CreateZeros(new[] { n, Math.Max(0, k) });
        }

        return Engine.TensorOneHot<T>(new Tensor<int>((int[])ids.Clone(), new[] { n }), k);
    }

    /// <summary>The ids start, start + 1, ..., start + count - 1.</summary>
    internal static int[] Range(int start, int count) => Enumerable.Range(start, count).ToArray();

    /// <summary>True when every id is a valid slot in [0, k).</summary>
    private static bool AllWithin(int[] ids, int k)
    {
        foreach (int id in ids)
        {
            if (id < 0 || id >= k)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Number of samples carrying each label 0..k-1 ([k], counted in int and converted to T).</summary>
    internal static Vector<T> ClusterCounts<T>(Vector<T> labels, int k)
        => FromIntArray<T>(CountIds(ToIntArray(labels), k));

    /// <summary>Per-cluster sums of the rows of x [k, d] and member counts [k].</summary>
    internal static Matrix<T> ClusterSums<T>(Matrix<T> x, Vector<T> labels, int k, out Vector<T> counts)
        => ClusterSums(ToTensor(x), ToIntArray(labels), k, out counts);

    /// <summary>
    /// Per-cluster sums of the rows of <paramref name="x"/> ([n, d]) by cluster index <paramref name="ids"/> ([n]), as
    /// [k, d], and member counts [k]; indices outside [0, k) are not counted.
    /// </summary>
    internal static Matrix<T> ClusterSums<T>(Tensor<T> x, int[] ids, int k, out Vector<T> counts)
    {
        var engine = Engine;
        int n = ids.Length;

        // Counts are taken in int: accumulated in half precision they stop growing at 2048.
        counts = FromIntArray<T>(CountIds(ids, k));
        if (n > 0 && AllWithin(ids, k))
        {
#if NET7_0_OR_GREATER
            if (typeof(T) == typeof(double) && x.IsContiguous)
            {
                // Row by row into the cluster's sum in ascending row order, as the scatter-add below accumulates, so
                // the sums are bit-identical to it (SIMD runs across columns only, never across rows). One pass
                // without the scatter-add's per-call setup, which dominated k-means' update step.
                int d = x.Shape[1];
                var sums = new Tensor<T>(new[] { k, d });
                var source = AsDoubles(x.AsSpan());
                var target = AsWritableDoubles(sums.AsWritableSpan());
                int width = System.Numerics.Vector<double>.Count;
                for (int r = 0; r < n; r++)
                {
                    var row = source.Slice(r * d, d);
                    var sum = target.Slice(ids[r] * d, d);
                    int c = 0;
                    for (; c + width <= d; c += width)
                        (new System.Numerics.Vector<double>(sum.Slice(c)) + new System.Numerics.Vector<double>(row.Slice(c)))
                            .CopyTo(sum.Slice(c));
                    for (; c < d; c++) sum[c] += row[c];
                }

                return sums.ToMatrix();
            }
#endif
            // Every label names a cluster: scatter-add the rows, O(n d), instead of an [n, k] mask product.
            var index = Vector<int>.Wrap(ids).AsTensor();
            return engine.TensorIndexAdd(new Tensor<T>(new[] { k, x.Shape[1] }), 0, index, x).ToMatrix();
        }

        var mask = OneHotRows<T>(ids, k);
        return engine.TensorMatMul(engine.TensorTranspose(mask), x).ToMatrix();
    }

#if NET7_0_OR_GREATER
    /// <summary>A span of T as doubles; only valid when T is double.</summary>
    internal static ReadOnlySpan<double> AsDoubles<T>(ReadOnlySpan<T> values)
        => System.Runtime.InteropServices.MemoryMarshal.CreateReadOnlySpan(
            ref System.Runtime.CompilerServices.Unsafe.As<T, double>(
                ref System.Runtime.InteropServices.MemoryMarshal.GetReference(values)), values.Length);

    /// <summary>A writable span of T as doubles; only valid when T is double (only <see cref="ClusterSums{T}(Tensor{T}, int[], int, out Vector{T})"/> needs it).</summary>
    private static Span<double> AsWritableDoubles<T>(Span<T> values)
        => System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
            ref System.Runtime.CompilerServices.Unsafe.As<T, double>(
                ref System.Runtime.InteropServices.MemoryMarshal.GetReference(values)), values.Length);
#endif

    /// <summary>
    /// Per-cluster means of the rows of x [k, d]; clusters without members get a zero row. Also returns the counts.
    /// </summary>
    internal static Matrix<T> ClusterMeans<T>(Matrix<T> x, Vector<T> labels, int k, out Vector<T> counts)
    {
        var sums = ClusterSums(x, labels, k, out counts);
        return DivideRows(sums, counts);
    }

    /// <summary>
    /// Per-cluster means of the rows of <paramref name="x"/> ([n, d]) by cluster index <paramref name="ids"/>, as
    /// [k, d]; clusters without members get a zero row. Also returns the counts.
    /// </summary>
    internal static Matrix<T> ClusterMeans<T>(Tensor<T> x, int[] ids, int k, out Vector<T> counts)
    {
        var sums = ClusterSums(x, ids, k, out counts);
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
        => AssignedRows(ToIntArray(labels), centers, out valid);

    /// <summary>
    /// Gathers the row of <paramref name="centers"/> named by every cluster index ([n, d]); indices outside the centers
    /// give zero rows.
    /// </summary>
    internal static Tensor<T> AssignedRows<T>(int[] ids, Matrix<T> centers, out Tensor<T> valid)
    {
        var engine = Engine;
        int n = ids.Length;
        if (n > 0 && AllWithin(ids, centers.Rows))
        {
            // Every label names a center: gather the rows instead of multiplying by an [n, k] mask.
            valid = Tensor<T>.CreateDefault(new[] { n }, MathHelper.GetNumericOperations<T>().One);
            return engine.TensorIndexSelect(ToTensor(centers), Vector<int>.Wrap(ids).AsTensor(), 0);
        }

        var mask = OneHotRows<T>(ids, centers.Rows);
        valid = engine.ReduceSum(mask, new[] { 1 }, false);
        return engine.TensorMatMul(mask, ToTensor(centers));
    }

    /// <summary>Squared distance of every sample to its assigned center; zero for invalid labels.</summary>
    internal static Vector<T> AssignedSquaredDistances<T>(Matrix<T> x, Vector<T> labels, Matrix<T> centers)
        => AssignedSquaredDistances(x, ToIntArray(labels), centers);

    /// <summary>Squared distance of every sample to the center its cluster index names; zero for invalid indices.</summary>
    internal static Vector<T> AssignedSquaredDistances<T>(Matrix<T> x, int[] ids, Matrix<T> centers)
    {
        var engine = Engine;
        var assigned = AssignedRows(ids, centers, out var valid);
        return engine.TensorMultiply(RowSquaredEuclidean(ToTensor(x), assigned), valid).ToVector();
    }

    /// <summary>Within-cluster sum of squared distances to the assigned centers (labels outside the centers are skipped).</summary>
    internal static T Inertia<T>(Matrix<T> x, Vector<T> labels, Matrix<T> centers)
        => Inertia(x, ToIntArray(labels), centers);

    /// <summary>
    /// <see cref="Inertia{T}(Matrix{T}, Vector{T}, Matrix{T})"/> from <see cref="int"/> cluster indices, which stay exact
    /// where a label in <typeparamref name="T"/> would not (above 2048 in half precision).
    /// </summary>
    internal static T Inertia<T>(Matrix<T> x, int[] ids, Matrix<T> centers)
    {
        if (x.Rows == 0 || centers.Rows == 0)
        {
            return MathHelper.GetNumericOperations<T>().Zero;
        }

        return Engine.TensorSum(AssignedSquaredDistances(x, ids, centers).AsTensor());
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

    /// <summary>
    /// Draws <paramref name="count"/> indices, each with probability proportional to the (non-negative) weights, from
    /// one running sum: draw t is the first index whose running sum reaches u_t * total for the t-th
    /// random.NextDouble(), or 0 when none does. One draw matches <see cref="SampleProportional"/>.
    /// </summary>
    internal static int[] SampleProportionalMany<T>(Vector<T> weights, Random random, int count)
    {
        var draws = new int[count];
        if (weights.Length == 0)
        {
            for (int t = 0; t < count; t++) random.NextDouble();
            return draws;
        }

        var cumulative = Engine.TensorCumSum(weights.AsTensor(), 0);

        // The weights are non-negative, so the running sum ends at its maximum.
        T total = Engine.TensorMaxValue(cumulative);
        for (int t = 0; t < count; t++)
        {
            draws[t] = FirstReaching(cumulative, total, random.NextDouble());
        }

        return draws;
    }

    /// <summary>The first index whose running sum reaches <paramref name="fraction"/> * total, or 0 when none does.</summary>
    private static int FirstReaching<T>(Tensor<T> cumulative, T total, double fraction)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T threshold = numOps.Multiply(numOps.FromDouble(fraction), total);
        int position = Engine.TensorSearchSorted(cumulative, Vector<T>.FromArray(new[] { threshold }).AsTensor(), false).ToArray()[0];
        return position < cumulative.Length ? position : 0;
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
        => AssignedDistances(metric, x, ToIntArray(labels), centers);

    /// <summary>
    /// <see cref="AssignedDistances{T}(IDistanceMetric{T}, Matrix{T}, Vector{T}, Matrix{T})"/> from <see cref="int"/>
    /// cluster indices.
    /// </summary>
    internal static Vector<T> AssignedDistances<T>(IDistanceMetric<T> metric, Matrix<T> x, int[] ids, Matrix<T> centers)
    {
        var engine = Engine;
        var mask = OneHotRows<T>(ids, centers.Rows);
        var distances = Tensor<T>.FromMatrix(metric.ComputePairwise(x, centers));

        // Selected, not multiplied: 0 * inf for an infinite distance to another center would make the row NaN.
        return engine.ReduceSum(engine.TensorWhere(mask, distances, engine.TensorZerosLike(distances)), new[] { 1 }, false).ToVector();
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
    /// Column means and population standard deviations of x (deviations below 1e-10, or the smallest value
    /// <typeparamref name="T"/> resolves, become 1), and x standardized with them.
    /// </summary>
    internal static Matrix<T> StandardizeColumns<T>(Matrix<T> x, out Vector<T> means, out Vector<T> stds)
    {
        var engine = Engine;
        means = ColumnMeans(x);
        var centered = SubtractRow(x, means);
        var std = ColumnRootMeanSquare(centered);
        var tiny = engine.TensorLessThan(std, NumericTolerance.Of<T>(1e-10));
        stds = engine.TensorWhere(tiny, engine.TensorOnesLike(std), std).ToVector();
        return engine.TensorDivide(centered, TileRow(stds, x.Rows)).ToMatrix();
    }

    /// <summary>
    /// sqrt(mean of squares) of every column of a [n, d] tensor. Columns whose direct value overflows (in half
    /// precision any entry above ~256 squares past 65504) are recomputed as s * rms(c / s) with s the column's
    /// largest magnitude.
    /// </summary>
    private static Tensor<T> ColumnRootMeanSquare<T>(Tensor<T> centered)
    {
        var engine = Engine;
        var direct = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(centered), new[] { 0 }, false));
        if (direct.Length == 0 || !HasNonFinite(direct))
        {
            return direct;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        int rows = centered.Shape[0];
        int columns = centered.Shape[1];
        var scale = engine.ReduceMax(engine.TensorAbs(centered), new[] { 0 }, true);
        var safe = engine.TensorWhere(engine.TensorGreaterThan(scale, numOps.Zero), scale, engine.TensorOnesLike(scale));
        var scaled = engine.TensorDivide(centered, engine.TensorBroadcastTo(safe, new[] { rows, columns }));
        var rms = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(scaled), new[] { 0 }, false));
        var rescaled = engine.TensorMultiply(rms, engine.Reshape(safe, new[] { columns }));
        return engine.TensorWhere(engine.TensorIsFinite(direct), direct, rescaled);
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

        // 1 where a sorted entry differs from its predecessor; the running sum is the dense rank. The sum is taken
        // in int: a running sum in half precision stops growing at 2048.
        var changes = n == 1
            ? new int[1]
            : ToIntArray(engine.ScalarMinusTensor(numOps.One, engine.TensorEquals(
                engine.TensorSlice(sorted, new[] { 1 }, new[] { n - 1 }),
                engine.TensorSlice(sorted, new[] { 0 }, new[] { n - 1 }))).ToVector());
        var positions = order.ToArray();
        var ranks = new int[n];
        int rank = 0;
        for (int i = 1; i < n; i++)
        {
            rank += changes[i - 1];
            ranks[positions[i]] = rank;
        }

        return ranks;
    }
}
