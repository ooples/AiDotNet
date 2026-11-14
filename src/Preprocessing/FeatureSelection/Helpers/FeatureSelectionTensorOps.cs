using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Helpers;

/// <summary>
/// Whole-array engine helpers shared by the feature selectors: column gathers, filled tensors, per-column
/// moments and ranked index selection with LINQ-compatible tie order.
/// </summary>
/// <remarks>
/// Counts and ranks are kept in int, never accumulated in T (a Half running count stops at 2048, where
/// 2048 + 1 rounds back to 2048), and sums of squares are formed on pre-scaled values so that they do not
/// overflow Half (65504) once the values pass 256.
/// </remarks>
internal static class FeatureSelectionTensorOps
{
    /// <summary>The smallest positive normal Half, 2^-14.</summary>
    private const double HalfSmallestNormal = 6.103515625e-05;

    /// <summary>
    /// <paramref name="epsilon"/> as a T, raised to the smallest positive normal value of T where it is below
    /// that. Half cannot hold 1e-10 (it rounds to 0), which turns "not above epsilon" guards off; float, double
    /// and decimal keep <paramref name="epsilon"/> unchanged.
    /// </summary>
    private static T Tolerance<T>(double epsilon) =>
        MathHelper.GetNumericOperations<T>().FromDouble(typeof(T) == typeof(Half) ? Math.Max(epsilon, HalfSmallestNormal) : epsilon);

    /// <summary>
    /// A path length longer than any real one: +infinity for the IEEE types, and half of the largest value for
    /// the others (decimal has no infinity, and half the maximum leaves room for one addition of two of them).
    /// </summary>
    private static T Unreachable<T>()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (typeof(T) == typeof(double) || typeof(T) == typeof(float) || typeof(T) == typeof(Half))
            return ops.FromDouble(double.PositiveInfinity);

        return ops.Divide(ops.MaxValue, ops.FromDouble(2));
    }

    /// <summary>Gathers the given columns of <paramref name="data"/> into a new matrix, in the given order.</summary>
    public static Matrix<T> SelectColumns<T>(Matrix<T> data, int[] columns)
    {
        if (columns.Length == 0 || data.Rows == 0)
            return new Matrix<T>(data.Rows, columns.Length);

        var indices = new Tensor<int>((int[])columns.Clone(), new[] { columns.Length });
        return AiDotNetEngine.Current.TensorIndexSelect(Tensor<T>.FromMatrix(data), indices, 1).ToMatrix();
    }

    /// <summary>A tensor of the given shape with every element set to <paramref name="value"/>.</summary>
    public static Tensor<T> Filled<T>(int[] shape, T value)
    {
        var result = new Tensor<T>(shape);
        AiDotNetEngine.Current.TensorFill(result, value);
        return result;
    }

    /// <summary>Element-wise <c>1 - x</c> (logical not of a 0/1 mask).</summary>
    public static Tensor<T> OneMinus<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorAddScalar(engine.TensorNegate(x), MathHelper.GetNumericOperations<T>().One);
    }

    /// <summary>Column-wise minimum of a 2-D tensor ([n, p] to [p]).</summary>
    public static Tensor<T> ColumnMin<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(x), new[] { 0 }, false));
    }

    /// <summary>Column-wise maximum of a 2-D tensor ([n, p] to [p]).</summary>
    public static Tensor<T> ColumnMax<T>(Tensor<T> x) => AiDotNetEngine.Current.ReduceMax(x, new[] { 0 }, false);

    /// <summary>
    /// Sliding-window maximum along the last axis of a 2-D tensor ([rows, n]): entry i becomes the maximum over
    /// positions i - halfWindow .. i + halfWindow, clipped to the row (a flat dilation with border replication).
    /// </summary>
    public static Tensor<T> SlidingMax<T>(Tensor<T> x, int halfWindow)
    {
        if (halfWindow <= 0 || x.Length == 0)
            return x;

        var engine = AiDotNetEngine.Current;
        var padded = engine.PadNd(x, new[] { halfWindow, halfWindow }, PadMode.Replicate, MathHelper.GetNumericOperations<T>().Zero);
        var windows = engine.TensorUnfold(padded, 1, 2 * halfWindow + 1, 1);
        return engine.ReduceMax(windows, new[] { 2 }, false);
    }

    /// <summary>Sliding-window minimum along the last axis of a 2-D tensor (a flat erosion), see <see cref="SlidingMax{T}"/>.</summary>
    public static Tensor<T> SlidingMin<T>(Tensor<T> x, int halfWindow)
    {
        if (halfWindow <= 0 || x.Length == 0)
            return x;

        var engine = AiDotNetEngine.Current;
        return engine.TensorNegate(SlidingMax(engine.TensorNegate(x), halfWindow));
    }

    /// <summary>Pairwise Euclidean distances between the rows of a 2-D tensor ([n, p] to [n, n]).</summary>
    public static Tensor<T> PairwiseDistances<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorSqrt(PairwiseSquaredDistances(x));
    }

    /// <summary>
    /// Pairwise squared Euclidean distances between the rows of a 2-D tensor ([n, p] to [n, n]), clamped at 0
    /// against rounding in the <c>|a|^2 + |b|^2 - 2ab</c> expansion, with an exact zero diagonal.
    /// </summary>
    public static Tensor<T> PairwiseSquaredDistances<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        var squared = engine.TensorMax(engine.PairwiseDistanceSquared(x, x), ops.Zero);
        // the diagonal is set, not masked by a multiply: 0 * infinity (|a|^2 overflows Half past 256) is NaN
        return engine.TensorWhere(engine.TensorEye<T>(n), Filled(new[] { n, n }, ops.Zero), squared);
    }

    /// <summary>
    /// The k-nearest-neighbour mask of a pairwise distance matrix ([n, n]): entry (i, j) is 1 when j is one of
    /// the <paramref name="k"/> rows closest to row i, excluding i itself (k is capped at n - 1). Ties at the k-th
    /// distance go to the lower indices, like a stable <c>OrderBy(distance).Take(k)</c> over <c>j != i</c>.
    /// </summary>
    /// <remarks>
    /// One stable sort of all n * n distances orders them by (distance, row, column), and every row then takes its
    /// first k columns other than itself. The places are counted in int: a running tie count in Half stops at 2048
    /// (2048 + 1 rounds back to 2048), so rows with more than 2048 tied candidates would take every one of them.
    /// </remarks>
    public static Tensor<T> KNearestMask<T>(Tensor<T> distances, int k)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = distances.Shape[0];
        int take = Math.Min(k, n - 1);
        if (take <= 0)
            return new Tensor<T>(new[] { n, n });

        var order = StableOrder(distances, descending: false);
        var positions = new int[n * take];
        var taken = new int[n];
        foreach (int position in order)
        {
            int row = position / n;
            if (position - row * n == row || taken[row] == take)
                continue;

            positions[row * take + taken[row]] = position;
            taken[row]++;
        }

        var mask = engine.TensorScatter(
            new Tensor<T>(new[] { n * n }),
            new Tensor<int>(positions, new[] { n * take }),
            Filled(new[] { n * take }, ops.One),
            0);
        return engine.Reshape(mask, new[] { n, n });
    }

    /// <summary>
    /// All-pairs shortest path lengths (Floyd-Warshall) over a weighted graph ([n, n] edge lengths where
    /// <paramref name="edges"/> is 1, no edge elsewhere). Returns the path lengths, 0 for the pairs that are not
    /// connected, and, in <paramref name="reachable"/>, the 0/1 mask of the connected pairs (the diagonal included).
    /// </summary>
    /// <remarks>
    /// Pairs without an edge start at +infinity (half the largest value for types without an infinity) and are
    /// built with a where, not a mask multiply: a finite sentinel such as the sum of all edge lengths overflows
    /// Half past 65504, and 0 * infinity in a mask multiply is NaN on every edge.
    /// </remarks>
    public static Tensor<T> ShortestPaths<T>(Tensor<T> lengths, Tensor<T> edges, out Tensor<T> reachable)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = lengths.Shape[0];
        var eye = engine.TensorEye<T>(n);
        T unreachable = Unreachable<T>();

        var offDiagonalEdges = engine.TensorMultiply(edges, OneMinus(eye));
        var start = engine.TensorWhere(eye, Filled(new[] { n, n }, ops.Zero), Filled(new[] { n, n }, unreachable));
        var graph = engine.TensorWhere(offDiagonalEdges, lengths, start);

        for (int via = 0; via < n; via++)
        {
            var throughVia = engine.TensorAdd(
                engine.TensorBroadcastTo(engine.TensorSlice(graph, new[] { 0, via }, new[] { n, 1 }), new[] { n, n }),
                engine.TensorBroadcastTo(engine.TensorSlice(graph, new[] { via, 0 }, new[] { 1, n }), new[] { n, n }));
            graph = engine.TensorMin(graph, throughVia);
        }

        reachable = engine.TensorLessThan(graph, unreachable);
        return engine.TensorWhere(reachable, graph, Filled(new[] { n, n }, ops.Zero));
    }

    /// <summary>
    /// For every feature column d of <paramref name="x"/> ([n, p]) and every weight matrix W ([n, n]), the
    /// weighted sum of the single-feature distances <c>sum_ij W[i, j] * |x[i, d] - x[j, d]|</c>. Returns [p, weights].
    /// </summary>
    public static Tensor<T> AbsDifferenceSums<T>(Tensor<T> x, params Tensor<T>[] weights)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int p = x.Shape[1];
        if (p == 0 || weights.Length == 0)
            return new Tensor<T>(new[] { p, weights.Length });

        var flatWeights = engine.TensorStack(weights.Select(w => engine.Reshape(w, new[] { n * n })).ToArray(), 1);
        var rows = new Tensor<T>[p];
        for (int d = 0; d < p; d++)
        {
            var column = engine.TensorSlice(x, new[] { 0, d }, new[] { n, 1 });
            var distances = engine.TensorAbs(engine.TensorSubtract(
                engine.TensorBroadcastTo(column, new[] { n, n }),
                engine.TensorBroadcastTo(engine.TensorTranspose(column), new[] { n, n })));
            rows[d] = engine.TensorMatMul(engine.Reshape(distances, new[] { 1, n * n }), flatWeights);
        }

        return engine.TensorConcatenate(rows, 0);
    }

    /// <summary>
    /// The neighbour indices of a k-nearest-neighbour mask (exactly <paramref name="k"/> set entries per row),
    /// as a flat [n * k] array of column indices in ascending order within each row.
    /// </summary>
    public static int[] NeighborIndices<T>(Tensor<T> neighborMask, int k)
    {
        int n = neighborMask.Shape[0];
        var positions = MaskPositions(neighborMask);
        if (positions.Length != n * k)
            throw new InvalidOperationException("Every row of the neighbour mask must hold exactly k neighbours.");

        var indices = new int[positions.Length];
        for (int t = 0; t < positions.Length; t++)
            indices[t] = positions[t] % n;
        return indices;
    }

    /// <summary>
    /// Locally linear reconstruction (barycenter) weights, as in scikit-learn's <c>barycenter_kneighbors_graph</c>:
    /// for each row i with neighbours N(i), solve <c>(G + R I) w = 1</c> with the local Gram matrix
    /// <c>G = (x_i - x_N)(x_i - x_N)'</c> and <c>R = reg * trace(G)</c> (<c>reg</c> when the trace is 0), then
    /// normalize w to sum 1 (when the sum is not within the tolerance of 0). Returns the [n, n] weight matrix.
    /// </summary>
    /// <remarks>
    /// Every row's differences are divided by their largest magnitude s before the Gram matrix is formed:
    /// (x_i - x_N) squared overflows Half once a difference passes 256, which left an infinite system and zero
    /// weights. The ridge scales with the trace, so the scaled system is the original one divided by s^2 and its
    /// solution is s^2 w, which normalizes to the same weights. The "sum is 0" guard is taken on that
    /// dimensionless sum with a T-aware tolerance (1e-10, raised to Half's smallest normal 2^-14 since 1e-10 is 0
    /// in Half); rows that do not normalize are divided back by s^2.
    /// </remarks>
    public static Tensor<T> BarycenterWeights<T>(Tensor<T> x, int[] neighborIndices, int k, double reg)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        if (k <= 0 || n == 0)
            return new Tensor<T>(new[] { n, n });

        var index = new Tensor<int>((int[])neighborIndices.Clone(), new[] { n * k });
        var neighbors = engine.Reshape(engine.TensorIndexSelect(x, index, 0), new[] { n, k, p });
        var differences = engine.TensorSubtract(engine.Reshape(x, new[] { n, 1, p }), neighbors);
        var scale = Filled(new[] { n, 1, 1 }, ops.One);
        if (p > 0)
        {
            var largest = engine.ReduceMax(engine.TensorAbs(differences), new[] { 1, 2 }, true); // [n, 1, 1]
            scale = engine.TensorWhere(engine.TensorGreaterThan(largest, ops.Zero), largest, scale);
            differences = engine.TensorDivide(differences, scale);
        }

        var gram = engine.TensorBatchMatMul(differences, engine.TensorPermute(differences, new[] { 0, 2, 1 }));

        var eye = engine.Reshape(engine.TensorEye<T>(k), new[] { 1, k, k });
        var trace = engine.ReduceSum(engine.TensorMultiply(gram, eye), new[] { 1, 2 }, true); // [n, 1, 1]
        T regularization = ops.FromDouble(reg);
        var ridge = engine.TensorWhere(
            engine.TensorGreaterThan(trace, ops.Zero),
            engine.TensorMultiplyScalar(trace, regularization),
            Filled(new[] { n, 1, 1 }, regularization));
        var system = engine.TensorAdd(gram, engine.TensorMultiply(engine.TensorBroadcastTo(ridge, new[] { n, k, k }), eye));

        var weights = PreprocessingTensorOps.Solve(system, Filled(new[] { n, k, 1 }, ops.One)); // [n, k, 1]
        var sum = engine.ReduceSum(weights, new[] { 1 }, true); // [n, 1, 1]
        var normalizable = engine.TensorGreaterThan(engine.TensorAbs(sum), Tolerance<T>(1e-10));
        var divisor = engine.TensorWhere(normalizable, sum, engine.TensorSquare(scale));
        weights = engine.TensorDivide(weights, divisor);

        // Scatter-added straight into [n, n] (a repeated neighbour accumulates, as in the dense sum): an [n, k, n]
        // one-hot product needed k times the output's storage, 8 GB for n = 10000 and k = 10 in double.
        return engine.TensorScatterAdd(
            new Tensor<T>(new[] { n, n }),
            new Tensor<int>((int[])neighborIndices.Clone(), new[] { n, k }),
            engine.Reshape(weights, new[] { n, k }),
            1);
    }

    /// <summary>
    /// The <paramref name="count"/> largest eigenvalues of a symmetric matrix ([p, p]) and their eigenvectors
    /// (as the columns of a [p, count] tensor), in descending eigenvalue order.
    /// </summary>
    public static Tensor<T> TopEigenpairs<T>(Tensor<T> symmetric, int count, out Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        int p = symmetric.Shape[0];
        if (count <= 0 || p == 0)
        {
            values = new Tensor<T>(new[] { 0 });
            return new Tensor<T>(new[] { p, 0 });
        }

        var (ascendingValues, ascendingVectors) = PreprocessingTensorOps.SymmetricEigh(symmetric);
        var order = new int[count];
        for (int c = 0; c < count; c++)
            order[c] = p - 1 - c;
        var index = new Tensor<int>(order, new[] { count });
        values = engine.TensorIndexSelect(ascendingValues, index, 0);
        return engine.TensorIndexSelect(ascendingVectors, index, 1);
    }

    /// <summary>The sample covariance matrix (divisor n - 1) of the columns of an [n, p] tensor.</summary>
    /// <remarks>
    /// Every centred column is divided by its largest magnitude s and by sqrt(n - 1) before the product, and the
    /// result is multiplied back by s_j and then by s_k. Forming x^T x first overflows Half (65504) as soon as
    /// sum_i x_ij^2 does (twenty deviations of 100 already sum to 200000), although the covariance itself, 10526,
    /// is representable.
    /// </remarks>
    public static Tensor<T> SampleCovariance<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var centered = engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));
        var scale = ColumnScale(centered);
        var unit = engine.TensorDivideScalar(
            engine.TensorDivide(centered, engine.TensorBroadcastTo(engine.Reshape(scale, new[] { 1, p }), new[] { n, p })),
            ops.FromDouble(Math.Sqrt(n - 1)));
        var product = engine.TensorMatMul(engine.TensorTranspose(unit), unit);
        var byColumn = engine.TensorMultiply(
            product, engine.TensorBroadcastTo(engine.Reshape(scale, new[] { 1, p }), new[] { p, p }));
        return engine.TensorMultiply(
            byColumn, engine.TensorBroadcastTo(engine.Reshape(scale, new[] { p, 1 }), new[] { p, p }));
    }

    /// <summary>
    /// For every feature column d of <paramref name="x"/> ([n, p]) with its [n, n] kernel matrix
    /// <c>K_d = columnKernel(x[:, d])</c> (the column passed as [n, 1]), the weighted sums
    /// <c>sum_ij W[i, j] * K_d[i, j]</c> for each matrix in <paramref name="weights"/>, followed by
    /// <c>sum_ij V[i, j] * K_d[i, j]^2</c> for each matrix in <paramref name="squaredWeights"/>.
    /// Returns [p, weights + squaredWeights].
    /// </summary>
    public static Tensor<T> KernelWeightedSums<T>(
        Tensor<T> x, Func<Tensor<T>, Tensor<T>> columnKernel, Tensor<T>[] weights, Tensor<T>[] squaredWeights)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int p = x.Shape[1];
        int r = weights.Length + squaredWeights.Length;
        if (p == 0 || r == 0)
            return new Tensor<T>(new[] { p, r });

        var flat = weights.Concat(squaredWeights).Select(w => engine.Reshape(w, new[] { n * n })).ToArray();
        var linear = weights.Length > 0 ? engine.TensorStack(flat.Take(weights.Length).ToArray(), 1) : null;
        var quadratic = squaredWeights.Length > 0 ? engine.TensorStack(flat.Skip(weights.Length).ToArray(), 1) : null;
        var rows = new Tensor<T>[p];
        for (int d = 0; d < p; d++)
        {
            var kernel = engine.Reshape(columnKernel(engine.TensorSlice(x, new[] { 0, d }, new[] { n, 1 })), new[] { 1, n * n });
            var parts = new List<Tensor<T>>(2);
            if (linear is not null)
                parts.Add(engine.TensorMatMul(kernel, linear));
            if (quadratic is not null)
                parts.Add(engine.TensorMatMul(engine.TensorSquare(kernel), quadratic));
            rows[d] = parts.Count == 1 ? parts[0] : engine.TensorConcatenate(parts.ToArray(), 1);
        }

        return engine.TensorConcatenate(rows, 0);
    }

    /// <summary>
    /// Element-wise integer power <c>x^exponent</c> by repeated multiplication (exact for negative bases, like
    /// <c>Math.Pow</c> with an integral exponent); a negative exponent gives the reciprocal.
    /// </summary>
    public static Tensor<T> IntegerPower<T>(Tensor<T> x, int exponent)
    {
        var engine = AiDotNetEngine.Current;
        var ones = Filled(x.Shape.ToArray(), MathHelper.GetNumericOperations<T>().One);
        var result = ones;
        int remaining = Math.Abs(exponent);
        var factor = x;
        while (remaining > 0)
        {
            if ((remaining & 1) == 1)
                result = engine.TensorMultiply(result, factor);
            remaining >>= 1;
            if (remaining > 0)
                factor = engine.TensorSquare(factor);
        }

        return exponent < 0 ? engine.TensorDivide(ones, result) : result;
    }

    /// <summary>
    /// For every feature column d of <paramref name="x"/> ([n, p]), the row sums <c>sum_j K_d[i, j]</c> of its
    /// [n, n] kernel matrix <c>K_d = columnKernel(x[:, d])</c>. Returns [n, p].
    /// </summary>
    public static Tensor<T> KernelRowSums<T>(Tensor<T> x, Func<Tensor<T>, Tensor<T>> columnKernel)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int p = x.Shape[1];
        if (p == 0)
            return new Tensor<T>(new[] { n, 0 });

        var columns = new Tensor<T>[p];
        for (int d = 0; d < p; d++)
            columns[d] = engine.ReduceSum(columnKernel(engine.TensorSlice(x, new[] { 0, d }, new[] { n, 1 })), new[] { 1 }, true);

        return engine.TensorConcatenate(columns, 1);
    }

    /// <summary>
    /// The double-centred distance matrix of an [n, 1] column (distance covariance, Szekely et al. 2007):
    /// <c>a_ij - mean_i. - mean_.j + mean_..</c> with <c>a_ij = |c_i - c_j|</c>.
    /// </summary>
    public static Tensor<T> DoubleCenteredDistances<T>(Tensor<T> column)
    {
        var engine = AiDotNetEngine.Current;
        int n = column.Shape[0];
        var distances = engine.TensorAbs(engine.TensorSubtract(
            engine.TensorBroadcastTo(column, new[] { n, n }), engine.TensorBroadcastTo(engine.TensorTranspose(column), new[] { n, n })));
        return engine.TensorAddScalar(
            engine.TensorSubtract(
                engine.TensorSubtract(distances, engine.ReduceMean(distances, new[] { 1 }, true)),
                engine.ReduceMean(distances, new[] { 0 }, true)),
            engine.TensorMean(distances));
    }

    /// <summary>
    /// Pearson correlations between the columns of <paramref name="a"/> ([n, pa]) and of <paramref name="b"/>
    /// ([n, pb]) as a [pa, pb] tensor; 0 where <c>sqrt(sxx) sqrt(syy)</c> is not above a T-aware tolerance
    /// (1e-10, raised to Half's smallest normal 2^-14 since 1e-10 is 0 in Half).
    /// </summary>
    /// <remarks>
    /// The correlation is scale-free, so every centred column is first divided by its largest magnitude: sums of
    /// squares of the raw values overflow Half (65504) once the values pass 256, which made r inf / inf = NaN.
    /// The guard is taken on the unscaled <c>sqrt(sxx) sqrt(syy)</c>, rebuilt from the scaled sums.
    /// </remarks>
    public static Tensor<T> ColumnCorrelations<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int pa = a.Shape[1];
        int pb = b.Shape[1];
        var ac = engine.TensorSubtract(a, engine.ReduceMean(a, new[] { 0 }, true));
        var bc = engine.TensorSubtract(b, engine.ReduceMean(b, new[] { 0 }, true));
        var aScale = ColumnScale(ac);
        var bScale = ColumnScale(bc);
        var au = engine.TensorDivide(ac, engine.Reshape(aScale, new[] { 1, pa }));
        var bu = engine.TensorDivide(bc, engine.Reshape(bScale, new[] { 1, pb }));
        var cross = engine.TensorMatMul(engine.TensorTranspose(au), bu);
        var aNorm = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(au), new[] { 0 }, false));
        var bNorm = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(bu), new[] { 0 }, false));
        var denominator = engine.TensorOuter(aNorm, bNorm);

        // sqrt(sxx) sqrt(syy) = (scale_a |a'|)(scale_b |b'|); an overflow to +infinity only passes the guard
        var magnitude = engine.TensorOuter(engine.TensorMultiply(aScale, aNorm), engine.TensorMultiply(bScale, bNorm));
        var defined = engine.TensorGreaterThan(magnitude, Tolerance<T>(1e-10));
        var safe = engine.TensorWhere(defined, denominator, Filled(new[] { pa, pb }, ops.One));
        return engine.TensorWhere(defined, engine.TensorDivide(cross, safe), new Tensor<T>(new[] { pa, pb }));
    }

    /// <summary>A 1-D tensor of the given length that is 1 at <paramref name="index"/> and 0 elsewhere.</summary>
    public static Tensor<T> OneHot<T>(int index, int length) =>
        AiDotNetEngine.Current.Reshape(
            AiDotNetEngine.Current.TensorOneHot<T>(new Tensor<int>(new[] { index }, new[] { 1 }), length), new[] { length });

    /// <summary>A count (or other integer) as a scalar of type T.</summary>
    public static T Count<T>(int value) => MathHelper.GetNumericOperations<T>().FromDouble(value);

    /// <summary>The 0/1 mask of the strictly upper-triangular entries (i &lt; j) of an [n, n] matrix.</summary>
    public static Tensor<T> StrictUpperMask<T>(int n) =>
        AiDotNetEngine.Current.TensorTriu(Filled(new[] { n, n }, MathHelper.GetNumericOperations<T>().One), 1);

    /// <summary>
    /// Element-wise RBF kernel <c>exp(-gamma (a_i - a_j)^2)</c> between the entries of an [n, 1] column ([n, n]).
    /// </summary>
    public static Tensor<T> RbfColumnKernel<T>(Tensor<T> column, T negativeGamma)
    {
        var engine = AiDotNetEngine.Current;
        int n = column.Shape[0];
        var difference = engine.TensorSubtract(
            engine.TensorBroadcastTo(column, new[] { n, n }), engine.TensorBroadcastTo(engine.TensorTranspose(column), new[] { n, n }));
        return engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorSquare(difference), negativeGamma));
    }

    /// <summary>
    /// Population standard deviation of every column of an already centred [n, p] tensor ([p]): the square root
    /// of the column mean of the squares (numpy.std, ddof = 0, of the uncentred data).
    /// </summary>
    /// <remarks>
    /// The squares are taken of the column divided by its largest magnitude s, and the result multiplied back by
    /// s: squaring the raw values overflows Half (65504) once they pass 256, which made the std +infinity.
    /// </remarks>
    public static Tensor<T> CenteredColumnStd<T>(Tensor<T> centered)
    {
        var engine = AiDotNetEngine.Current;
        int p = centered.Shape[1];
        var scale = ColumnScale(centered);
        var unit = engine.TensorDivide(centered, engine.Reshape(scale, new[] { 1, p }));
        return engine.TensorMultiply(scale, engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(unit), new[] { 0 }, false)));
    }

    /// <summary>
    /// Column-wise standardization (population std); columns whose std is not above a T-aware tolerance (1e-10,
    /// raised to Half's smallest normal 2^-14 since 1e-10 is 0 in Half) are left as they are.
    /// </summary>
    public static Tensor<T> StandardizeColumnsOrKeep<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var centered = engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));
        var std = CenteredColumnStd(centered);
        var varying = engine.TensorGreaterThan(std, Tolerance<T>(1e-10));
        var safeStd = engine.Reshape(engine.TensorWhere(varying, std, Filled(new[] { p }, ops.One)), new[] { 1, p });
        var mask = engine.TensorBroadcastTo(engine.Reshape(varying, new[] { 1, p }), new[] { n, p });
        return engine.TensorWhere(mask, engine.TensorDivide(centered, safeStd), x);
    }

    /// <summary>
    /// Among the entries where the 0/1 <paramref name="mask"/> is set, the positions of the (at most
    /// <paramref name="k"/>) largest scores, returned in ascending index order; ties keep the lower index first.
    /// </summary>
    public static int[] TopKWhere<T>(Tensor<T> scores, Tensor<T> mask, int k)
    {
        var positions = MaskPositions(mask);
        if (positions.Length == 0 || k <= 0)
            return Array.Empty<int>();

        // ranked among the masked entries only: a fill of (min - 1) for the others equals min itself once the
        // scores are past 2^24 in float (2048 in Half) or -infinity, and then ties pulled unmasked entries in
        var candidateScores = AiDotNetEngine.Current.TensorIndexSelect(
            scores, new Tensor<int>((int[])positions.Clone(), new[] { positions.Length }), 0);
        var order = StableOrder(candidateScores, descending: true);
        var selected = order.Take(Math.Min(k, positions.Length)).Select(rank => positions[rank]).ToArray();
        Array.Sort(selected);
        return selected;
    }

    /// <summary>
    /// The positions of the <paramref name="k"/> largest entries of a 1-D score tensor, returned in ascending
    /// index order; ties keep the lower index first (LINQ <c>OrderByDescending(...).Take(k).OrderBy(i)</c>).
    /// </summary>
    public static int[] TopKIndicesSorted<T>(Tensor<T> scores, int k)
    {
        int count = Math.Min(Math.Max(k, 0), scores.Length);
        if (count == 0)
            return Array.Empty<int>();

        var selected = StableOrder(scores, descending: true).AsSpan().Slice(0, count).ToArray();
        Array.Sort(selected);
        return selected;
    }

    /// <summary>
    /// The indices that sort a 1-D tensor ascending (or descending), with tied values kept in their original
    /// index order like LINQ <c>OrderBy</c>/<c>OrderByDescending</c>. The engine argsort is not documented as
    /// stable, so it sorts a distinct composite key (rank of the value, then the index), with the ranks in int.
    /// </summary>
    /// <remarks>
    /// NaN orders as the worst value in both directions: after every number (infinities included) when
    /// ascending, and after every number when descending, so a NaN score is never the best.
    /// </remarks>
    public static int[] StableOrder<T>(Tensor<T> values, bool descending)
    {
        int count = values.Length;
        if (count == 0)
            return Array.Empty<int>();

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var flat = values.Rank == 1 ? values : engine.Reshape(values, new[] { count });

        // NaN takes a placeholder 0 for the rank search (ranks stay monotone in the real values) and is then
        // given rank `count`, past every real value's rank, whichever way the order runs.
        var isNaN = engine.TensorIsNan(flat);
        var zeros = Filled(new[] { count }, ops.Zero);
        var nanPositions = MaskPositions(engine.TensorWhere(isNaN, Filled(new[] { count }, ops.One), zeros));
        var ranked = nanPositions.Length == 0 ? flat : engine.TensorWhere(isNaN, zeros, flat);
        var sorted = engine.TensorSort(ranked, 0, false).Item1;

        // ascending rank: #{m : v[m] < v[j]}; descending rank: #{m : v[m] > v[j]} = count - #{m : v[m] <= v[j]}
        var less = engine.TensorSearchSorted(sorted, ranked, false).AsSpan().ToArray();
        var atMost = engine.TensorSearchSorted(sorted, ranked, true).AsSpan().ToArray();
        var notANumber = new bool[count];
        foreach (int position in nanPositions)
            notANumber[position] = true;

        var keys = new long[count];
        for (int j = 0; j < count; j++)
        {
            long rank = notANumber[j] ? count : descending ? count - atMost[j] : less[j];
            keys[j] = rank * count + j;
        }

        return engine.TensorArgsort(new Tensor<long>(keys, new[] { count }), 0, false).AsSpan().ToArray();
    }

    /// <summary>The positions where a 0/1 mask tensor is non-zero, in ascending order.</summary>
    public static int[] MaskPositions<T>(Tensor<T> mask)
    {
        var flat = mask.Rank == 1 ? mask : AiDotNetEngine.Current.Reshape(mask, new[] { mask.Length });
        return AiDotNetEngine.Current.TensorNonzero(flat).AsSpan().ToArray();
    }

    /// <summary>The number of non-zero entries of a 0/1 mask tensor.</summary>
    public static int CountNonZero<T>(Tensor<T> mask) => AiDotNetEngine.Current.TensorCountNonzero(mask);

    /// <summary>
    /// The largest magnitude of every column of an [n, p] tensor ([p]), with 1 for the all-zero (and empty)
    /// columns, used to pre-scale columns before their squares are summed.
    /// </summary>
    private static Tensor<T> ColumnScale<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = x.Shape[1];
        var ones = Filled(new[] { p }, ops.One);
        if (x.Shape[0] == 0 || p == 0)
            return ones;

        var largest = engine.ReduceMax(engine.TensorAbs(x), new[] { 0 }, false);
        return engine.TensorWhere(engine.TensorGreaterThan(largest, ops.Zero), largest, ones);
    }
}
