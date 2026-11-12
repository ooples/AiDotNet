using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Helpers;

/// <summary>
/// Whole-array engine helpers shared by the feature selectors: column gathers, filled tensors, per-column
/// moments and ranked index selection with LINQ-compatible tie order.
/// </summary>
internal static class FeatureSelectionTensorOps
{

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
        return engine.TensorMultiply(squared, PreprocessingTensorOps.OneMinus(engine.TensorEye<T>(n)));
    }

    /// <summary>
    /// The k-nearest-neighbour mask of a pairwise distance matrix ([n, n]): entry (i, j) is 1 when j is one of
    /// the <paramref name="k"/> rows closest to row i, excluding i itself. Ties at the k-th distance go to the
    /// lower indices, like a stable <c>OrderBy(distance).Take(k)</c> over <c>j != i</c>.
    /// </summary>
    public static Tensor<T> KNearestMask<T>(Tensor<T> distances, int k)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = distances.Shape[0];
        if (k <= 0 || n == 0)
            return new Tensor<T>(new[] { n, n });

        // Push the diagonal past every real distance so a row never picks itself
        var eye = engine.TensorEye<T>(n);
        T beyond = ops.Add(engine.TensorMaxValue(distances), ops.One);
        var d = engine.TensorAdd(engine.TensorMultiply(distances, PreprocessingTensorOps.OneMinus(eye)), engine.TensorMultiplyScalar(eye, beyond));

        var sorted = engine.TensorSort(d, 1, false).Item1;
        var kth = engine.TensorBroadcastTo(engine.TensorSlice(sorted, new[] { 0, k - 1 }, new[] { n, 1 }), new[] { n, n });
        var less = engine.TensorLessThan(d, kth);
        var lessCount = engine.TensorBroadcastTo(engine.ReduceSum(less, new[] { 1 }, true), new[] { n, n });
        var tie = engine.TensorEquals(d, kth);
        var tieRank = engine.TensorCumSum(tie, 1);

        // Ties at the k-th distance fill the remaining k - lessCount places in index order
        var remaining = engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { n, n }, ops.FromDouble(k)), lessCount);
        var takeTie = engine.TensorMultiply(tie, PreprocessingTensorOps.OneMinus(engine.TensorGreaterThan(tieRank, remaining)));
        return engine.TensorAdd(less, takeTie);
    }

    /// <summary>
    /// All-pairs shortest path lengths (Floyd-Warshall) over a weighted graph ([n, n] edge lengths where
    /// <paramref name="edges"/> is 1, no edge elsewhere). Returns the path lengths and, in
    /// <paramref name="reachable"/>, the 0/1 mask of the connected pairs (the diagonal included).
    /// </summary>
    public static Tensor<T> ShortestPaths<T>(Tensor<T> lengths, Tensor<T> edges, out Tensor<T> reachable)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = lengths.Shape[0];
        var eye = engine.TensorEye<T>(n);

        // Unreachable pairs hold a sentinel longer than any simple path (the sum of all edge lengths, plus 1)
        var edgeLengths = engine.TensorMultiply(lengths, engine.TensorMultiply(edges, PreprocessingTensorOps.OneMinus(eye)));
        T unreachable = ops.Add(engine.TensorSum(edgeLengths), ops.One);
        var noEdge = PreprocessingTensorOps.OneMinus(engine.TensorMax(edges, eye));
        var graph = engine.TensorAdd(edgeLengths, engine.TensorMultiplyScalar(noEdge, unreachable));

        for (int via = 0; via < n; via++)
        {
            var throughVia = engine.TensorAdd(
                engine.TensorBroadcastTo(engine.TensorSlice(graph, new[] { 0, via }, new[] { n, 1 }), new[] { n, n }),
                engine.TensorBroadcastTo(engine.TensorSlice(graph, new[] { via, 0 }, new[] { 1, n }), new[] { n, n }));
            graph = engine.TensorMin(graph, throughVia);
        }

        reachable = engine.TensorLessThan(graph, unreachable);
        return graph;
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
        var positions = PreprocessingTensorOps.MaskPositions(neighborMask);
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
    /// normalize w to sum 1 (when the sum is not within 1e-10 of 0). Returns the [n, n] weight matrix.
    /// </summary>
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
        var gram = engine.TensorBatchMatMul(differences, engine.TensorPermute(differences, new[] { 0, 2, 1 }));

        var eye = engine.Reshape(engine.TensorEye<T>(k), new[] { 1, k, k });
        var trace = engine.ReduceSum(engine.TensorMultiply(gram, eye), new[] { 1, 2 }, true); // [n, 1, 1]
        T regularization = ops.FromDouble(reg);
        var ridge = engine.TensorWhere(
            engine.TensorGreaterThan(trace, ops.Zero),
            engine.TensorMultiplyScalar(trace, regularization),
            PreprocessingTensorOps.Filled(new[] { n, 1, 1 }, regularization));
        var system = engine.TensorAdd(gram, engine.TensorMultiply(engine.TensorBroadcastTo(ridge, new[] { n, k, k }), eye));

        var weights = LinalgDispatch<T>.Solve(system, PreprocessingTensorOps.Filled(new[] { n, k, 1 }, ops.One)); // [n, k, 1]
        var sum = engine.ReduceSum(weights, new[] { 1 }, true); // [n, 1, 1]
        var normalizable = engine.TensorGreaterThan(engine.TensorAbs(sum), ops.FromDouble(1e-10));
        var divisor = engine.TensorWhere(normalizable, sum, PreprocessingTensorOps.Filled(new[] { n, 1, 1 }, ops.One));
        weights = engine.TensorDivide(weights, divisor);

        var oneHot = engine.TensorOneHot<T>(new Tensor<int>((int[])neighborIndices.Clone(), new[] { n, k }), n); // [n, k, n]
        return engine.ReduceSum(engine.TensorMultiply(oneHot, weights), new[] { 1 }, false);
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

        var (ascendingValues, ascendingVectors) = LinalgDispatch<T>.Eigh(symmetric);
        var order = new int[count];
        for (int c = 0; c < count; c++)
            order[c] = p - 1 - c;
        var index = new Tensor<int>(order, new[] { count });
        values = engine.TensorIndexSelect(ascendingValues, index, 0);
        return engine.TensorIndexSelect(ascendingVectors, index, 1);
    }

    /// <summary>The sample covariance matrix (divisor n - 1) of the columns of an [n, p] tensor.</summary>
    public static Tensor<T> SampleCovariance<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var centered = engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));
        return engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(centered), centered), ops.FromDouble(x.Shape[0] - 1));
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
        var ones = PreprocessingTensorOps.Filled(x.Shape.ToArray(), MathHelper.GetNumericOperations<T>().One);
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
    /// ([n, pb]) as a [pa, pb] tensor; 0 where <c>sqrt(sxx * syy)</c> is not above 1e-10.
    /// </summary>
    public static Tensor<T> ColumnCorrelations<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int pa = a.Shape[1];
        int pb = b.Shape[1];
        var ac = engine.TensorSubtract(a, engine.ReduceMean(a, new[] { 0 }, true));
        var bc = engine.TensorSubtract(b, engine.ReduceMean(b, new[] { 0 }, true));
        var cross = engine.TensorMatMul(engine.TensorTranspose(ac), bc);
        var aSq = engine.ReduceSum(engine.TensorSquare(ac), new[] { 0 }, false);
        var bSq = engine.ReduceSum(engine.TensorSquare(bc), new[] { 0 }, false);
        // sqrt(sxx) sqrt(syy), not sqrt(sxx syy): the product overflows Half once both sums pass 256
        var denominator = engine.TensorOuter(engine.TensorSqrt(aSq), engine.TensorSqrt(bSq));
        var defined = engine.TensorGreaterThan(denominator, ops.FromDouble(1e-10));
        var safe = engine.TensorWhere(defined, denominator, PreprocessingTensorOps.Filled(new[] { pa, pb }, ops.One));
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
        AiDotNetEngine.Current.TensorTriu(PreprocessingTensorOps.Filled(new[] { n, n }, MathHelper.GetNumericOperations<T>().One), 1);

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
    public static Tensor<T> CenteredColumnStd<T>(Tensor<T> centered)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(centered), new[] { 0 }, false));
    }

    /// <summary>Column-wise standardization (population std); columns whose std is not above 1e-10 are left as they are.</summary>
    public static Tensor<T> StandardizeColumnsOrKeep<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var centered = FilterStatisticTensorOps.CenterColumns(x);
        var std = CenteredColumnStd(centered);
        var varying = engine.TensorGreaterThan(std, ops.FromDouble(1e-10));
        var safeStd = engine.Reshape(engine.TensorWhere(varying, std, PreprocessingTensorOps.Filled(new[] { p }, ops.One)), new[] { 1, p });
        var mask = engine.TensorBroadcastTo(engine.Reshape(varying, new[] { 1, p }), new[] { n, p });
        return engine.TensorWhere(mask, engine.TensorDivide(centered, safeStd), x);
    }

    /// <summary>
    /// Among the entries where the 0/1 <paramref name="mask"/> is set, the positions of the (at most
    /// <paramref name="k"/>) largest scores, returned in ascending index order; ties keep the lower index first.
    /// </summary>
    public static int[] TopKWhere<T>(Tensor<T> scores, Tensor<T> mask, int k)
    {
        var positions = PreprocessingTensorOps.MaskPositions(mask);
        if (positions.Length == 0 || k <= 0)
            return Array.Empty<int>();

        // ranked among the masked entries only: a fill of (min - 1) for the others equals min itself once the
        // scores are past 2^24 in float (2048 in Half) or -infinity, and then ties pulled unmasked entries in
        var candidateScores = AiDotNetEngine.Current.TensorIndexSelect(
            scores, new Tensor<int>((int[])positions.Clone(), new[] { positions.Length }), 0);
        var order = PreprocessingTensorOps.StableOrder(candidateScores, descending: true);
        var selected = order.Take(Math.Min(k, positions.Length)).Select(rank => positions[rank]).ToArray();
        Array.Sort(selected);
        return selected;
    }
}
