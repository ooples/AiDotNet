using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Helpers;

/// <summary>
/// Whole-array engine helpers shared by the feature selectors: column gathers, filled tensors, per-column
/// moments and ranked index selection with LINQ-compatible tie order.
/// </summary>
internal static class FeatureSelectionTensorOps
{
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
    /// Per-column population moments of a 2-D tensor ([n, p]): the column means ([1, p]), the population
    /// standard deviations ([p]) and the centred data ([n, p]).
    /// </summary>
    public static Tensor<T> CenterColumns<T>(Tensor<T> x, out Tensor<T> mean, out Tensor<T> std)
    {
        var engine = AiDotNetEngine.Current;
        mean = engine.ReduceMean(x, new[] { 0 }, true);
        var centered = engine.TensorSubtract(x, mean);
        std = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(centered), new[] { 0 }, false));
        return centered;
    }

    /// <summary>
    /// The <paramref name="quantile"/>-th quantile of every column of a 2-D tensor ([n, p] to [p]), interpolating
    /// linearly between order statistics at position <c>quantile * (n - 1)</c> (numpy's default method).
    /// </summary>
    public static Tensor<T> ColumnQuantile<T>(Tensor<T> x, double quantile) =>
        AiDotNetEngine.Current.TensorQuantile(x, quantile, 0, false, QuantileInterpolation.Linear);

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
        return engine.TensorMultiply(squared, OneMinus(engine.TensorEye<T>(n)));
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
        var d = engine.TensorAdd(engine.TensorMultiply(distances, OneMinus(eye)), engine.TensorMultiplyScalar(eye, beyond));

        var sorted = engine.TensorSort(d, 1, false).Item1;
        var kth = engine.TensorBroadcastTo(engine.TensorSlice(sorted, new[] { 0, k - 1 }, new[] { n, 1 }), new[] { n, n });
        var less = engine.TensorLessThan(d, kth);
        var lessCount = engine.TensorBroadcastTo(engine.ReduceSum(less, new[] { 1 }, true), new[] { n, n });
        var tie = engine.TensorEquals(d, kth);
        var tieRank = engine.TensorCumSum(tie, 1);

        // Ties at the k-th distance fill the remaining k - lessCount places in index order
        var remaining = engine.TensorSubtract(Filled(new[] { n, n }, ops.FromDouble(k)), lessCount);
        var takeTie = engine.TensorMultiply(tie, OneMinus(engine.TensorGreaterThan(tieRank, remaining)));
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
        var edgeLengths = engine.TensorMultiply(lengths, engine.TensorMultiply(edges, OneMinus(eye)));
        T unreachable = ops.Add(engine.TensorSum(edgeLengths), ops.One);
        var noEdge = OneMinus(engine.TensorMax(edges, eye));
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
            Filled(new[] { n, 1, 1 }, regularization));
        var system = engine.TensorAdd(gram, engine.TensorMultiply(engine.TensorBroadcastTo(ridge, new[] { n, k, k }), eye));

        var weights = LinalgDispatch<T>.Solve(system, Filled(new[] { n, k, 1 }, ops.One)); // [n, k, 1]
        var sum = engine.ReduceSum(weights, new[] { 1 }, true); // [n, 1, 1]
        var normalizable = engine.TensorGreaterThan(engine.TensorAbs(sum), ops.FromDouble(1e-10));
        var divisor = engine.TensorWhere(normalizable, sum, Filled(new[] { n, 1, 1 }, ops.One));
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
    /// The positions of the <paramref name="k"/> smallest entries of a 1-D score tensor, returned in ascending
    /// index order; ties keep the lower index first (LINQ <c>OrderBy(...).Take(k).OrderBy(i)</c>).
    /// </summary>
    public static int[] BottomKIndicesSorted<T>(Tensor<T> scores, int k)
    {
        int count = Math.Min(Math.Max(k, 0), scores.Length);
        if (count == 0)
            return Array.Empty<int>();

        var selected = StableOrder(scores, descending: false).AsSpan().Slice(0, count).ToArray();
        Array.Sort(selected);
        return selected;
    }

    /// <summary>
    /// Among the entries where the 0/1 <paramref name="mask"/> is set, the positions of the (at most
    /// <paramref name="k"/>) largest scores, returned in ascending index order; ties keep the lower index first.
    /// </summary>
    public static int[] TopKWhere<T>(Tensor<T> scores, Tensor<T> mask, int k)
    {
        int candidates = CountNonZero(mask);
        if (candidates == 0)
            return Array.Empty<int>();

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T below = ops.Subtract(engine.TensorMinValue(scores), ops.One);
        var ranked = engine.TensorWhere(mask, scores, Filled(scores.Shape.ToArray(), below));
        return TopKIndicesSorted(ranked, Math.Min(k, candidates));
    }

    /// <summary>
    /// The indices that sort a 1-D tensor ascending (or descending), with tied values kept in their original
    /// index order like LINQ <c>OrderBy</c>/<c>OrderByDescending</c>. The engine argsort is not documented as
    /// stable, so it sorts a distinct composite key (rank of the value, then the index).
    /// </summary>
    public static int[] StableOrder<T>(Tensor<T> values, bool descending)
    {
        int count = values.Length;
        if (count == 0)
            return Array.Empty<int>();

        var engine = AiDotNetEngine.Current;
        var flat = values.Rank == 1 ? values : engine.Reshape(values, new[] { count });
        var sorted = engine.TensorSort(flat, 0, false).Item1;

        // ascending rank: #{m : v[m] < v[j]}; descending rank: #{m : v[m] > v[j]} = count - #{m : v[m] <= v[j]}
        var less = engine.TensorSearchSorted(sorted, flat, false).AsSpan().ToArray();
        var atMost = engine.TensorSearchSorted(sorted, flat, true).AsSpan().ToArray();
        var keys = new long[count];
        for (int j = 0; j < count; j++)
        {
            long rank = descending ? count - atMost[j] : less[j];
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
}
