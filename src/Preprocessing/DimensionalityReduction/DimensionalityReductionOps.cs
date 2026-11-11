using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Whole-array building blocks shared by the dimensionality-reduction transformers: centering, covariance,
/// distances, neighbour selection and the symmetric eigen solve, all generic in <typeparamref name="T"/>.
/// </summary>
/// <remarks>
/// Every member works on <see cref="Tensor{T}"/> through <see cref="AiDotNetEngine.Current"/>, so the transformers
/// never stage numeric data in <c>double</c> buffers. Sorting relies on <see cref="IEngine.TensorArgsort{T}"/>,
/// which keeps equal keys in increasing index order (the same tie order as LINQ <c>OrderBy</c>/<c>OrderByDescending</c>).
/// </remarks>
internal static class DimensionalityReductionOps
{
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Wraps an index list as a rank-1 index tensor.</summary>
    internal static Tensor<int> IndexTensor(int[] indices) => new Tensor<int>(indices, new[] { indices.Length });

    /// <summary>The indices <c>0 .. count - 1</c> as a rank-1 index tensor.</summary>
    internal static Tensor<int> Range(int count) => IndexTensor(Enumerable.Range(0, count).ToArray());

    /// <summary>A rank-1 tensor of <paramref name="length"/> copies of <paramref name="value"/>.</summary>
    internal static Tensor<T> Full<T>(int length, T value) => Tensor<T>.CreateDefault(new[] { length }, value);

    /// <summary>A tensor of the given shape filled with <paramref name="value"/>.</summary>
    internal static Tensor<T> Full<T>(int[] shape, T value) => Tensor<T>.CreateDefault(shape, value);

    /// <summary>A 0-d-like single element tensor (shape [1]) holding <paramref name="value"/>.</summary>
    internal static Tensor<T> Scalar<T>(T value) => Tensor<T>.CreateDefault(new[] { 1 }, value);

    /// <summary>Column means of a [n, p] tensor as a [p] tensor.</summary>
    internal static Tensor<T> ColumnMeans<T>(Tensor<T> x) => Engine.ReduceMean(x, new[] { 0 }, false);

    /// <summary>Column sums of a [n, p] tensor as a [p] tensor.</summary>
    internal static Tensor<T> ColumnSums<T>(Tensor<T> x) => Engine.ReduceSum(x, new[] { 0 }, false);

    /// <summary>Row sums of a [n, p] tensor as a [n] tensor.</summary>
    internal static Tensor<T> RowSums<T>(Tensor<T> x) => Engine.ReduceSum(x, new[] { 1 }, false);

    /// <summary>Reshapes a [p] tensor to a [1, p] row so it broadcasts over the rows of a matrix.</summary>
    internal static Tensor<T> AsRow<T>(Tensor<T> v) => Engine.Reshape(v, new[] { 1, v.Length });

    /// <summary>Reshapes a [n] tensor to a [n, 1] column so it broadcasts over the columns of a matrix.</summary>
    internal static Tensor<T> AsColumn<T>(Tensor<T> v) => Engine.Reshape(v, new[] { v.Length, 1 });

    /// <summary>x - row for every row of x (row is [p]).</summary>
    internal static Tensor<T> SubtractRow<T>(Tensor<T> x, Tensor<T> row) => Engine.TensorSubtract(x, AsRow(row));

    /// <summary>x + row for every row of x (row is [p]).</summary>
    internal static Tensor<T> AddRow<T>(Tensor<T> x, Tensor<T> row) => Engine.TensorAdd(x, AsRow(row));

    /// <summary>Scales column j of x by scale[j].</summary>
    internal static Tensor<T> ScaleColumns<T>(Tensor<T> x, Tensor<T> scale) => Engine.TensorMultiply(x, AsRow(scale));

    /// <summary>Scales row i of x by scale[i].</summary>
    internal static Tensor<T> ScaleRows<T>(Tensor<T> x, Tensor<T> scale) => Engine.TensorMultiply(x, AsColumn(scale));

    /// <summary>a^T b.</summary>
    internal static Tensor<T> TransposeMatMul<T>(Tensor<T> a, Tensor<T> b) => Engine.TensorMatMul(Engine.TensorTranspose(a), b);

    /// <summary>a b^T.</summary>
    internal static Tensor<T> MatMulTranspose<T>(Tensor<T> a, Tensor<T> b) => Engine.TensorMatMul(a, Engine.TensorTranspose(b));

    /// <summary>(a + a^T) / 2, removing the rounding asymmetry a blocked matmul can leave in a Gram matrix.</summary>
    internal static Tensor<T> Symmetrize<T>(Tensor<T> a)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return Engine.TensorMultiplyScalar(Engine.TensorAdd(a, Engine.TensorTranspose(a)), numOps.FromDouble(0.5));
    }

    /// <summary>The [n, n] identity.</summary>
    internal static Tensor<T> Identity<T>(int n) => Engine.TensorEye<T>(n);

    /// <summary>
    /// Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations (accurate to working precision for
    /// every eigenpair, unlike power iteration with deflation). Eigenvalues are ordered by value (descending or
    /// ascending, ties in index order) and the eigenvectors are the matching columns, each signed so its entries
    /// sum to a non-negative value (the sign power iteration from the uniform start vector converges to).
    /// </summary>
    internal static (Tensor<T> Values, Tensor<T> Vectors) SymmetricEigen<T>(Tensor<T> a, bool descending)
    {
        var engine = Engine;
        var eigen = new EigenDecomposition<T>(Symmetrize(a).ToMatrix(), EigenAlgorithmType.Jacobi);
        var values = Tensor<T>.FromVector(eigen.EigenValues);
        var order = engine.TensorArgsort(values, 0, descending);
        var vectors = engine.TensorIndexSelect(Tensor<T>.FromMatrix(eigen.EigenVectors), order, 1);
        return (engine.TensorIndexSelect(values, order, 0), AlignColumnSigns(vectors));
    }

    /// <summary>Flips each column whose entries sum to a negative value.</summary>
    internal static Tensor<T> AlignColumnSigns<T>(Tensor<T> columns)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var negative = engine.TensorLessThan(engine.ReduceSum(columns, new[] { 0 }, true), numOps.Zero);
        var sign = engine.TensorAddScalar(engine.TensorMultiplyScalar(negative, numOps.FromDouble(-2.0)), numOps.One);
        return engine.TensorMultiply(columns, sign);
    }

    /// <summary>Euclidean distances between the rows of a [n, p] and b [m, p] (direct differences, no Gram expansion).</summary>
    internal static Tensor<T> EuclideanDistances<T>(Tensor<T> a, Tensor<T> b) => Engine.TensorCDist(a, b, 2.0);

    /// <summary>Squared Euclidean distances between the rows of a and b.</summary>
    internal static Tensor<T> SquaredDistances<T>(Tensor<T> a, Tensor<T> b) => Engine.TensorSquare(Engine.TensorCDist(a, b, 2.0));

    /// <summary>
    /// For every row i of a square distance matrix, the indices of the <paramref name="k"/> smallest entries
    /// excluding the diagonal, nearest first, ties in index order (LINQ <c>Where(j != i).OrderBy(d).Take(k)</c>).
    /// </summary>
    internal static Tensor<int> NearestNeighbors<T>(Tensor<T> distances, int k)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = distances.Shape[0];
        int take = Math.Min(k, n - 1);
        var masked = distances.Clone();
        engine.TensorFillDiagonalInPlace(masked, numOps.FromDouble(double.PositiveInfinity));
        var order = engine.TensorArgsort(masked, 1, false);
        return take == n ? order : engine.TensorNarrow(order, 1, 0, take);
    }

    /// <summary>Gathers source[indices[i, j], j] for a [n, k] index matrix.</summary>
    internal static Tensor<T> TakeRows<T>(Tensor<T> source, Tensor<int> indices) => Engine.TensorTakeAlongDim(source, indices, 1);

    /// <summary>
    /// Scatters a [n, k] value matrix into a dense [n, n] matrix at (i, indices[i, j]); other entries are
    /// <paramref name="fill"/>.
    /// </summary>
    internal static Tensor<T> ScatterRows<T>(Tensor<int> indices, Tensor<T> values, int n, T fill)
    {
        var dense = Full(new[] { indices.Shape[0], n }, fill);
        return Engine.TensorScatter(dense, indices, values, 1);
    }

    /// <summary>Selects the given rows of a matrix.</summary>
    internal static Tensor<T> SelectRows<T>(Tensor<T> x, Tensor<int> rows) => Engine.TensorIndexSelect(x, rows, 0);

    /// <summary>Selects the given columns of a matrix.</summary>
    internal static Tensor<T> SelectColumns<T>(Tensor<T> x, Tensor<int> columns) => Engine.TensorIndexSelect(x, columns, 1);

    /// <summary>The first <paramref name="count"/> columns of a matrix.</summary>
    internal static Tensor<T> FirstColumns<T>(Tensor<T> x, int count) => Engine.TensorNarrow(x, 1, 0, count);

    /// <summary>The first <paramref name="count"/> rows of a matrix.</summary>
    internal static Tensor<T> FirstRows<T>(Tensor<T> x, int count) => Engine.TensorNarrow(x, 0, 0, count);

    /// <summary>Row-major [rows, columns] uniform [0, 1) draws consuming <paramref name="random"/> like a nested NextDouble loop.</summary>
    internal static Tensor<T> Uniform<T>(Random random, int rows, int columns) =>
        Tensor<T>.FromMatrix(Matrix<T>.CreateRandom(random, rows, columns));

    /// <summary>
    /// Row-major [rows, columns] standard normal draws consuming <paramref name="random"/> exactly like a nested
    /// loop of <c>RandomExtensions.NextGaussian</c>: two NextDouble values per entry (u1 then u2),
    /// z = sqrt(-2 ln max(u1, double.Epsilon)) sin(2 pi u2).
    /// </summary>
    internal static Tensor<T> StandardNormal<T>(Random random, int rows, int columns)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var draws = engine.Reshape(Uniform<T>(random, rows * columns, 2), new[] { rows * columns, 2 });
        var u1 = engine.TensorClampMin(engine.TensorSliceAxis(draws, 1, 0), numOps.FromDouble(double.Epsilon));
        var u2 = engine.TensorSliceAxis(draws, 1, 1);
        var radius = engine.TensorSqrt(engine.TensorMultiplyScalar(engine.TensorLog(u1), numOps.FromDouble(-2.0)));
        var angle = engine.TensorSin(engine.TensorMultiplyScalar(u2, numOps.FromDouble(2.0 * Math.PI)));
        return engine.Reshape(engine.TensorMultiply(radius, angle), new[] { rows, columns });
    }

    /// <summary>Solves a x = b for a square a and a [n] or [n, m] right-hand side by LU with partial pivoting.</summary>
    internal static Tensor<T> Solve<T>(Tensor<T> a, Tensor<T> b)
    {
        var lu = new LuDecomposition<T>(a.ToMatrix());
        if (b.Rank == 1)
            return Tensor<T>.FromVector(lu.Solve(b.ToVector()));

        var rhs = b.ToMatrix();
        var solution = new Matrix<T>(rhs.Rows, rhs.Columns);
        for (int c = 0; c < rhs.Columns; c++)
            solution.SetColumn(c, lu.Solve(rhs.GetColumn(c)));
        return Tensor<T>.FromMatrix(solution);
    }

    /// <summary>The inverse of a square matrix by LU with partial pivoting.</summary>
    internal static Tensor<T> Inverse<T>(Tensor<T> a) => Solve(a, Identity<T>(a.Shape[0]));

    /// <summary>A loop-level constant (step size, weight) as T; one conversion per iteration, not per element.</summary>
    internal static T Constant<T>(double value) => MathHelper.GetNumericOperations<T>().FromDouble(value);

    /// <summary>Euclidean (Frobenius) norm of a tensor.</summary>
    internal static T Norm<T>(Tensor<T> x) =>
        MathHelper.GetNumericOperations<T>().Sqrt(Engine.TensorSum(Engine.TensorSquare(x)));

    /// <summary>Index of the first largest entry of a rank-1 tensor.</summary>
    internal static int FirstArgMax<T>(Tensor<T> values) => Engine.TensorArgMax(values, 0)[0];

    /// <summary>Index of the first smallest entry of a rank-1 tensor.</summary>
    internal static int FirstArgMin<T>(Tensor<T> values) => Engine.TensorArgMin(values, 0)[0];

    /// <summary>Returns a copy of a rank-1 0/1 mask with position <paramref name="index"/> set to 1.</summary>
    internal static Tensor<T> MarkIndex<T>(Tensor<T> mask, int index)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return Engine.TensorScatter(mask, IndexTensor(new[] { index }), Scalar(numOps.One), 0);
    }

    /// <summary>|a - b| &lt; tolerance: the scalar convergence test of an iterative fit (one call per iteration).</summary>
    internal static bool ChangeBelow<T>(T a, T b, T tolerance)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.LessThan(numOps.Abs(numOps.Subtract(a, b)), tolerance);
    }

    /// <summary>value &lt; threshold for a scalar reduction result.</summary>
    internal static bool Below<T>(T value, T threshold) => MathHelper.GetNumericOperations<T>().LessThan(value, threshold);

    /// <summary>value &gt; threshold for a scalar reduction result.</summary>
    internal static bool Above<T>(T value, T threshold) => MathHelper.GetNumericOperations<T>().GreaterThan(value, threshold);

    /// <summary>
    /// Double centering of a squared-distance matrix: B = -1/2 (D2 - row means - column means + grand mean).
    /// </summary>
    internal static Tensor<T> DoubleCenter<T>(Tensor<T> squaredDistances)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var rowMeans = engine.ReduceMean(squaredDistances, new[] { 1 }, true);
        var columnMeans = engine.ReduceMean(squaredDistances, new[] { 0 }, true);
        var grandMean = engine.TensorMean(rowMeans);
        var centered = engine.TensorAddScalar(
            engine.TensorSubtract(engine.TensorSubtract(squaredDistances, rowMeans), columnMeans), grandMean);
        return engine.TensorMultiplyScalar(centered, numOps.FromDouble(-0.5));
    }

    /// <summary>
    /// Classical (Torgerson) MDS of a distance matrix: the leading <paramref name="components"/> eigenvectors of the
    /// double-centred squared distances, each scaled by sqrt(max(eigenvalue, 0)).
    /// </summary>
    internal static Tensor<T> ClassicalMds<T>(Tensor<T> distances, int components)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        int k = Math.Min(components, distances.Shape[0]);
        var b = DoubleCenter(engine.TensorSquare(distances));
        var (values, vectors) = SymmetricEigen(b, descending: true);
        var scale = engine.TensorSqrt(engine.TensorClampMin(engine.TensorNarrow(values, 0, 0, k), numOps.Zero));
        return ScaleColumns(FirstColumns(vectors, k), scale);
    }

    /// <summary>
    /// Modified Gram-Schmidt on the columns of a [m, n] matrix: each column in turn is normalised (left as is when
    /// its norm is at most <paramref name="tolerance"/>) and its projection removed from every later column.
    /// </summary>
    internal static Tensor<T> OrthonormalizeColumns<T>(Tensor<T> a, double tolerance = 1e-10)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        T tol = numOps.FromDouble(tolerance);
        int n = a.Shape[1];
        var columns = new Tensor<T>[n];
        var remaining = a;
        for (int j = 0; j < n; j++)
        {
            var column = engine.TensorNarrow(remaining, 1, 0, 1);
            var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(column), new[] { 0 }, true));
            var divisor = engine.TensorWhere(engine.TensorGreaterThan(norm, tol), norm, engine.TensorOnesLike(norm));
            column = engine.TensorDivide(column, divisor);
            columns[j] = column;

            if (j + 1 < n)
            {
                var rest = engine.TensorNarrow(remaining, 1, 1, n - j - 1);
                remaining = engine.TensorSubtract(rest, engine.TensorMatMul(column, TransposeMatMul(column, rest)));
            }
        }

        return engine.TensorConcatenate(columns, 1);
    }

    /// <summary>Soft thresholding sign(x) max(|x| - lambda, 0), with lambda broadcast against x.</summary>
    internal static Tensor<T> SoftThreshold<T>(Tensor<T> x, Tensor<T> lambda)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var shrunk = engine.TensorClampMin(engine.TensorSubtract(engine.TensorAbs(x), lambda), numOps.Zero);
        return engine.TensorMultiply(engine.TensorSign(x), shrunk);
    }

    /// <summary>Soft thresholding by a scalar lambda.</summary>
    internal static Tensor<T> SoftThreshold<T>(Tensor<T> x, T lambda) => SoftThreshold(x, Scalar(lambda));

    /// <summary>Divides every row by its Euclidean norm, leaving rows whose norm is at most <paramref name="tolerance"/> unchanged.</summary>
    internal static Tensor<T> NormalizeRows<T>(Tensor<T> x, double tolerance = 1e-10)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(x), new[] { 1 }, true));
        var divisor = engine.TensorWhere(
            engine.TensorGreaterThan(norm, numOps.FromDouble(tolerance)), norm, engine.TensorOnesLike(norm));
        return engine.TensorDivide(x, divisor);
    }

    /// <summary>Divides every column by its Euclidean norm, leaving columns whose norm is at most <paramref name="tolerance"/> unchanged.</summary>
    internal static Tensor<T> NormalizeColumns<T>(Tensor<T> x, double tolerance = 1e-10)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(x), new[] { 0 }, true));
        var divisor = engine.TensorWhere(
            engine.TensorGreaterThan(norm, numOps.FromDouble(tolerance)), norm, engine.TensorOnesLike(norm));
        return engine.TensorDivide(x, divisor);
    }

    /// <summary>Largest absolute entry of a - b.</summary>
    internal static T MaxAbsDifference<T>(Tensor<T> a, Tensor<T> b) =>
        Engine.TensorMaxValue(Engine.TensorAbs(Engine.TensorSubtract(a, b)));

    /// <summary>Replaces row <paramref name="row"/> of a [m, n] matrix with a [1, n] or [n] tensor (returns a new matrix).</summary>
    internal static Tensor<T> WithRow<T>(Tensor<T> matrix, int row, Tensor<T> values)
    {
        var engine = Engine;
        int m = matrix.Shape[0];
        var parts = new List<Tensor<T>>(3);
        if (row > 0) parts.Add(engine.TensorNarrow(matrix, 0, 0, row));
        parts.Add(engine.Reshape(values, new[] { 1, matrix.Shape[1] }));
        if (row + 1 < m) parts.Add(engine.TensorNarrow(matrix, 0, row + 1, m - row - 1));
        return parts.Count == 1 ? parts[0] : engine.TensorConcatenate(parts.ToArray(), 0);
    }

    /// <summary>Replaces column <paramref name="column"/> of a [m, n] matrix with a [m, 1] or [m] tensor (returns a new matrix).</summary>
    internal static Tensor<T> WithColumn<T>(Tensor<T> matrix, int column, Tensor<T> values)
    {
        var engine = Engine;
        int n = matrix.Shape[1];
        var parts = new List<Tensor<T>>(3);
        if (column > 0) parts.Add(engine.TensorNarrow(matrix, 1, 0, column));
        parts.Add(engine.Reshape(values, new[] { matrix.Shape[0], 1 }));
        if (column + 1 < n) parts.Add(engine.TensorNarrow(matrix, 1, column + 1, n - column - 1));
        return parts.Count == 1 ? parts[0] : engine.TensorConcatenate(parts.ToArray(), 1);
    }

    /// <summary>
    /// Moore-Penrose pseudo-inverse of a [k, p] matrix with k &lt;= p: A^T (A A^T)^+, where (A A^T)^+ inverts the
    /// eigenvalues above 1e-10 times the largest (Jacobi eigen solve of the small Gram matrix).
    /// </summary>
    internal static Tensor<T> PseudoInverse<T>(Tensor<T> a)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        bool wide = a.Shape[0] <= a.Shape[1];
        var gram = wide ? MatMulTranspose(a, a) : TransposeMatMul(a, a);
        var (values, vectors) = SymmetricEigen(gram, descending: true);
        T cutoff = numOps.Multiply(engine.TensorMaxValue(engine.TensorAbs(values)), numOps.FromDouble(1e-10));
        var inverted = engine.TensorWhere(
            engine.TensorGreaterThan(values, cutoff), engine.TensorReciprocal(values), engine.TensorZerosLike(values));
        var gramPinv = MatMulTranspose(ScaleColumns(vectors, inverted), vectors);
        return wide ? TransposeMatMul(a, gramPinv) : MatMulTranspose(gramPinv, a);
    }

    /// <summary>
    /// Neighbourhood offsets for locally linear methods: result[i, j, :] = x[neighbors[i, j]] - x[i], shape [n, k, p].
    /// </summary>
    internal static Tensor<T> NeighborhoodDifferences<T>(Tensor<T> x, Tensor<int> neighbors)
    {
        var engine = Engine;
        int n = neighbors.Shape[0];
        int k = neighbors.Shape[1];
        int p = x.Shape[1];
        var gathered = engine.Reshape(engine.TensorIndexSelect(x, engine.Reshape(neighbors, new[] { n * k }), 0), new[] { n, k, p });
        return engine.TensorSubtract(gathered, engine.Reshape(x, new[] { n, 1, p }));
    }

    /// <summary>Gathers the neighbourhood rows x[neighbors[i, j]] into an [n, k, p] tensor.</summary>
    internal static Tensor<T> NeighborhoodRows<T>(Tensor<T> x, Tensor<int> neighbors)
    {
        var engine = Engine;
        int n = neighbors.Shape[0];
        int k = neighbors.Shape[1];
        return engine.Reshape(engine.TensorIndexSelect(x, engine.Reshape(neighbors, new[] { n * k }), 0), new[] { n, k, x.Shape[1] });
    }

    /// <summary>Batched a bᵀ over the leading axis: [n, r, c] x [n, s, c] -> [n, r, s].</summary>
    internal static Tensor<T> BatchMatMulTranspose<T>(Tensor<T> a, Tensor<T> b) =>
        Engine.TensorBatchMatMul(a, Engine.TensorSwapAxes(b, 1, 2));

    /// <summary>Batched aᵀ b over the leading axis: [n, r, c] x [n, r, s] -> [n, c, s].</summary>
    internal static Tensor<T> BatchTransposeMatMul<T>(Tensor<T> a, Tensor<T> b) =>
        Engine.TensorBatchMatMul(Engine.TensorSwapAxes(a, 1, 2), b);

    /// <summary>
    /// Solves a[i] x[i] = b[i] for every i of a batch of square systems ([n, k, k] with [n, k]) by LU with
    /// partial pivoting. The engine has no batched solver, so the systems are solved one by one.
    /// </summary>
    internal static Tensor<T> SolveEach<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = Engine;
        var systems = engine.TensorUnstack(a, 0);
        var rhs = engine.TensorUnstack(b, 0);
        var solutions = new Tensor<T>[systems.Length];
        for (int i = 0; i < systems.Length; i++)
            solutions[i] = Solve(systems[i], rhs[i]);
        return engine.TensorStack(solutions, 0);
    }

    /// <summary>
    /// Symmetric eigen-decomposition of every matrix in an [n, k, k] batch, eigenvalues in the requested order with
    /// ties in index order. Returns ([n, k] values, [n, k, k] vectors as columns).
    /// </summary>
    internal static (Tensor<T> Values, Tensor<T> Vectors) SymmetricEigenEach<T>(Tensor<T> a, bool descending)
    {
        var engine = Engine;
        var matrices = engine.TensorUnstack(a, 0);
        var values = new Tensor<T>[matrices.Length];
        var vectors = new Tensor<T>[matrices.Length];
        for (int i = 0; i < matrices.Length; i++)
            (values[i], vectors[i]) = SymmetricEigen(matrices[i], descending);
        return (engine.TensorStack(values, 0), engine.TensorStack(vectors, 0));
    }

    /// <summary>Per-matrix trace of an [n, k, k] batch.</summary>
    internal static Tensor<T> BatchTrace<T>(Tensor<T> a)
    {
        int k = a.Shape[1];
        var identity = Engine.Reshape(Identity<T>(k), new[] { 1, k, k });
        return Engine.ReduceSum(Engine.TensorMultiply(a, identity), new[] { 1, 2 }, false);
    }
}
