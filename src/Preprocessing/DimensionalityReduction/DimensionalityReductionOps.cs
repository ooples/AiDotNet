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
    /// Row-major [rows, columns] standard normal draws by Box-Muller, consuming two NextDouble values per entry
    /// (u1 then u2, z = sqrt(-2 ln u1) cos(2 pi u2)).
    /// </summary>
    internal static Tensor<T> StandardNormal<T>(Random random, int rows, int columns)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var draws = engine.Reshape(Uniform<T>(random, rows * columns, 2), new[] { rows * columns, 2 });
        var u1 = engine.TensorSliceAxis(draws, 1, 0);
        var u2 = engine.TensorSliceAxis(draws, 1, 1);
        var radius = engine.TensorSqrt(engine.TensorMultiplyScalar(engine.TensorLog(u1), numOps.FromDouble(-2.0)));
        var angle = engine.TensorCos(engine.TensorMultiplyScalar(u2, numOps.FromDouble(2.0 * Math.PI)));
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

    /// <summary>Converts a scalar to int by truncation toward zero.</summary>
    internal static int ToInt<T>(T value) => (int)MathHelper.GetNumericOperations<T>().ToDouble(value);
}
