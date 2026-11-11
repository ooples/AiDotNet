using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// Whole-array engine building blocks for the regression models (index tensors, scatter/gather,
/// row selection), so the models do not walk vectors and matrices one element at a time.
/// </summary>
internal static class RegressionEngineOps2
{
    /// <summary>
    /// Wraps integer positions as a rank-1 <see cref="Tensor{T}"/> of <see cref="int"/> for the
    /// engine's index operations.
    /// </summary>
    public static Tensor<int> IndexTensor(IReadOnlyList<int> indices)
    {
        var data = new int[indices.Count];
        if (indices is int[] array)
        {
            Array.Copy(array, data, data.Length);
        }
        else if (indices is List<int> list)
        {
            list.CopyTo(data);
        }
        else
        {
            data = indices.ToArray();
        }

        return new Tensor<int>(data, new[] { data.Length });
    }

    /// <summary>
    /// A length-<paramref name="length"/> vector of zeros with <c>values[k]</c> written at
    /// position <c>indices[k]</c> (later duplicates win).
    /// </summary>
    public static Vector<T> ScatterToLength<T>(int length, IReadOnlyList<int> indices, Vector<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var zeros = engine.FillZero<T>(length);
        if (indices.Count == 0)
        {
            return zeros;
        }

        return engine.TensorIndexCopy(zeros.AsTensor(), 0, IndexTensor(indices), values.AsTensor()).ToVector();
    }

    /// <summary>
    /// The elements of <paramref name="values"/> at <paramref name="indices"/>, in that order.
    /// </summary>
    public static Vector<T> Gather<T>(Vector<T> values, IReadOnlyList<int> indices)
    {
        if (indices.Count == 0)
        {
            return new Vector<T>(0);
        }

        return AiDotNetEngine.Current.TensorIndexSelect(values.AsTensor(), IndexTensor(indices), 0).ToVector();
    }

    /// <summary>
    /// The rows of <paramref name="matrix"/> at <paramref name="indices"/>, in that order.
    /// </summary>
    public static Matrix<T> GatherRows<T>(Matrix<T> matrix, IReadOnlyList<int> indices)
    {
        if (indices.Count == 0)
        {
            return new Matrix<T>(0, matrix.Columns);
        }

        return AiDotNetEngine.Current
            .TensorIndexSelect(Tensor<T>.FromMatrix(matrix), IndexTensor(indices), 0).ToMatrix();
    }

    /// <summary>
    /// <paramref name="matrix"/> + diag(<paramref name="diagonal"/>) for a square matrix.
    /// </summary>
    public static Matrix<T> AddDiagonal<T>(Matrix<T> matrix, Vector<T> diagonal)
    {
        return AiDotNetEngine.Current.MatrixAdd(matrix, Matrix<T>.CreateDiagonal(diagonal));
    }

    /// <summary>
    /// <paramref name="matrix"/> + <paramref name="value"/> * I for a square matrix.
    /// </summary>
    public static Matrix<T> AddDiagonal<T>(Matrix<T> matrix, T value)
    {
        return AddDiagonal(matrix, AiDotNetEngine.Current.Fill(matrix.Rows, value));
    }

    /// <summary>
    /// Scale-adaptive ridge jitter: adds max(|d_i| * <paramref name="relative"/>, <paramref name="floor"/>)
    /// to each diagonal entry d_i of a square matrix.
    /// </summary>
    public static Matrix<T> AddRelativeDiagonalJitter<T>(Matrix<T> matrix, T relative, T floor)
    {
        var engine = AiDotNetEngine.Current;
        var diagonal = matrix.Diagonal();
        var jitter = engine.Max(
            engine.Multiply(engine.Abs(diagonal), relative),
            engine.Fill(diagonal.Length, floor));
        return AddDiagonal(matrix, jitter);
    }

    /// <summary>
    /// intercept + X[:, :k] * coefficients[:k] with k = min(columns, coefficient count).
    /// </summary>
    public static Vector<T> LinearPredict<T>(Matrix<T> x, Vector<T> coefficients, T intercept)
    {
        var engine = AiDotNetEngine.Current;
        int k = Math.Min(x.Columns, coefficients.Length);
        var xUsed = k == x.Columns ? x : x.GetSubMatrix(0, 0, x.Rows, k);
        var cUsed = k == coefficients.Length ? coefficients : coefficients.GetSubVector(0, k);
        var linear = k == 0 ? engine.FillZero<T>(x.Rows) : engine.MatrixVectorMultiply(xUsed, cUsed);
        return engine.Add(engine.Fill(x.Rows, intercept), linear);
    }

    /// <summary>
    /// intercept + x[:k] . coefficients[:k] with k = min(lengths).
    /// </summary>
    public static T LinearPredictSingle<T>(Vector<T> x, Vector<T> coefficients, T intercept)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int k = Math.Min(x.Length, coefficients.Length);
        if (k == 0)
        {
            return intercept;
        }

        var xUsed = k == x.Length ? x : x.GetSubVector(0, k);
        var cUsed = k == coefficients.Length ? coefficients : coefficients.GetSubVector(0, k);
        return numOps.Add(intercept, AiDotNetEngine.Current.DotProduct(xUsed, cUsed));
    }

    /// <summary>
    /// True when every element of <paramref name="values"/> is finite (no NaN, no infinity).
    /// </summary>
    public static bool AllFinite<T>(Vector<T> values)
    {
        if (values.Length == 0)
        {
            return true;
        }

        var engine = AiDotNetEngine.Current;
        var tensor = values.AsTensor();
        var notFinite = engine.TensorWhere(
            engine.TensorIsFinite(tensor), engine.TensorZerosLike(tensor), engine.TensorOnesLike(tensor));
        return engine.TensorCountNonzero(notFinite) == 0;
    }

    /// <summary>
    /// Element-wise max(<paramref name="values"/>, <paramref name="floor"/>).
    /// </summary>
    public static Vector<T> ClampMin<T>(Vector<T> values, T floor)
    {
        return AiDotNetEngine.Current.TensorMax(values.AsTensor(), floor).ToVector();
    }

    /// <summary>
    /// <paramref name="values"/> + <paramref name="scalar"/> element-wise.
    /// </summary>
    public static Vector<T> AddScalar<T>(Vector<T> values, T scalar)
    {
        return AiDotNetEngine.Current.TensorAddScalar(values.AsTensor(), scalar).ToVector();
    }

    /// <summary>
    /// <paramref name="values"/> with <paramref name="last"/> appended.
    /// </summary>
    public static Vector<T> Append<T>(Vector<T> values, T last)
    {
        return Vector<T>.Concatenate(values, new Vector<T>(new[] { last }));
    }

    /// <summary>
    /// Euclidean distances ||left[i] - right[j]|| as a [left.Rows, right.Rows] matrix: the square
    /// root of the per-feature squared differences summed, the same formula as
    /// <c>VectorHelper.EuclideanDistance</c> (no ||a||^2 + ||b||^2 - 2ab expansion, which loses
    /// precision). Left rows are processed in blocks so the difference tensor stays bounded.
    /// </summary>
    public static Matrix<T> PairwiseDistances<T>(Matrix<T> left, Matrix<T> right)
    {
        if (left.Columns != right.Columns)
        {
            throw new ArgumentException(
                $"Vectors must have the same length. Got {left.Columns} and {right.Columns}.", nameof(right));
        }

        if (left.Rows == 0 || right.Rows == 0 || left.Columns == 0)
        {
            return new Matrix<T>(left.Rows, right.Rows);
        }

        var engine = AiDotNetEngine.Current;
        int d = left.Columns;
        long perRow = Math.Max(1L, (long)right.Rows * Math.Max(1, d));
        int blockRows = (int)Math.Max(1L, Math.Min(left.Rows, PairwiseBlockElements / perRow));
        var rightBroadcastSource = engine.Reshape(Tensor<T>.FromMatrix(right), new[] { 1, right.Rows, d });
        var blocks = new List<Tensor<T>>();
        for (int start = 0; start < left.Rows; start += blockRows)
        {
            int count = Math.Min(blockRows, left.Rows - start);
            var block = count == left.Rows ? left : left.GetSubMatrix(start, 0, count, d);
            var shape = new[] { count, right.Rows, d };
            var l = engine.TensorBroadcastTo(engine.Reshape(Tensor<T>.FromMatrix(block), new[] { count, 1, d }), shape);
            var r = engine.TensorBroadcastTo(rightBroadcastSource, shape);
            var squared = engine.TensorSquare(engine.TensorSubtract(l, r));
            blocks.Add(engine.ReduceSum(squared, new[] { 2 }, keepDims: false));
        }

        var sumSq = blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), 0);
        return engine.TensorSqrt(sumSq).ToMatrix();
    }

    private const long PairwiseBlockElements = 1L << 24;

    /// <summary>
    /// A one-row matrix holding <paramref name="row"/>.
    /// </summary>
    public static Matrix<T> AsRowMatrix<T>(Vector<T> row)
    {
        return new Tensor<T>(new[] { 1, row.Length }, row.Clone()).ToMatrix();
    }

    /// <summary>
    /// Sum over columns of the population variance of each column, computed two-pass
    /// (mean first, then squared deviations from it).
    /// </summary>
    public static T TotalColumnVariance<T>(Matrix<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        if (x.Columns == 0)
        {
            return numOps.Zero;
        }

        var xt = Tensor<T>.FromMatrix(x);
        var means = engine.ReduceMean(xt, new[] { 0 }, keepDims: true);
        var centered = engine.TensorSubtract(xt, engine.TensorBroadcastTo(means, new[] { x.Rows, x.Columns }));
        var variances = engine.TensorDivideScalar(
            engine.ReduceSum(engine.TensorSquare(centered), new[] { 0 }, keepDims: false),
            numOps.FromDouble(x.Rows));
        return engine.TensorSum(variances);
    }

    /// <summary>
    /// Largest absolute value, or zero for an empty vector.
    /// </summary>
    public static T MaxAbs<T>(Vector<T> values)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (values.Length == 0)
        {
            return numOps.Zero;
        }

        return AiDotNetEngine.Current.TensorMaxValue(AiDotNetEngine.Current.TensorAbs(values.AsTensor()));
    }

    /// <summary>
    /// Sum of squares of <paramref name="values"/>, or zero for an empty vector.
    /// </summary>
    public static T SumOfSquares<T>(Vector<T> values)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (values.Length == 0)
        {
            return numOps.Zero;
        }

        return AiDotNetEngine.Current.DotProduct(values, values);
    }

    /// <summary>
    /// The kernel matrix K[i, j] = k(left[i], right[j]) for the non-linear regression kernels, with the
    /// same formulas as <c>NonLinearRegressionBase.KernelFunction</c>.
    /// </summary>
    public static Matrix<T> KernelMatrix<T>(Matrix<T> left, Matrix<T> right, NonLinearRegressionOptions options)
    {
        if (left.Rows == 0 || right.Rows == 0)
        {
            return new Matrix<T>(left.Rows, right.Rows);
        }

        if (left.Columns != right.Columns)
        {
            throw new ArgumentException(
                $"Vectors must have the same length. Got {right.Columns} and {left.Columns}.", nameof(right));
        }

        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        bool distanceKernel = options.KernelType is KernelType.RBF or KernelType.Laplacian;
        long perRow = Math.Max(1L, (long)right.Rows * Math.Max(1, left.Columns));
        int blockRows = distanceKernel
            ? (int)Math.Max(1L, Math.Min(left.Rows, PairwiseBlockElements / perRow))
            : left.Rows;
        if (blockRows < left.Rows)
        {
            var blocks = new List<Tensor<T>>();
            for (int start = 0; start < left.Rows; start += blockRows)
            {
                int count = Math.Min(blockRows, left.Rows - start);
                blocks.Add(Tensor<T>.FromMatrix(
                    KernelMatrix(left.GetSubMatrix(start, 0, count, left.Columns), right, options)));
            }

            return engine.TensorConcatenate(blocks.ToArray(), 0).ToMatrix();
        }

        T gamma = numOps.FromDouble(options.Gamma);
        T negativeGamma = numOps.FromDouble(-options.Gamma);
        T coef0 = numOps.FromDouble(options.Coef0);
        Tensor<T> gram;
        switch (options.KernelType)
        {
            case KernelType.Linear:
                gram = CrossDotProducts(left, right);
                break;

            case KernelType.RBF:
                gram = engine.TensorExp(engine.TensorMultiplyScalar(
                    ReduceLastAxis(engine.TensorSquare(PairwiseDifferences(left, right)), left.Rows, right.Rows),
                    negativeGamma));
                break;

            case KernelType.Polynomial:
                gram = engine.TensorPower(
                    engine.TensorAddScalar(engine.TensorMultiplyScalar(CrossDotProducts(left, right), gamma), coef0),
                    numOps.FromDouble(options.PolynomialDegree));
                break;

            case KernelType.Sigmoid:
                gram = engine.TensorTanh(
                    engine.TensorAddScalar(engine.TensorMultiplyScalar(CrossDotProducts(left, right), gamma), coef0));
                break;

            case KernelType.Laplacian:
                gram = engine.TensorExp(engine.TensorMultiplyScalar(
                    ReduceLastAxis(engine.TensorAbs(PairwiseDifferences(left, right)), left.Rows, right.Rows),
                    negativeGamma));
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(options.KernelType), options.KernelType, "Unsupported kernel type");
        }

        return gram.ToMatrix();
    }

    /// <summary>
    /// A square kernel matrix that is exactly symmetric: the lower triangle (including the diagonal)
    /// mirrored onto the upper one, as a scan that fills K[i, j] for j &lt;= i and copies it to K[j, i].
    /// </summary>
    public static Matrix<T> SymmetricFromLower<T>(Matrix<T> square)
    {
        var engine = AiDotNetEngine.Current;
        var t = Tensor<T>.FromMatrix(square);
        var lower = engine.TensorTril(t, 0);
        var strictlyLower = engine.TensorTril(t, -1);
        return engine.TensorAdd(lower, engine.TensorTranspose(strictlyLower)).ToMatrix();
    }

    private static Tensor<T> PairwiseDifferences<T>(Matrix<T> left, Matrix<T> right)
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

    private static Tensor<T> ReduceLastAxis<T>(Tensor<T> values, int rows, int columns)
    {
        if (values.Length == 0)
        {
            return new Tensor<T>(new[] { rows, columns });
        }

        return AiDotNetEngine.Current.ReduceSum(values, new[] { 2 }, keepDims: false);
    }

    private static Tensor<T> CrossDotProducts<T>(Matrix<T> left, Matrix<T> right)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorMatMul(Tensor<T>.FromMatrix(left), Tensor<T>.FromMatrix(engine.MatrixTranspose(right)));
    }

    /// <summary>
    /// The smallest and largest elements (the vector must not be empty).
    /// </summary>
    public static (T Min, T Max) MinMax<T>(Vector<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var tensor = values.AsTensor();
        return (engine.TensorMinValue(tensor), engine.TensorMaxValue(tensor));
    }

    /// <summary>
    /// Population standard deviation sqrt(mean((v - mean(v))^2)), two-pass.
    /// </summary>
    public static T PopulationStdDev<T>(Vector<T> values)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        T mean = engine.Mean(values);
        T sumSq = SumOfSquares(AddScalar(values, numOps.Negate(mean)));
        return numOps.Sqrt(numOps.Divide(sumSq, numOps.FromDouble(values.Length)));
    }

    /// <summary>
    /// Per-feature powers: column j * degree + d holds x[:, j]^(d + 1) for d = 0 .. degree - 1.
    /// </summary>
    public static Matrix<T> PolynomialExpand<T>(Matrix<T> x, int degree)
    {
        int rows = x.Rows;
        int cols = x.Columns * degree;
        if (degree <= 0 || rows == 0 || x.Columns == 0)
        {
            return new Matrix<T>(rows, cols);
        }

        // Powers on a trailing axis: bases x broadcast to [rows, columns, degree] against the
        // exponents 1 .. degree, then the trailing axis is flattened into the column axis.
        var engine = AiDotNetEngine.Current;
        var shape = new[] { rows, x.Columns, degree };
        var bases = engine.TensorBroadcastTo(
            engine.Reshape(Tensor<T>.FromMatrix(x), new[] { rows, x.Columns, 1 }), shape);
        var exponents = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorArange<T>(1, degree + 1, 1), new[] { 1, 1, degree }), shape);
        return engine.Reshape(engine.TensorPower(bases, exponents), new[] { rows, cols }).ToMatrix();
    }

    /// <summary>
    /// Inverse of a square matrix by Gauss-Jordan elimination with partial pivoting: each column
    /// takes the first row with the largest |value| at or below the diagonal as pivot, a pivot
    /// smaller than <paramref name="pivotFloor"/> in magnitude is replaced by
    /// <paramref name="pivotFloor"/>, and each step eliminates the column from every other row
    /// with one rank-1 update.
    /// </summary>
    public static Matrix<T> GaussJordanInverse<T>(Matrix<T> a, T pivotFloor)
    {
        var engine = AiDotNetEngine.Current;
        int n = a.Rows;
        if (n == 0)
        {
            return new Matrix<T>(0, 0);
        }

        var augmented = engine.TensorConcatenate(
            new[] { Tensor<T>.FromMatrix(a), engine.TensorEye<T>(n) }, 1).ToMatrix();
        return GaussJordanEliminate(augmented, n, pivotFloor).GetSubMatrix(0, n, n, n);
    }

    /// <summary>
    /// Solves a x = b for the leading n x n block of <paramref name="a"/> by Gauss-Jordan
    /// elimination on [a | b], with the same pivoting and pivot floor as
    /// <see cref="GaussJordanInverse{T}(Matrix{T}, T)"/>.
    /// </summary>
    public static Vector<T> GaussJordanSolve<T>(Matrix<T> a, Vector<T> b, int n, T pivotFloor)
    {
        var engine = AiDotNetEngine.Current;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        var block = a.Rows == n && a.Columns == n ? a : a.GetSubMatrix(0, 0, n, n);
        var rhs = b.Length == n ? b : b.GetSubVector(0, n);
        var augmented = engine.TensorConcatenate(
            new[] { Tensor<T>.FromMatrix(block), new Tensor<T>(new[] { n, 1 }, rhs.Clone()) }, 1).ToMatrix();
        return GaussJordanEliminate(augmented, n, pivotFloor).GetColumn(n);
    }

    /// <summary>
    /// Gauss-Jordan elimination of the leading n columns of an augmented matrix: each column takes
    /// the first row with the largest |value| at or below the diagonal as pivot (a strict-greater
    /// scan), a pivot below <paramref name="pivotFloor"/> in magnitude is replaced by
    /// <paramref name="pivotFloor"/>, the pivot row is scaled and the column is eliminated from
    /// every other row with one rank-1 update.
    /// </summary>
    private static Matrix<T> GaussJordanEliminate<T>(Matrix<T> augmented, int n, T pivotFloor)
    {
        var engine = AiDotNetEngine.Current;
        int width = augmented.Columns;

        for (int col = 0; col < n; col++)
        {
            // Pivot: first row at or below col with the largest |value| in this column.
            var column = augmented.GetColumn(col);
            var candidates = engine.Abs(column.GetSubVector(col, n - col));
            int maxRow = col + engine.TensorArgMax(candidates.AsTensor(), 0).ToArray()[0];
            if (maxRow != col)
            {
                var pivotCandidate = augmented.GetRow(maxRow);
                augmented.SetRow(maxRow, augmented.GetRow(col));
                augmented.SetRow(col, pivotCandidate);
                column = augmented.GetColumn(col);
            }

            // Scale the pivot row by its pivot, a pivot below the floor in magnitude becoming the floor.
            var pivotRow = augmented.GetRow(col);
            var pivot = pivotRow.GetSubVector(col, 1).AsTensor();
            var safePivot = engine.TensorWhere(
                engine.TensorLessThan(engine.TensorAbs(pivot), pivotFloor),
                engine.Fill(1, pivotFloor).AsTensor(),
                pivot);
            var scaledPivotRow = engine.TensorDivide(
                pivotRow.AsTensor(), engine.TensorBroadcastTo(safePivot, new[] { width })).ToVector();

            // row -= row[col] * scaledPivotRow for every row != col
            var factors = Vector<T>.Concatenate(
                column.GetSubVector(0, col), engine.FillZero<T>(1), column.GetSubVector(col + 1, n - col - 1));
            var update = engine.MatrixMultiply(
                new Tensor<T>(new[] { n, 1 }, factors).ToMatrix(),
                new Tensor<T>(new[] { 1, width }, scaledPivotRow).ToMatrix());
            augmented = engine.MatrixSubtract(augmented, update);
            augmented.SetRow(col, scaledPivotRow);
        }

        return augmented;
    }

    /// <summary>
    /// The m x m masks whose row j is all ones except a zero at position j.
    /// </summary>
    public static Matrix<T> LeaveOneOutMasks<T>(int m)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        if (m == 0)
        {
            return new Matrix<T>(0, 0);
        }

        return engine.ScalarMinusTensor(numOps.One, engine.TensorEye<T>(m)).ToMatrix();
    }

    /// <summary>
    /// Column-wise standardization statistics: population mean and standard deviation per column
    /// (two-pass), with standard deviations below <paramref name="floor"/> replaced by one.
    /// </summary>
    public static (Vector<T> Means, Vector<T> Stds) ColumnStandardization<T>(Matrix<T> x, T floor)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var xt = Tensor<T>.FromMatrix(x);
        var means = engine.ReduceMean(xt, new[] { 0 }, keepDims: true);
        var centered = engine.TensorSubtract(xt, engine.TensorBroadcastTo(means, new[] { x.Rows, x.Columns }));
        var stds = engine.TensorSqrt(engine.TensorDivideScalar(
            engine.ReduceSum(engine.TensorSquare(centered), new[] { 0 }, keepDims: false),
            numOps.FromDouble(x.Rows)));
        var tiny = engine.TensorLessThan(stds, floor);
        stds = engine.TensorWhere(tiny, engine.TensorOnesLike(stds), stds);
        return (engine.Reshape(means, new[] { x.Columns }).ToVector(), stds.ToVector());
    }

    /// <summary>
    /// (x - means) / stds per column.
    /// </summary>
    public static Matrix<T> Standardize<T>(Matrix<T> x, Vector<T> means, Vector<T> stds)
    {
        var engine = AiDotNetEngine.Current;
        var shape = new[] { x.Rows, x.Columns };
        var m = engine.TensorBroadcastTo(new Tensor<T>(new[] { 1, means.Length }, means), shape);
        var s = engine.TensorBroadcastTo(new Tensor<T>(new[] { 1, stds.Length }, stds), shape);
        return engine.TensorDivide(engine.TensorSubtract(Tensor<T>.FromMatrix(x), m), s).ToMatrix();
    }

    /// <summary>
    /// Index of the first non-zero element of <paramref name="mask"/>, or -1 when there is none.
    /// </summary>
    public static int FirstNonzero<T>(Tensor<T> mask)
    {
        if (mask.Length == 0)
        {
            return -1;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var flat = mask.Rank == 1 ? mask : mask.Reshape(new[] { mask.Length });
        var notZero = AiDotNetEngine.Current.TensorNotEqualMask(flat, numOps);
        int first = AiDotNetEngine.Current.TensorArgMax(notZero, 0)[0];
        return numOps.Equals(notZero[first], numOps.Zero) ? -1 : first;
    }

    private static Tensor<T> TensorNotEqualMask<T>(this IEngine engine, Tensor<T> tensor, INumericOperations<T> numOps)
    {
        // 1 where tensor != 0 (NaN counts as non-zero, like !Equals(x, 0)), else 0.
        return engine.ScalarMinusTensor(numOps.One, engine.TensorEquals(tensor, numOps.Zero));
    }
}
