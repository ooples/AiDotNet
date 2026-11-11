namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Base class for matrix decomposition algorithms that break down matrices into simpler components.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Matrix decomposition is like factoring a number, but for matrices.
/// Just as we can break down 12 into 3 * 4, we can break down complex matrices into simpler
/// matrices that are easier to work with. Different decomposition methods have different strengths
/// and are used for different purposes in machine learning and numerical computing.
/// </para>
/// <para>
/// This base class provides common functionality that all matrix decompositions need, such as:
/// - Storing the original matrix
/// - Providing numeric operations
/// - Implementing matrix inversion
/// - Solving linear systems
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
public abstract class MatrixDecompositionBase<T> : IMatrixDecomposition<T>
{
    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper object knows how to perform math operations
    /// on the specific number type you're using (like float or double). It ensures
    /// that mathematical operations work correctly regardless of the numeric type.
    /// </para>
    /// </remarks>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// The original matrix that was decomposed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the matrix you started with before breaking it down
    /// into simpler components. It's kept here so you can refer back to it if needed.
    /// </para>
    /// </remarks>
    public Matrix<T> A { get; protected set; }

    /// <summary>
    /// Initializes a new instance of the matrix decomposition class.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the common parts that all matrix
    /// decompositions need. It stores the original matrix and initializes the numeric
    /// operations helper.
    /// </para>
    /// </remarks>
    protected MatrixDecompositionBase(Matrix<T> matrix)
    {
        Guard.NotNull(matrix);
        A = matrix;
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Performs the actual decomposition of the matrix into its components.
    /// This method must be implemented by derived classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where the actual "work" of breaking down the matrix happens.
    /// Each specific decomposition method (LU, QR, SVD, NMF, ICA, etc.) has its own unique
    /// algorithm, which is why each derived class implements this method differently.
    /// </para>
    /// </remarks>
    protected abstract void Decompose();

    /// <summary>
    /// Solves a linear system of equations Ax = b, where A is the decomposed matrix.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method solves equations of the form "Ax = b" where:
    /// - A is your matrix (the one that was decomposed)
    /// - b is a known vector (like a set of results or outcomes)
    /// - x is what you're trying to find (like the unknown variables)
    /// </para>
    /// <para>
    /// Real-world example: If you're calculating the best mix of ingredients for a recipe:
    /// - Matrix A represents how each ingredient affects taste, texture, and nutrition
    /// - Vector b represents your desired taste, texture, and nutrition targets
    /// - Vector x (the solution) tells you how much of each ingredient to use
    /// </para>
    /// <para>
    /// Using the decomposition to solve the system is much faster and more accurate
    /// than directly calculating the inverse of A.
    /// </para>
    /// </remarks>
    public abstract Vector<T> Solve(Vector<T> b);

    /// <summary>
    /// Calculates the inverse of the original matrix A using the decomposition.
    /// </summary>
    /// <returns>The inverse of the matrix A.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The inverse of a matrix is like the reciprocal of a number.
    /// Just as 5 * (1/5) = 1, a matrix multiplied by its inverse gives the identity matrix
    /// (which is like the number 1 in matrix form).
    /// </para>
    /// <para>
    /// For example, if matrix A represents a transformation (like rotating or scaling),
    /// then A-¹ represents the opposite transformation that "undoes" the original:
    /// - If A rotates data clockwise, A-¹ rotates it counterclockwise
    /// - If A scales data up by 2x, A-¹ scales it down by 1/2
    /// </para>
    /// <para>
    /// This default implementation uses the decomposition to compute the inverse efficiently.
    /// Some derived classes may override this method if they have a more efficient
    /// algorithm specific to their decomposition type.
    /// </para>
    /// </remarks>
    public virtual Matrix<T> Invert()
    {
        return MatrixHelper<T>.InvertUsingDecomposition(this);
    }

    /// <summary>
    /// Validates that the matrix meets the requirements for this decomposition method.
    /// </summary>
    /// <param name="matrix">The matrix to validate.</param>
    /// <param name="requireSquare">Whether the matrix must be square.</param>
    /// <param name="requireNonNegative">Whether all elements must be non-negative.</param>
    /// <param name="requireSymmetric">Whether the matrix must be symmetric.</param>
    /// <exception cref="ArgumentException">Thrown when the matrix doesn't meet the requirements.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different decomposition methods have different requirements.
    /// For example:
    /// - Some methods only work on square matrices (same number of rows and columns)
    /// - Some require all values to be non-negative (zero or positive)
    /// - Some require the matrix to be symmetric (mirror image across the diagonal)
    /// </para>
    /// <para>
    /// This helper method checks these requirements and gives a clear error message
    /// if something is wrong, rather than producing incorrect results.
    /// </para>
    /// </remarks>
    protected void ValidateMatrix(
        Matrix<T> matrix,
        bool requireSquare = false,
        bool requireNonNegative = false,
        bool requireSymmetric = false)
    {
        if (requireSquare && matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException(
                $"Matrix must be square for this decomposition. Got {matrix.Rows}x{matrix.Columns} matrix.",
                nameof(matrix));
        }

        if (requireNonNegative && matrix.Rows > 0 && matrix.Columns > 0)
        {
            var values = Tensor<T>.FromMatrix(matrix);
            if (NumOps.LessThan(Engine.TensorMinValue(values), NumOps.Zero))
            {
                // Report the first negative entry in row-major order, as the scan did.
                var negativeMask = Engine.TensorLessThan(values.Reshape(new[] { values.Length }), NumOps.Zero);
                int flat = Engine.TensorArgMax(negativeMask, 0)[0];
                int row = flat / matrix.Columns;
                int col = flat % matrix.Columns;
                throw new ArgumentException(
                    $"Matrix must contain only non-negative values for this decomposition. " +
                    $"Found negative value {matrix[row, col]} at position ({row}, {col}).",
                    nameof(matrix));
            }
        }

        if (requireSymmetric)
        {
            if (matrix.Rows != matrix.Columns)
            {
                throw new ArgumentException(
                    "Matrix must be square to be symmetric.",
                    nameof(matrix));
            }

            if (matrix.Rows == 0)
            {
                return;
            }

            T tolerance = NumOps.FromDouble(1e-10);
            var values = Tensor<T>.FromMatrix(matrix);
            var asymmetry = Engine.TensorAbs(Engine.TensorSubtract(values, Engine.TensorTranspose(values)));
            if (NumOps.GreaterThan(Engine.TensorMaxValue(asymmetry), tolerance))
            {
                // First offending (i, j) with j > i in row-major order, as the scan reported it.
                var violations = Engine.TensorTriu(Engine.TensorGreaterThan(asymmetry, tolerance), 1);
                int flat = Engine.TensorArgMax(violations.Reshape(new[] { violations.Length }), 0)[0];
                int i = flat / matrix.Columns;
                int j = flat % matrix.Columns;
                T diff = asymmetry[i, j];
                throw new ArgumentException(
                    $"Matrix must be symmetric for this decomposition. " +
                    $"Elements at ({i}, {j}) and ({j}, {i}) differ by {diff}.",
                    nameof(matrix));
            }
        }
    }

    /// <summary>
    /// Computes the Frobenius norm of a matrix (square root of sum of squared elements).
    /// </summary>
    /// <param name="matrix">The matrix to compute the norm for.</param>
    /// <returns>The Frobenius norm of the matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Frobenius norm is a way to measure the "size" or "magnitude"
    /// of a matrix. It's like measuring the length of a vector, but for a matrix.
    /// </para>
    /// <para>
    /// It's calculated by:
    /// 1. Squaring every element in the matrix
    /// 2. Adding all those squares together
    /// 3. Taking the square root of that sum
    /// </para>
    /// <para>
    /// This is useful for measuring how different two matrices are (by computing the norm
    /// of their difference) or for checking convergence in iterative algorithms.
    /// </para>
    /// </remarks>
    protected T FrobeniusNorm(Matrix<T> matrix)
    {
        return NumOps.Sqrt(Engine.MatrixSumOfSquares(matrix));
    }

    /// <summary>
    /// Unpivoted symmetric factorization A = L D L^T (L unit lower triangular, D diagonal) computed
    /// with right-looking whole-block engine updates: for each pivot j, the column below the pivot
    /// is scaled by 1 / D[j] and the trailing block receives the rank-1 update l c^T.
    /// </summary>
    /// <param name="matrix">The symmetric matrix to factor (only read, never modified).</param>
    /// <returns>The unit lower triangular factor L and the diagonal D.</returns>
    /// <remarks>
    /// This is the same elimination as the left-looking dot-product form
    /// (L[i,j] = (A[i,j] - sum_k L[i,k] L[j,k] D[k]) / D[j]); it works for indefinite matrices with
    /// non-zero leading minors and, like that form, divides by a zero pivot rather than throwing.
    /// </remarks>
    protected (Matrix<T> L, Vector<T> D) ComputeUnpivotedLdl(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n == 0)
        {
            return (new Matrix<T>(0, 0), new Vector<T>(0));
        }

        var work = Tensor<T>.FromMatrix(matrix.Clone());
        var lower = Engine.TensorEye<T>(n);
        var diagonal = new Tensor<T>(new[] { n });

        for (int j = 0; j < n; j++)
        {
            var pivot = Engine.TensorSlice(work, new[] { j, j }, new[] { 1, 1 });
            diagonal = Engine.TensorSetSlice(diagonal, pivot.Reshape(new[] { 1 }), new[] { j });

            int rest = n - j - 1;
            if (rest == 0)
            {
                break;
            }

            T inversePivot = Engine.TensorSum(Engine.TensorReciprocal(pivot));
            var column = Engine.TensorSlice(work, new[] { j + 1, j }, new[] { rest, 1 });
            var multipliers = Engine.TensorMultiplyScalar(column, inversePivot);
            lower = Engine.TensorSetSlice(lower, multipliers, new[] { j + 1, j });

            var trailing = Engine.TensorSlice(work, new[] { j + 1, j + 1 }, new[] { rest, rest });
            trailing = Engine.TensorSubtract(trailing, Engine.TensorMatMul(multipliers, Engine.TensorTranspose(column)));
            work = Engine.TensorSetSlice(work, trailing, new[] { j + 1, j + 1 });
        }

        return (lower.ToMatrix(), diagonal.ToVector());
    }

    /// <summary>
    /// Solves (L D L^T) x = b with two unit-triangular engine solves and one diagonal scaling.
    /// </summary>
    protected static Matrix<T> SolveLdlSystem(Matrix<T> unitLower, Vector<T> diagonal, Matrix<T> rhs)
    {
        var engine = AiDotNetEngine.Current;
        var l = Tensor<T>.FromMatrix(unitLower);
        var y = Linalg.SolveTriangular(l, Tensor<T>.FromMatrix(rhs), upper: false, unitDiagonal: true);
        var d = engine.TensorBroadcastTo(
            Tensor<T>.FromVector(diagonal, new[] { diagonal.Length, 1 }),
            new[] { rhs.Rows, rhs.Columns });
        y = engine.TensorMultiply(y, engine.TensorReciprocal(d));
        return Linalg.SolveTriangular(engine.TensorTranspose(l), y, upper: true, unitDiagonal: true).ToMatrix();
    }

    /// <summary>
    /// Orthogonal similarity reduction H = Q^T A Q by Householder reflectors applied to the columns
    /// below the subdiagonal (upper Hessenberg form; tridiagonal for symmetric A), with Q e1 = e1.
    /// </summary>
    /// <param name="matrix">The square matrix to reduce (not modified).</param>
    /// <returns>H (entries below the subdiagonal are rounding residue) and the orthogonal Q.</returns>
    /// <remarks>
    /// Step k reflects x = H[k+1:, k] onto alpha e1 with alpha = -sign(x0) ‖x‖ (sign(0) = +1),
    /// using u = (x - alpha e1) / ‖x - alpha e1‖; steps whose ‖x‖ or ‖u‖ is below 1e-14 are
    /// skipped. Each reflection is applied as two rank-1 whole-matrix engine updates.
    /// </remarks>
    protected (Matrix<T> H, Matrix<T> Q) ReduceByHouseholderSimilarity(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var h = Tensor<T>.FromMatrix(matrix.Clone());
        var q = Engine.TensorEye<T>(n);
        T tiny = NumOps.FromDouble(1e-14);
        T two = NumOps.FromDouble(2);
        T zero = NumOps.Zero;

        for (int k = 0; k < n - 2; k++)
        {
            int length = n - k - 1;
            var x = Engine.TensorSlice(h, new[] { k + 1, k }, new[] { length, 1 });
            var xNorm = Engine.TensorNorm(x).Reshape(new[] { 1, 1 });
            if (AnyBelow(xNorm, tiny))
            {
                continue;
            }

            // u0 = x0 - alpha = x0 + sign(x0) ‖x‖, sign(x0) = -1 for x0 < 0 and +1 otherwise.
            var x0 = Engine.TensorSlice(x, new[] { 0, 0 }, new[] { 1, 1 });
            var negative = Engine.TensorLessThan(x0, zero);
            var sign = Engine.TensorSubtract(Engine.TensorOnesLike(negative), Engine.TensorMultiplyScalar(negative, two));
            var u = Engine.TensorSetSlice(x, Engine.TensorAdd(x0, Engine.TensorMultiply(sign, xNorm)), new[] { 0, 0 });

            var uNorm = Engine.TensorNorm(u).Reshape(new[] { 1, 1 });
            if (AnyBelow(uNorm, tiny))
            {
                continue;
            }

            u = Engine.TensorMultiplyScalar(u, Engine.TensorSum(Engine.TensorReciprocal(uNorm)));
            var v = Engine.TensorSetSlice(new Tensor<T>(new[] { n, 1 }), u, new[] { k + 1, 0 });
            var vT = Engine.TensorTranspose(v);

            // H <- P H P and Q <- Q P with P = I - 2 v v^T.
            h = Engine.TensorSubtract(h, Engine.TensorMultiplyScalar(Engine.TensorMatMul(v, Engine.TensorMatMul(vT, h)), two));
            h = Engine.TensorSubtract(h, Engine.TensorMultiplyScalar(Engine.TensorMatMul(Engine.TensorMatMul(h, v), vT), two));
            q = Engine.TensorSubtract(q, Engine.TensorMultiplyScalar(Engine.TensorMatMul(Engine.TensorMatMul(q, v), vT), two));
        }

        return (h.ToMatrix(), q.ToMatrix());
    }

    /// <summary>
    /// Rescales a reduction H = Q^T A Q by D = diag(±1) so that every subdiagonal entry
    /// H[k+1,k] is non-negative (H' = D H D, Q' = Q D).
    /// </summary>
    /// <remarks>
    /// Rotation-based (Givens) and Lanczos reductions produce non-negative subdiagonals
    /// (r = sqrt(a^2 + b^2), beta = ‖w‖); by the implicit-Q theorem an unreduced reduction with
    /// Q e1 = e1 is unique up to exactly this sign scaling. d_0 = 1 and d_(k+1) = d_k sign(H[k+1,k])
    /// (sign(0) = +1), computed with an engine cumulative product.
    /// </remarks>
    protected (Matrix<T> H, Matrix<T> Q) NormalizeSubdiagonalSigns(Matrix<T> hMatrix, Matrix<T> qMatrix)
    {
        int n = hMatrix.Rows;
        if (n < 2)
        {
            return (hMatrix, qMatrix);
        }

        var h = Tensor<T>.FromMatrix(hMatrix);
        var subdiagonal = Engine.TensorDiagonal(Engine.TensorSlice(h, new[] { 1, 0 }, new[] { n - 1, n - 1 }));
        var signs = Engine.TensorSign(subdiagonal);
        signs = Engine.TensorAdd(signs, Engine.TensorSubtract(Engine.TensorOnesLike(signs), Engine.TensorAbs(signs)));
        var d = Engine.Concat(
            new[] { Engine.TensorOnesLike(new Tensor<T>(new[] { 1 })), Engine.TensorCumProd(signs, 0) },
            0);
        var scaling = Engine.TensorDiagEmbed(d);

        var scaledH = Engine.TensorMatMul(Engine.TensorMatMul(scaling, h), scaling);
        var scaledQ = Engine.TensorMatMul(Tensor<T>.FromMatrix(qMatrix), scaling);
        return (scaledH.ToMatrix(), scaledQ.ToMatrix());
    }

    /// <summary>
    /// Solves a triangular system <c>T x = rhs</c> with one whole-matrix engine triangular solve.
    /// </summary>
    /// <param name="triangular">The square triangular coefficient matrix.</param>
    /// <param name="rhs">The right-hand side vector.</param>
    /// <param name="upper">True when <paramref name="triangular"/> is upper triangular; false for lower.</param>
    /// <param name="unitDiagonal">True to treat the diagonal as all ones without reading it.</param>
    /// <returns>The solution vector x.</returns>
    protected static Vector<T> SolveTriangularSystem(Matrix<T> triangular, Vector<T> rhs, bool upper, bool unitDiagonal = false)
    {
        var solution = Linalg.SolveTriangular(
            Tensor<T>.FromMatrix(triangular),
            Tensor<T>.FromVector(rhs, new[] { rhs.Length, 1 }),
            upper,
            unitDiagonal);
        return solution.Reshape(new[] { rhs.Length }).ToVector();
    }

    /// <summary>
    /// Solves a triangular system <c>T X = B</c> for every column of <paramref name="rhs"/> at once.
    /// </summary>
    protected static Matrix<T> SolveTriangularSystem(Matrix<T> triangular, Matrix<T> rhs, bool upper, bool unitDiagonal = false)
    {
        return Linalg.SolveTriangular(
            Tensor<T>.FromMatrix(triangular),
            Tensor<T>.FromMatrix(rhs),
            upper,
            unitDiagonal).ToMatrix();
    }

    /// <summary>
    /// True when any element of <paramref name="value"/> (typically a single-element tensor) is
    /// strictly below <paramref name="threshold"/>; evaluated with an engine comparison.
    /// </summary>
    protected bool AnyBelow(Tensor<T> value, T threshold)
    {
        return Engine.TensorCountNonzero(Engine.TensorLessThan(value, threshold)) > 0;
    }

    /// <summary>
    /// True when any element of <paramref name="value"/> (typically a single-element tensor) is
    /// strictly above <paramref name="threshold"/>; evaluated with an engine comparison.
    /// </summary>
    protected bool AnyAbove(Tensor<T> value, T threshold)
    {
        return Engine.TensorCountNonzero(Engine.TensorGreaterThan(value, threshold)) > 0;
    }

    /// <summary>
    /// Thin QR factorization with a non-negative diagonal in R, computed with one engine QR call.
    /// </summary>
    /// <param name="matrix">The m x n matrix to factor.</param>
    /// <returns>
    /// Q (m x n, orthonormal columns) and R (n x n, upper triangular, R[i,i] &gt;= 0) with A = QR.
    /// When m &lt; n the trailing n - m columns of Q and rows of R are zero.
    /// </returns>
    /// <remarks>
    /// The thin QR factorization of a full-column-rank matrix with a positive R diagonal is unique,
    /// so this matches Gram-Schmidt orthogonalization (which produces R[j,j] = ‖v_j‖ &gt; 0).
    /// The engine QR uses Householder reflections; its signs are normalized here by scaling
    /// column j of Q and row j of R by sign(R[j,j]).
    /// </remarks>
    protected (Matrix<T> Q, Matrix<T> R) ComputePositiveDiagonalQr(Matrix<T> matrix)
    {
        return ComputeSignNormalizedQr(matrix, complete: false, nonNegativeDiagonal: true);
    }

    /// <summary>
    /// QR factorization from one engine QR call with a fixed sign convention on the R diagonal.
    /// </summary>
    /// <param name="matrix">The m x n matrix to factor.</param>
    /// <param name="complete">
    /// True for the complete factorization (Q m x m, R m x n); false for the thin one (Q m x n,
    /// R n x n, zero-padded when m &lt; n).
    /// </param>
    /// <param name="nonNegativeDiagonal">
    /// True to make R[j,j] &gt;= 0 (Gram-Schmidt, Givens); false to make R[j,j] &lt;= 0 (reflecting
    /// each column with u = x + ‖x‖ e1).
    /// </param>
    /// <remarks>
    /// Column j of Q and row j of R are scaled by the same sign, so A = QR is preserved. For a
    /// full-column-rank matrix this pins down the leading min(m, n) columns of Q uniquely.
    /// </remarks>
    protected (Matrix<T> Q, Matrix<T> R) ComputeSignNormalizedQr(Matrix<T> matrix, bool complete, bool nonNegativeDiagonal)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        if (m == 0 || n == 0)
        {
            return complete
                ? (Matrix<T>.CreateIdentity(m), new Matrix<T>(m, n))
                : (new Matrix<T>(m, n), new Matrix<T>(n, n));
        }

        var (q, r) = Linalg.QR(Tensor<T>.FromMatrix(matrix), complete ? "complete" : "reduced");

        // sign'(x) = +1 for x >= 0, -1 for x < 0 (sign(0) = 0 would erase the column).
        var signs = Engine.TensorSign(Engine.TensorDiagonal(r));
        signs = Engine.TensorAdd(signs, Engine.TensorSubtract(Engine.TensorOnesLike(signs), Engine.TensorAbs(signs)));
        if (!nonNegativeDiagonal)
        {
            signs = Engine.TensorNegate(signs);
        }

        int qColumns = q.Shape[1];
        if (signs.Length < qColumns)
        {
            signs = Engine.TensorSetSlice(Engine.TensorOnesLike(new Tensor<T>(new[] { qColumns })), signs, new[] { 0 });
        }

        var signMatrix = Engine.TensorDiagEmbed(signs);
        q = Engine.TensorMatMul(q, signMatrix);
        r = Engine.TensorMatMul(signMatrix, r);

        if (complete || m >= n)
        {
            return (q.ToMatrix(), r.ToMatrix());
        }

        // Thin factorization with m < n: pad Q with zero columns and R with zero rows so the
        // shapes stay m x n and n x n.
        var paddedQ = Engine.TensorSetSlice(new Tensor<T>(new[] { m, n }), q, new[] { 0, 0 });
        var paddedR = Engine.TensorSetSlice(new Tensor<T>(new[] { n, n }), r, new[] { 0, 0 });
        return (paddedQ.ToMatrix(), paddedR.ToMatrix());
    }

    /// <summary>
    /// Checks if two values are approximately equal within a tolerance.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <param name="tolerance">The maximum acceptable difference.</param>
    /// <returns>True if the values are within tolerance; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In numerical computing, we rarely check if two floating-point
    /// numbers are exactly equal because of rounding errors. Instead, we check if they're
    /// "close enough" within some small tolerance.
    /// </para>
    /// <para>
    /// For example, if tolerance is 0.001, then 1.0001 and 1.0002 would be considered equal,
    /// but 1.0 and 1.002 would not.
    /// </para>
    /// </remarks>
    protected bool ApproximatelyEqual(T a, T b, T tolerance)
    {
        T diff = NumOps.Abs(NumOps.Subtract(a, b));
        return NumOps.LessThanOrEquals(diff, tolerance);
    }
}
