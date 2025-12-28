namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Performs LQ decomposition on a matrix, factoring it into a lower triangular matrix L and an orthogonal matrix Q.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// LQ decomposition factors a matrix A into the product A = LQ, where L is a lower triangular matrix
/// and Q is an orthogonal matrix. This decomposition is the transpose version of QR decomposition and
/// is particularly useful when working with matrices that have more columns than rows.
/// </para>
/// <para>
/// <b>For Beginners:</b> LQ decomposition breaks down a matrix A into two components:
/// L (a lower triangular matrix with values only on and below the diagonal) and
/// Q (an orthogonal matrix with columns that are perpendicular to each other).
/// This decomposition is useful for solving linear systems, least squares problems,
/// and other numerical linear algebra tasks.
/// </para>
/// <para>
/// Real-world applications:
/// - Solving underdetermined systems of equations
/// - Least squares problems with wide matrices
/// - Numerical stability in various computations
/// </para>
/// </remarks>
public class LqDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the lower triangular matrix L from the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The L matrix contains values only on and below the diagonal.
    /// It represents one part of the factorization of the original matrix.
    /// </remarks>
    public Matrix<T> L { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the orthogonal matrix Q from the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> An orthogonal matrix has columns that are perpendicular to each other
    /// and have unit length. This means Q^T * Q = I (the identity matrix).
    /// </remarks>
    public Matrix<T> Q { get; private set; } = new Matrix<T>(0, 0);

    private readonly LqAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the LqDecomposition class and performs the decomposition.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for the decomposition (default is Householder).</param>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor takes your input matrix and immediately performs
    /// the LQ decomposition using the specified algorithm. After creating this object,
    /// you can access the L and Q matrices or use methods like Solve() to work with the decomposition.
    /// </remarks>
    public LqDecomposition(Matrix<T> matrix, LqAlgorithmType algorithm = LqAlgorithmType.Householder)
        : base(matrix)
    {
        _algorithm = algorithm;
        Decompose();
    }

    /// <summary>
    /// Performs the LQ decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (L, Q) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Solves the linear system Ax = b using the LQ decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method finds the values of x that satisfy the equation Ax = b.
    /// It uses the LQ decomposition to solve this in two steps:
    ///
    /// 1. Forward substitution: Solve Ly = b for y
    /// 2. Multiply by Q^T: x = Q^T * y
    ///
    /// This approach is more efficient than directly inverting the matrix A.
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        var y = ForwardSubstitution(L, b);
        return Q.Transpose().Multiply(y);
    }

    /// <summary>
    /// Selects and applies the appropriate decomposition algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for decomposition.</param>
    /// <returns>A tuple containing the L and Q matrices.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    private (Matrix<T> L, Matrix<T> Q) ComputeDecomposition(Matrix<T> matrix, LqAlgorithmType algorithm)
    {
        return algorithm switch
        {
            LqAlgorithmType.Householder => ComputeLqHouseholder(matrix),
            LqAlgorithmType.GramSchmidt => ComputeLqGramSchmidt(matrix),
            LqAlgorithmType.Givens => ComputeLqGivens(matrix),
            _ => throw new ArgumentException("Unsupported LQ decomposition algorithm."),
        };
    }

    /// <summary>
    /// Computes the LQ decomposition using the Householder reflection method.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the L and Q matrices.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The Householder method uses reflections to transform the matrix
    /// into lower triangular form. It's numerically stable and efficient.
    ///
    /// LQ decomposition of A is computed via QR decomposition of A^T:
    /// If A^T = Q_qr * R, then A = R^T * Q_qr^T = L * Q
    /// where L = R^T (lower triangular) and Q = Q_qr^T (orthogonal).
    /// </remarks>
    private (Matrix<T> L, Matrix<T> Q) ComputeLqHouseholder(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        // LQ of A is computed via QR of A^T: A^T = Q_qr * R, so A = R^T * Q_qr^T
        var At = matrix.Transpose();

        // Perform QR decomposition on A^T using Householder reflections
        var (Q_qr, R) = ComputeQrHouseholder(At);

        // L = R^T (transpose of upper triangular R gives lower triangular L)
        // We need L to be m x min(m,n) for proper dimensions
        var L = new Matrix<T>(m, n);
        int minDim = Math.Min(m, n);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                // R is min(n,m) x m (from QR of n x m matrix A^T)
                // R^T is m x min(n,m)
                if (j < R.Rows && i < R.Columns)
                {
                    L[i, j] = R[j, i];
                }
            }
        }

        // Q = Q_qr^T (transpose of Q from QR gives our Q)
        // Q_qr is n x n, so Q is n x n
        var Q = Q_qr.Transpose();

        return (L, Q);
    }

    /// <summary>
    /// Computes QR decomposition using Householder reflections (internal helper for LQ).
    /// </summary>
    /// <param name="matrix">The matrix to decompose (m x n).</param>
    /// <returns>A tuple containing Q (m x m orthogonal) and R (m x n upper triangular).</returns>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrHouseholder(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        var R = matrix.Clone();
        var Q = Matrix<T>.CreateIdentityMatrix(m);

        int minDim = Math.Min(m, n);
        for (int k = 0; k < minDim; k++)
        {
            // Extract column k from row k onwards
            var x = new Vector<T>(m - k);
            for (int i = k; i < m; i++)
            {
                x[i - k] = R[i, k];
            }

            // Compute the norm of x
            T normX = NumOps.Sqrt(x.DotProduct(x));
            if (NumOps.Equals(normX, NumOps.Zero))
                continue;

            // Compute alpha = -sign(x[0]) * ||x||
            T alpha = NumOps.Negate(NumOps.SignOrZero(x[0]));
            alpha = NumOps.Multiply(alpha, normX);

            // Compute u = x - alpha * e1
            var u = new Vector<T>(x.Length);
            u[0] = NumOps.Subtract(x[0], alpha);
            for (int i = 1; i < x.Length; i++)
            {
                u[i] = x[i];
            }

            // Normalize u
            T normU = NumOps.Sqrt(u.DotProduct(u));
            if (NumOps.Equals(normU, NumOps.Zero))
                continue;

            for (int i = 0; i < u.Length; i++)
            {
                u[i] = NumOps.Divide(u[i], normU);
            }

            // Apply Householder transformation to R: R = (I - 2*u*u^T) * R
            // Only affect rows k to m-1
            for (int j = k; j < n; j++)
            {
                // Compute u^T * R[k:m, j]
                T dot = NumOps.Zero;
                for (int i = 0; i < u.Length; i++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(u[i], R[k + i, j]));
                }

                // R[k:m, j] -= 2 * dot * u
                T twoTimeDot = NumOps.Multiply(NumOps.FromDouble(2.0), dot);
                for (int i = 0; i < u.Length; i++)
                {
                    R[k + i, j] = NumOps.Subtract(R[k + i, j], NumOps.Multiply(twoTimeDot, u[i]));
                }
            }

            // Apply Householder transformation to Q: Q = Q * (I - 2*u*u^T)
            // This is equivalent to: Q[:, k:m] = Q[:, k:m] - 2 * Q[:, k:m] * u * u^T
            for (int i = 0; i < m; i++)
            {
                // Compute Q[i, k:m] * u
                T dot = NumOps.Zero;
                for (int j = 0; j < u.Length; j++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(Q[i, k + j], u[j]));
                }

                // Q[i, k:m] -= 2 * dot * u^T
                T twoTimeDot = NumOps.Multiply(NumOps.FromDouble(2.0), dot);
                for (int j = 0; j < u.Length; j++)
                {
                    Q[i, k + j] = NumOps.Subtract(Q[i, k + j], NumOps.Multiply(twoTimeDot, u[j]));
                }
            }
        }

        return (Q, R);
    }

    /// <summary>
    /// Computes the LQ decomposition using the Gram-Schmidt orthogonalization process.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the L and Q matrices.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The Gram-Schmidt process transforms a set of vectors into a set of
    /// orthogonal vectors (vectors that are perpendicular to each other).
    ///
    /// LQ decomposition of A is computed via QR decomposition of A^T using Gram-Schmidt:
    /// If A^T = Q_qr * R, then A = R^T * Q_qr^T = L * Q
    /// where L = R^T (lower triangular) and Q = Q_qr^T (orthogonal).
    /// </remarks>
    private (Matrix<T> L, Matrix<T> Q) ComputeLqGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        // LQ of A is computed via QR of A^T
        var At = matrix.Transpose();

        // Perform QR decomposition on A^T using Gram-Schmidt
        var (Q_qr, R) = ComputeQrGramSchmidt(At);

        // L = R^T (transpose of upper triangular R gives lower triangular L)
        var L = new Matrix<T>(m, n);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (j < R.Rows && i < R.Columns)
                {
                    L[i, j] = R[j, i];
                }
            }
        }

        // Q = Q_qr^T
        var Q = Q_qr.Transpose();

        return (L, Q);
    }

    /// <summary>
    /// Computes QR decomposition using Gram-Schmidt (internal helper for LQ).
    /// </summary>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        var Q = new Matrix<T>(m, m);
        var R = new Matrix<T>(m, n);

        int minDim = Math.Min(m, n);

        for (int j = 0; j < minDim; j++)
        {
            // Get column j
            var v = new Vector<T>(m);
            for (int i = 0; i < m; i++)
            {
                v[i] = matrix[i, j];
            }

            // Orthogonalize against previous columns
            for (int k = 0; k < j; k++)
            {
                // Get Q column k
                var qk = new Vector<T>(m);
                for (int i = 0; i < m; i++)
                {
                    qk[i] = Q[i, k];
                }

                // R[k,j] = q_k^T * v
                T rVal = v.DotProduct(qk);
                R[k, j] = rVal;

                // v = v - R[k,j] * q_k
                for (int i = 0; i < m; i++)
                {
                    v[i] = NumOps.Subtract(v[i], NumOps.Multiply(rVal, qk[i]));
                }
            }

            // Normalize
            T norm = NumOps.Sqrt(v.DotProduct(v));
            R[j, j] = norm;

            if (!NumOps.Equals(norm, NumOps.Zero))
            {
                for (int i = 0; i < m; i++)
                {
                    Q[i, j] = NumOps.Divide(v[i], norm);
                }
            }
        }

        // Fill remaining Q columns for square matrix (Gram-Schmidt of remaining orthogonal vectors)
        for (int j = minDim; j < m; j++)
        {
            Q[j, j] = NumOps.One;
        }

        return (Q, R);
    }

    /// <summary>
    /// Computes the LQ decomposition using Givens rotations.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the L and Q matrices.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Givens rotations are a way to zero out specific elements in a matrix
    /// by rotating two rows or columns at a time.
    ///
    /// LQ decomposition of A is computed via QR decomposition of A^T using Givens:
    /// If A^T = Q_qr * R, then A = R^T * Q_qr^T = L * Q
    /// where L = R^T (lower triangular) and Q = Q_qr^T (orthogonal).
    /// </remarks>
    private (Matrix<T> L, Matrix<T> Q) ComputeLqGivens(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        // LQ of A is computed via QR of A^T
        var At = matrix.Transpose();

        // Perform QR decomposition on A^T using Givens rotations
        var (Q_qr, R) = ComputeQrGivens(At);

        // L = R^T (transpose of upper triangular R gives lower triangular L)
        var L = new Matrix<T>(m, n);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (j < R.Rows && i < R.Columns)
                {
                    L[i, j] = R[j, i];
                }
            }
        }

        // Q = Q_qr^T
        var Q = Q_qr.Transpose();

        return (L, Q);
    }

    /// <summary>
    /// Computes QR decomposition using Givens rotations (internal helper for LQ).
    /// </summary>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrGivens(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        var R = matrix.Clone();
        var Q = Matrix<T>.CreateIdentityMatrix(m);

        for (int j = 0; j < n; j++)
        {
            for (int i = m - 1; i > j; i--)
            {
                if (!NumOps.Equals(R[i, j], NumOps.Zero))
                {
                    T a = R[i - 1, j];
                    T b = R[i, j];
                    T r = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(a, a), NumOps.Multiply(b, b)));

                    if (NumOps.Equals(r, NumOps.Zero))
                        continue;

                    T c = NumOps.Divide(a, r);
                    T s = NumOps.Divide(b, r);

                    // Apply Givens rotation to R (rows i-1 and i)
                    for (int k = 0; k < n; k++)
                    {
                        T temp = R[i - 1, k];
                        R[i - 1, k] = NumOps.Add(NumOps.Multiply(c, temp), NumOps.Multiply(s, R[i, k]));
                        R[i, k] = NumOps.Add(NumOps.Multiply(NumOps.Negate(s), temp), NumOps.Multiply(c, R[i, k]));
                    }

                    // Apply Givens rotation to Q (columns i-1 and i)
                    for (int k = 0; k < m; k++)
                    {
                        T temp = Q[k, i - 1];
                        Q[k, i - 1] = NumOps.Add(NumOps.Multiply(c, temp), NumOps.Multiply(s, Q[k, i]));
                        Q[k, i] = NumOps.Add(NumOps.Multiply(NumOps.Negate(s), temp), NumOps.Multiply(c, Q[k, i]));
                    }
                }
            }
        }

        return (Q, R);
    }

    /// <summary>
    /// Solves a linear system Lx = b where L is a lower triangular matrix.
    /// </summary>
    /// <param name="L">The lower triangular matrix.</param>
    /// <param name="b">The right-hand side vector.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Forward substitution is a method to solve equations when the matrix
    /// is lower triangular (has non-zero values only on and below the diagonal).
    ///
    /// The process works by:
    /// 1. Solving for the first variable directly (since there are no other variables in the first equation)
    /// 2. Using that value to solve for the second variable
    /// 3. Continuing this pattern for all variables
    ///
    /// This is much faster than general matrix inversion because we can solve for each
    /// variable one at a time, working from top to bottom.
    /// </remarks>
    private Vector<T> ForwardSubstitution(Matrix<T> L, Vector<T> b)
    {
        int n = L.Rows;
        var y = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            // VECTORIZED: Use dot product for sum computation
            T sum = NumOps.Zero;
            if (i > 0)
            {
                var rowSlice = new T[i];
                var ySlice = new T[i];
                for (int k = 0; k < i; k++)
                {
                    rowSlice[k] = L[i, k];
                    ySlice[k] = y[k];
                }
                var rowVec = new Vector<T>(rowSlice);
                var yVec = new Vector<T>(ySlice);
                sum = rowVec.DotProduct(yVec);
            }

            y[i] = NumOps.Divide(NumOps.Subtract(b[i], sum), L[i, i]);
        }

        return y;
    }
}
