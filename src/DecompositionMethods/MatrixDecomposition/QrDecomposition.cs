namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Performs QR decomposition on a matrix, factoring it into an orthogonal matrix Q and an upper triangular matrix R.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> QR decomposition breaks down a matrix into two parts - Q (which has perpendicular columns with length 1) 
/// and R (which is triangular with zeros below the diagonal). This is useful for solving equations and other matrix operations.
/// Think of it like factoring a number into its prime components, but for matrices.
/// </para>
/// </remarks>
public class QrDecomposition<T> : IMatrixDecomposition<T>
{
    /// <summary>
    /// Gets the orthogonal matrix Q from the decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Q matrix has special properties - its columns are perpendicular to each other 
    /// (orthogonal) and each column has a length of 1. This makes it useful for many calculations.
    /// </para>
    /// </remarks>
    public Matrix<T> Q { get; private set; }

    /// <summary>
    /// Gets the upper triangular matrix R from the decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The R matrix is upper triangular, which means it has numbers on and above the diagonal,
    /// and zeros below the diagonal. This structure makes solving equations much easier.
    /// </para>
    /// </remarks>
    public Matrix<T> R { get; private set; }

    /// <summary>
    /// Gets the original matrix that was decomposed.
    /// </summary>
    public Matrix<T> A { get; private set; }

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps = default!;

    /// <summary>
    /// Creates a new QR decomposition of the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="qrAlgorithm">The algorithm to use for QR decomposition (default is Householder).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different algorithms can be used to perform QR decomposition. Each has advantages in terms of 
    /// speed, accuracy, or memory usage. Householder is generally a good default choice for most applications.
    /// </para>
    /// </remarks>
    public QrDecomposition(Matrix<T> matrix, QrAlgorithmType qrAlgorithm = QrAlgorithmType.Householder)
    {
        A = matrix;
        _numOps = MathHelper.GetNumericOperations<T>();
        (Q, R) = Decompose(matrix, qrAlgorithm);
    }

    /// <summary>
    /// Solves the linear system Ax = b using the QR decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector.</param>
    /// <returns>The solution vector x such that Ax ≈ b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method solves equations of the form Ax = b, where A is a matrix, and x and b are vectors.
    /// It finds the values of x that make the equation true. Using QR decomposition often gives more accurate results
    /// than other methods, especially for complex matrices.
    /// </para>
    /// </remarks>
    public Vector<T> Solve(Vector<T> b)
    {
        var y = Q.Transpose().Multiply(b);
        return BackSubstitution(R, y);
    }

    /// <summary>
    /// Selects and applies the appropriate QR decomposition algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    private (Matrix<T> Q, Matrix<T> R) Decompose(Matrix<T> matrix, QrAlgorithmType algorithm)
    {
        return algorithm switch
        {
            QrAlgorithmType.GramSchmidt => ComputeQrGramSchmidt(matrix),
            QrAlgorithmType.Householder => ComputeQrHouseholder(matrix),
            QrAlgorithmType.Givens => ComputeQrGivens(matrix),
            QrAlgorithmType.ModifiedGramSchmidt => ComputeQrModifiedGramSchmidt(matrix),
            QrAlgorithmType.IterativeGramSchmidt => ComputeQrIterativeGramSchmidt(matrix),
            _ => throw new ArgumentException("Unsupported QR decomposition algorithm.")
        };
    }

    /// <summary>
    /// Computes QR decomposition using the classical Gram-Schmidt process.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Gram-Schmidt process transforms a set of vectors into a set of perpendicular vectors.
    /// It works by taking each column of the matrix and removing any components that point in the same direction
    /// as previous columns. This is like ensuring that each new direction is completely different from all previous ones.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = new(m, n);
        Matrix<T> R = new(n, n);

        for (int j = 0; j < n; j++)
        {
            Vector<T> v = matrix.GetColumn(j);
            for (int i = 0; i < j; i++)
            {
                R[i, j] = Q.GetColumn(i).DotProduct(v);
                v = v.Subtract(Q.GetColumn(i).Multiply(R[i, j]));
            }
            R[j, j] = v.Norm();
            if (!_numOps.Equals(R[j, j], _numOps.Zero))
            {
                for (int i = 0; i < m; i++)
                {
                    Q[i, j] = _numOps.Divide(v[i], R[j, j]);
                }
            }
        }

        return (Q, R);
    }

    /// <summary>
    /// Computes QR decomposition using Householder reflections.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Householder reflections use special matrices that act like mirrors, reflecting vectors
    /// across a plane. This method is more numerically stable than Gram-Schmidt, meaning it gives more accurate
    /// results, especially for large or complex matrices. It works by systematically creating zeros below the diagonal
    /// of the matrix.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrHouseholder(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = Matrix<T>.CreateIdentityMatrix<T>(m);
        Matrix<T> R = matrix.Clone();

        for (int k = 0; k < n; k++)
        {
            Vector<T> x = R.GetSubColumn(k, k, m - k);
            T normX = x.Norm();
            Vector<T> e = new(m - k)
            {
                [0] = _numOps.One
            };

            Vector<T> u = x.Add(e.Multiply(normX));
            T normU = u.Norm();

            if (!_numOps.Equals(normU, _numOps.Zero))
            {
                u = u.Divide(normU);
                Matrix<T> H = Matrix<T>.CreateIdentityMatrix<T>(m - k)
                    .Subtract(u.OuterProduct(u).Multiply(_numOps.FromDouble(2)));

                Matrix<T> QkTranspose = Matrix<T>.CreateIdentityMatrix<T>(m);
                for (int i = k; i < m; i++)
                {
                    for (int j = k; j < m; j++)
                    {
                        QkTranspose[i, j] = H[i - k, j - k];
                    }
                }

                Q = Q.Multiply(QkTranspose);
                R = QkTranspose.Multiply(R);
            }
        }

        return (Q, R);
    }

    /// <summary>
    /// Computes QR decomposition using Givens rotations.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Givens rotations work by rotating the matrix in a plane to create zeros in specific positions.
    /// Think of it like carefully turning a dial to make certain values become zero. This method is particularly useful
    /// for sparse matrices (matrices with many zeros) because it can target specific elements without changing others.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrGivens(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = Matrix<T>.CreateIdentityMatrix<T>(m);
        Matrix<T> R = matrix.Clone();

        for (int j = 0; j < n; j++)
        {
            for (int i = m - 1; i > j; i--)
            {
                if (!_numOps.Equals(R[i, j], _numOps.Zero))
                {
                    T a = R[i - 1, j];
                    T b = R[i, j];
                    T r = _numOps.Sqrt(_numOps.Add(_numOps.Multiply(a, a), _numOps.Multiply(b, b)));
                    T c = _numOps.Divide(a, r);
                    T s = _numOps.Divide(b, r);

                    Matrix<T> G = Matrix<T>.CreateIdentityMatrix<T>(m);
                    G[i - 1, i - 1] = c;
                    G[i, i] = c;
                    G[i - 1, i] = s;
                    G[i, i - 1] = _numOps.Negate(s);

                    R = G.Multiply(R);
                    Q = Q.Multiply(G.Transpose());
                }
            }
        }

        return (Q, R);
    }

    /// <summary>
    /// Computes QR decomposition using the Modified Gram-Schmidt process, which is more numerically stable.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Modified Gram-Schmidt process is an improved version of the classical Gram-Schmidt method.
    /// It performs the same task (creating perpendicular vectors) but does it in a way that reduces rounding errors
    /// in calculations. This makes it more reliable for complex problems or when high precision is needed.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrModifiedGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = new(m, n);
        Matrix<T> R = new(n, n);

        for (int k = 0; k < n; k++)
        {
            Vector<T> v = matrix.GetColumn(k);
            R[k, k] = v.Norm();
            Q.SetColumn(k, v.Divide(R[k, k]));

            for (int j = k + 1; j < n; j++)
            {
                R[k, j] = Q.GetColumn(k).DotProduct(matrix.GetColumn(j));
                matrix.SetColumn(j, matrix.GetColumn(j).Subtract(Q.GetColumn(k).Multiply(R[k, j])));
            }
        }

        return (Q, R);
    }

    /// <summary>
    /// Computes QR decomposition using the Iterative Gram-Schmidt process for enhanced numerical stability.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Iterative Gram-Schmidt method repeats the orthogonalization process multiple times
    /// to achieve better accuracy. Think of it like double-checking your work to make sure the vectors are truly
    /// perpendicular to each other. This method is useful when working with matrices that might cause precision problems.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrIterativeGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = new(m, n);
        Matrix<T> R = new(n, n);

        for (int k = 0; k < n; k++)
        {
            Vector<T> v = matrix.GetColumn(k);
            for (int i = 0; i < 2; i++) // Perform two iterations
            {
                for (int j = 0; j < k; j++)
                {
                    T r = Q.GetColumn(j).DotProduct(v);
                    R[j, k] = _numOps.Add(R[j, k], r);
                    v = v.Subtract(Q.GetColumn(j).Multiply(r));
                }
            }
            R[k, k] = v.Norm();
            Q.SetColumn(k, v.Divide(R[k, k]));
        }

        return (Q, R);
    }

    /// <summary>
    /// Solves a system of linear equations using back substitution on an upper triangular matrix.
    /// </summary>
    /// <param name="R">The upper triangular matrix.</param>
    /// <param name="y">The right-hand side vector.</param>
    /// <returns>The solution vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Back substitution is a simple way to solve equations when your matrix is triangular.
    /// It works by starting at the bottom row (which has only one unknown) and solving for that value.
    /// Then it moves up row by row, using the values already found to solve for each new unknown.
    /// This is like solving a puzzle where each piece you place helps you figure out where the next one goes.
    /// </para>
    /// </remarks>
    private Vector<T> BackSubstitution(Matrix<T> R, Vector<T> y)
    {
        var x = new Vector<T>(R.Columns);
        for (int i = R.Columns - 1; i >= 0; i--)
        {
            T sum = _numOps.Zero;
            for (int j = i + 1; j < R.Columns; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(R[i, j], x[j]));
            }
            x[i] = _numOps.Divide(_numOps.Subtract(y[i], sum), R[i, i]);
        }

        return x;
    }

    /// <summary>
    /// Calculates the inverse of the original matrix using QR decomposition.
    /// </summary>
    /// <returns>The inverse of the original matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The inverse of a matrix is like the reciprocal of a number. When you multiply a matrix by its inverse,
    /// you get the identity matrix (similar to how multiplying a number by its reciprocal gives 1).
    /// Finding the inverse is useful for solving systems of equations and many other matrix operations.
    /// QR decomposition provides a stable way to calculate this inverse.
    /// </para>
    /// </remarks>
    public Matrix<T> Invert()
    {
        return MatrixHelper<T>.InvertUsingDecomposition(this);
    }
}