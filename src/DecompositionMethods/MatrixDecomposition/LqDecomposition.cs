namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Performs LQ decomposition on a matrix, factoring it into a lower triangular matrix L and an orthogonal matrix Q.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> LQ decomposition breaks down a matrix A into two components: A = LQ where:
/// - L is a lower triangular matrix (values only on and below the diagonal)
/// - Q is an orthogonal matrix (its columns are perpendicular to each other)
/// 
/// This decomposition is useful for solving linear systems, least squares problems,
/// and other numerical linear algebra tasks.
/// </remarks>
public class LqDecomposition<T> : IMatrixDecomposition<T>
{
    /// <summary>
    /// Provides operations for the numeric type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Gets the lower triangular matrix L from the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The L matrix contains values only on and below the diagonal.
    /// It represents one part of the factorization of the original matrix.
    /// </remarks>
    public Matrix<T> L { get; private set; }

    /// <summary>
    /// Gets the orthogonal matrix Q from the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> An orthogonal matrix has columns that are perpendicular to each other
    /// and have unit length. This means Q^T � Q = I (the identity matrix).
    /// </remarks>
    public Matrix<T> Q { get; private set; }

    /// <summary>
    /// Gets the original matrix that was decomposed.
    /// </summary>
    public Matrix<T> A { get; private set; }

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
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        A = matrix;
        (L, Q) = Decompose(matrix, algorithm);
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
    /// 2. Multiply by Q^T: x = Q^T � y
    /// 
    /// This approach is more efficient than directly inverting the matrix A.
    /// </remarks>
    public Vector<T> Solve(Vector<T> b)
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
    private (Matrix<T> L, Matrix<T> Q) Decompose(Matrix<T> matrix, LqAlgorithmType algorithm)
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
    /// A Householder reflection is a transformation that reflects a vector about a plane.
    /// This method applies a series of these reflections to create the decomposition.
    /// </remarks>
    private (Matrix<T> L, Matrix<T> Q) ComputeLqHouseholder(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        var L = new Matrix<T>(m, n);
        var Q = Matrix<T>.CreateIdentityMatrix(n);

        var a = matrix.Clone();

        for (int k = 0; k < Math.Min(m, n); k++)
        {
            var x = new Vector<T>(m - k);
            for (int i = k; i < m; i++)
            {
                x[i - k] = a[i, k];
            }

            T alpha = _numOps.Negate(_numOps.SignOrZero(x[0]));
            alpha = _numOps.Multiply(alpha, _numOps.Sqrt(x.DotProduct(x)));

            var u = new Vector<T>(x.Length);
            u[0] = _numOps.Subtract(x[0], alpha);
            for (int i = 1; i < x.Length; i++)
            {
                u[i] = x[i];
            }

            T norm_u = _numOps.Sqrt(u.DotProduct(u));
            for (int i = 0; i < u.Length; i++)
            {
                u[i] = _numOps.Divide(u[i], norm_u);
            }

            var uMatrix = new Matrix<T>(u.Length, 1);
            for (int i = 0; i < u.Length; i++)
            {
                uMatrix[i, 0] = u[i];
            }

            var uT = uMatrix.Transpose();
            var uTu = uMatrix.Multiply(uT);

            for (int i = 0; i < m - k; i++)
            {
                uTu[i, i] = _numOps.Subtract(uTu[i, i], _numOps.One);
            }

            var P = Matrix<T>.CreateIdentityMatrix(m);
            for (int i = k; i < m; i++)
            {
                for (int j = k; j < m; j++)
                {
                    P[i, j] = _numOps.Add(P[i, j], _numOps.Multiply(_numOps.FromDouble(2), uTu[i - k, j - k]));
                }
            }

            a = P.Multiply(a);
            Q = Q.Multiply(P);
        }

        L = a;

        // Ensure Q is orthogonal
        for (int i = 0; i < Q.Rows; i++)
        {
            for (int j = 0; j < Q.Columns; j++)
            {
                if (i == j)
                {
                    Q[i, j] = _numOps.One;
                }
                else
                {
                    Q[i, j] = _numOps.Negate(Q[i, j]);
                }
            }
        }

        return (L, Q);
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
    /// This method works by:
    /// 1. Taking each row of the matrix
    /// 2. Removing components that point in the same direction as previous rows
    /// 3. Normalizing the result to create orthogonal vectors
    /// 
    /// The resulting Q matrix has orthogonal columns, and L is lower triangular.
    /// </remarks>
    private (Matrix<T> L, Matrix<T> Q) ComputeLqGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        var L = new Matrix<T>(m, n);
        var Q = Matrix<T>.CreateIdentityMatrix(n);

        for (int i = 0; i < m; i++)
        {
            var v = matrix.GetRow(i);
            for (int j = 0; j < i; j++)
            {
                var q = Q.GetColumn(j);
                T proj = v.DotProduct(q);
                for (int k = 0; k < n; k++)
                {
                    v[k] = _numOps.Subtract(v[k], _numOps.Multiply(proj, q[k]));
                }
            }

            T norm = _numOps.Sqrt(v.DotProduct(v));
            for (int j = 0; j < n; j++)
            {
                Q[j, i] = _numOps.Divide(v[j], norm);
            }

            for (int j = 0; j <= i; j++)
            {
                L[i, j] = matrix.GetRow(i).DotProduct(Q.GetColumn(j));
            }
        }

        return (L, Q);
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
    /// This method:
    /// 1. Starts with a copy of the original matrix as L
    /// 2. Applies a series of rotations to transform it into lower triangular form
    /// 3. Tracks these rotations to form the Q matrix
    /// 
    /// Givens rotations are particularly useful when you need to zero out just a few elements,
    /// and they're numerically stable for many applications.
    /// </remarks>
    private (Matrix<T> L, Matrix<T> Q) ComputeLqGivens(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        var L = matrix.Clone();
        var Q = Matrix<T>.CreateIdentityMatrix(n);

        for (int i = m - 1; i >= 0; i--)
        {
            for (int j = n - 1; j > i; j--)
            {
                if (!_numOps.Equals(L[i, j], _numOps.Zero))
                {
                    T a = L[i, j - 1];
                    T b = L[i, j];
                    T r = _numOps.Sqrt(_numOps.Add(_numOps.Multiply(a, a), _numOps.Multiply(b, b)));
                    T c = _numOps.Divide(a, r);
                    T s = _numOps.Divide(b, r);

                    for (int k = 0; k < m; k++)
                    {
                        T temp = L[k, j - 1];
                        L[k, j - 1] = _numOps.Add(_numOps.Multiply(c, temp), _numOps.Multiply(s, L[k, j]));
                        L[k, j] = _numOps.Subtract(_numOps.Multiply(_numOps.Negate(s), temp), _numOps.Multiply(c, L[k, j]));
                    }

                    for (int k = 0; k < n; k++)
                    {
                        T temp = Q[j - 1, k];
                        Q[j - 1, k] = _numOps.Add(_numOps.Multiply(c, temp), _numOps.Multiply(s, Q[j, k]));
                        Q[j, k] = _numOps.Subtract(_numOps.Multiply(_numOps.Negate(s), temp), _numOps.Multiply(c, Q[j, k]));
                    }
                }
            }
        }

        return (L, Q.Transpose());
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
            T sum = _numOps.Zero;
            for (int j = 0; j < i; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(L[i, j], y[j]));
            }

            y[i] = _numOps.Divide(_numOps.Subtract(b[i], sum), L[i, i]);
        }

        return y;
    }

    /// <summary>
    /// Calculates the inverse of the original matrix using the LQ decomposition.
    /// </summary>
    /// <returns>The inverse of the original matrix A.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The inverse of a matrix A is another matrix A?� such that when multiplied 
    /// together, they give the identity matrix (A � A?� = I).
    /// 
    /// This method uses a helper function that efficiently computes the inverse using
    /// the LQ decomposition we've already calculated.
    /// </remarks>
    public Matrix<T> Invert()
    {
        return MatrixHelper<T>.InvertUsingDecomposition(this);
    }
}