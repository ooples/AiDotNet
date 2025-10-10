namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Represents a tridiagonal decomposition of a matrix, which decomposes a matrix A into Q*T*Q^T,
/// where Q is orthogonal and T is tridiagonal.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A tridiagonal matrix is a special type of matrix where non-zero elements
/// can only appear on the main diagonal and the diagonals directly above and below it.
/// This decomposition transforms a complex matrix into this simpler form, making many
/// calculations much faster and more efficient.
/// </para>
/// </remarks>
public class TridiagonalDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> _numOps = default!;

    /// <summary>
    /// Gets the original matrix being decomposed.
    /// </summary>
    public Matrix<T> A { get; }
    
    /// <summary>
    /// Gets the orthogonal matrix Q in the decomposition A = Q*T*Q^T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An orthogonal matrix has the special property that when multiplied by its
    /// transpose, it gives the identity matrix (like multiplying a number by its reciprocal gives 1).
    /// This property makes orthogonal matrices very useful in many calculations.
    /// </para>
    /// </remarks>
    public Matrix<T> QMatrix { get; private set; }
    
    /// <summary>
    /// Gets the tridiagonal matrix T in the decomposition A = Q*T*Q^T.
    /// </summary>
    public Matrix<T> TMatrix { get; private set; }

    /// <summary>
    /// Initializes a new instance of the TridiagonalDecomposition class.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <param name="algorithm">The algorithm to use for decomposition (default is Householder)</param>
    public TridiagonalDecomposition(Matrix<T> matrix, TridiagonalAlgorithmType algorithm = TridiagonalAlgorithmType.Householder)
    {
        A = matrix;
        _numOps = MathHelper.GetNumericOperations<T>();
        QMatrix = new Matrix<T>(matrix.Rows, matrix.Columns);
        TMatrix = new Matrix<T>(matrix.Rows, matrix.Columns);
        Decompose(algorithm);
    }

    /// <summary>
    /// Performs the tridiagonal decomposition using the specified algorithm.
    /// </summary>
    /// <param name="algorithm">The algorithm to use for decomposition</param>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified</exception>
    public void Decompose(TridiagonalAlgorithmType algorithm = TridiagonalAlgorithmType.Householder)
    {
        switch (algorithm)
        {
            case TridiagonalAlgorithmType.Householder:
                DecomposeHouseholder();
                break;
            case TridiagonalAlgorithmType.Givens:
                DecomposeGivens();
                break;
            case TridiagonalAlgorithmType.Lanczos:
                DecomposeLanczos();
                break;
            default:
                throw new ArgumentException("Unsupported Tridiagonal decomposition algorithm.");
        }
    }

    /// <summary>
    /// Performs tridiagonal decomposition using the Householder algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Householder algorithm uses special reflections (like mirrors) to
    /// transform the matrix step by step into a tridiagonal form. It's one of the most stable
    /// and widely used methods for this type of decomposition.
    /// </para>
    /// </remarks>
    private void DecomposeHouseholder()
    {
        int n = A.Rows;
        TMatrix = A.Clone();
        QMatrix = Matrix<T>.CreateIdentity(n);

        for (int k = 0; k < n - 2; k++)
        {
            Vector<T> x = TMatrix.GetColumn(k).GetSubVector(k + 1, n - k - 1);
            T alpha = _numOps.Multiply(_numOps.SignOrZero(x[0]), x.Norm());
            Vector<T> u = x.Subtract(Vector<T>.CreateDefault(x.Length, alpha).SetValue(0, x[0]));
            u = u.Divide(u.Norm());

            Matrix<T> P = Matrix<T>.CreateIdentity(n);
            for (int i = k + 1; i < n; i++)
            {
                for (int j = k + 1; j < n; j++)
                {
                    P[i, j] = _numOps.Subtract(P[i, j], _numOps.Multiply(_numOps.FromDouble(2), _numOps.Multiply(u[i - k - 1], u[j - k - 1])));
                }
            }

            TMatrix = P.Multiply(TMatrix).Multiply(P);
            QMatrix = QMatrix.Multiply(P);
        }
    }

    /// <summary>
    /// Performs tridiagonal decomposition using the Givens algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Givens algorithm uses a series of rotations to gradually transform
    /// the matrix into tridiagonal form. Think of it like rotating parts of the matrix to
    /// move unwanted values to zero, one element at a time.
    /// </para>
    /// </remarks>
    private void DecomposeGivens()
    {
        int n = A.Rows;
        QMatrix = Matrix<T>.CreateIdentity(n);
        TMatrix = A.Clone();

        for (int i = 0; i < n - 1; i++)
        {
            for (int j = i + 2; j < n; j++)
            {
                if (!_numOps.Equals(TMatrix[j, i], _numOps.Zero))
                {
                    // Calculate Givens rotation
                    T a = TMatrix[i + 1, i];
                    T b = TMatrix[j, i];
                    T r = _numOps.Sqrt(_numOps.Add(_numOps.Multiply(a, a), _numOps.Multiply(b, b)));
                    T c = _numOps.Divide(a, r);
                    T s = _numOps.Divide(b, r);

                    // Apply Givens rotation to TMatrix
                    for (int k = i; k < n; k++)
                    {
                        T temp1 = TMatrix[i + 1, k];
                        T temp2 = TMatrix[j, k];
                        TMatrix[i + 1, k] = _numOps.Add(_numOps.Multiply(c, temp1), _numOps.Multiply(s, temp2));
                        TMatrix[j, k] = _numOps.Subtract(_numOps.Multiply(_numOps.Negate(s), temp1), _numOps.Multiply(c, temp2));
                    }

                    // Update QMatrix
                    for (int k = 0; k < n; k++)
                    {
                        T temp1 = QMatrix[k, i + 1];
                        T temp2 = QMatrix[k, j];
                        QMatrix[k, i + 1] = _numOps.Add(_numOps.Multiply(c, temp1), _numOps.Multiply(s, temp2));
                        QMatrix[k, j] = _numOps.Subtract(_numOps.Multiply(_numOps.Negate(s), temp1), _numOps.Multiply(c, temp2));
                    }
                }
            }
        }

        // Ensure TMatrix is tridiagonal (set small values to zero)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (Math.Abs(i - j) > 1)
                {
                    TMatrix[i, j] = _numOps.Zero;
                }
            }
        }
    }

    /// <summary>
    /// Performs tridiagonal decomposition using the Lanczos algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Lanczos algorithm is an iterative method that builds the tridiagonal
    /// matrix step by step. It's particularly efficient for large, sparse matrices (matrices with
    /// lots of zeros). Think of it like summarizing a large dataset by capturing only its most
    /// important patterns.
    /// </para>
    /// </remarks>
    private void DecomposeLanczos()
    {
        int n = A.Rows;
        QMatrix = new Matrix<T>(n, n);
        TMatrix = new Matrix<T>(n, n);

        Vector<T> v = new Vector<T>(n);
        v[0] = _numOps.One;
        Vector<T> w = A.Multiply(v);
        T alpha = w.DotProduct(v);
        w = w.Subtract(v.Multiply(alpha));
        T beta = w.Norm();

        QMatrix.SetColumn(0, v);
        TMatrix[0, 0] = alpha;

        for (int j = 1; j < n; j++)
        {
            if (_numOps.Equals(beta, _numOps.Zero))
            {
                break; // Early termination if beta becomes zero
            }

            v = w.Divide(beta);
            QMatrix.SetColumn(j, v);

            w = A.Multiply(v).Subtract(v.Multiply(beta));
            alpha = w.DotProduct(v);
            w = w.Subtract(v.Multiply(alpha));

            if (j < n - 1)
            {
                beta = w.Norm();
            }

            TMatrix[j, j] = alpha;
            TMatrix[j, j - 1] = beta;
            TMatrix[j - 1, j] = beta;
        }

        // Ensure QMatrix is orthogonal
        for (int i = 0; i < n; i++)
        {
            Vector<T> col = QMatrix.GetColumn(i);
            col = col.Divide(col.Norm());
            QMatrix.SetColumn(i, col);
        }
    }

    /// <summary>
    /// Solves the linear system Ax = b using the tridiagonal decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector</param>
    /// <returns>The solution vector x</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the values of x in the equation Ax = b.
    /// Think of it like solving for x in the equation 3x = 6 (where x = 2),
    /// but with matrices instead of simple numbers. Using the decomposition makes
    /// this process much more efficient than directly solving the original system.
    /// </para>
    /// </remarks>
    public Vector<T> Solve(Vector<T> b)
    {
        // Solve Tx = Q^T b
        Vector<T> y = QMatrix.Transpose().Multiply(b);
        Vector<T> x = SolveTridiagonal(y);

        return x;
    }

    /// <summary>
    /// Solves a tridiagonal linear system of equations Tx = b.
    /// </summary>
    /// <param name="b">The right-hand side vector</param>
    /// <returns>The solution vector x</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method efficiently solves equations where T is a tridiagonal matrix
    /// (a matrix with non-zero values only on the main diagonal and the diagonals directly above and below it).
    /// It uses a technique called the Thomas algorithm, which is much faster than general equation-solving
    /// methods. Think of it like taking a shortcut when solving a maze because you know a special property
    /// about the maze's structure.
    /// </para>
    /// </remarks>
    private Vector<T> SolveTridiagonal(Vector<T> b)
    {
        int n = TMatrix.Rows;
        Vector<T> x = new(n);
        Vector<T> d = new(n);
        Vector<T> temp = new(n);

        // Forward elimination
        d[0] = TMatrix[0, 0];
        x[0] = b[0];
        for (int i = 1; i < n; i++)
        {
            temp[i] = _numOps.Divide(TMatrix[i, i - 1], d[i - 1]);
            d[i] = _numOps.Subtract(TMatrix[i, i], _numOps.Multiply(temp[i], TMatrix[i - 1, i]));
            x[i] = _numOps.Subtract(b[i], _numOps.Multiply(temp[i], x[i - 1]));
        }

        // Back substitution
        x[n - 1] = _numOps.Divide(x[n - 1], d[n - 1]);
        for (int i = n - 2; i >= 0; i--)
        {
            x[i] = _numOps.Divide(_numOps.Subtract(x[i], _numOps.Multiply(TMatrix[i, i + 1], x[i + 1])), d[i]);
        }

        return x;
    }

    /// <summary>
    /// Calculates the inverse of the original matrix A using the tridiagonal decomposition.
    /// </summary>
    /// <returns>The inverse of matrix A</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The inverse of a matrix is like the reciprocal of a number. Just as 1/3 is the
    /// reciprocal of 3 (because 3 × 1/3 = 1), the inverse of a matrix A is another matrix that,
    /// when multiplied by A, gives the identity matrix (the matrix equivalent of the number 1).
    /// Finding the inverse is useful for solving multiple systems of equations with the same coefficient matrix.
    /// </para>
    /// </remarks>
    public Matrix<T> Invert()
    {
        // Invert T
        Matrix<T> invT = InvertTridiagonal();

        // Compute Q * invT * Q^T
        return QMatrix.Multiply(invT).Multiply(QMatrix.Transpose());
    }

    /// <summary>
    /// Calculates the inverse of the tridiagonal matrix T.
    /// </summary>
    /// <returns>The inverse of the tridiagonal matrix T</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the inverse of the tridiagonal matrix by solving
    /// multiple equation systems. Each column of the inverse is found by solving the system
    /// Tx = e_j, where e_j is a vector with 1 in position j and 0 elsewhere. This approach
    /// is more efficient than general matrix inversion methods because it takes advantage of
    /// the tridiagonal structure.
    /// </para>
    /// </remarks>
    private Matrix<T> InvertTridiagonal()
    {
        int n = TMatrix.Rows;
        Matrix<T> inv = new(n, n);

        for (int j = 0; j < n; j++)
        {
            Vector<T> e = Vector<T>.CreateStandardBasis(n, j);
            inv.SetColumn(j, SolveTridiagonal(e));
        }

        return inv;
    }

    /// <summary>
    /// Returns the matrices Q and T from the decomposition A = Q*T*Q^T.
    /// </summary>
    /// <returns>A tuple containing the orthogonal matrix Q and the tridiagonal matrix T</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method gives you access to the two key components of the tridiagonal
    /// decomposition: the orthogonal matrix Q and the tridiagonal matrix T. These matrices can be
    /// used for various calculations like solving equations, finding eigenvalues, or further
    /// analysis of the original matrix A.
    /// </para>
    /// </remarks>
    public (Matrix<T> QMatrix, Matrix<T> TMatrix) GetFactors()
    {
        return (QMatrix, TMatrix);
    }
}