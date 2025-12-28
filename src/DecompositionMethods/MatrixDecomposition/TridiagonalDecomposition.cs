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
public class TridiagonalDecomposition<T> : MatrixDecompositionBase<T>
{
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
    public Matrix<T> QMatrix { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the tridiagonal matrix T in the decomposition A = Q*T*Q^T.
    /// </summary>
    public Matrix<T> TMatrix { get; private set; } = new Matrix<T>(0, 0);

    private readonly TridiagonalAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the TridiagonalDecomposition class.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <param name="algorithm">The algorithm to use for decomposition (default is Householder)</param>
    public TridiagonalDecomposition(Matrix<T> matrix, TridiagonalAlgorithmType algorithm = TridiagonalAlgorithmType.Householder)
        : base(matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Tridiagonal decomposition requires a square matrix.", nameof(matrix));
        }

        _algorithm = algorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the tridiagonal decomposition.
    /// </summary>
    protected override void Decompose()
    {
        ComputeDecomposition(_algorithm);
    }

    /// <summary>
    /// Computes the tridiagonal decomposition using the specified algorithm.
    /// </summary>
    /// <param name="algorithm">The algorithm to use for decomposition</param>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified</exception>
    private void ComputeDecomposition(TridiagonalAlgorithmType algorithm)
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
            // Get the subcolumn below the diagonal
            Vector<T> x = TMatrix.GetColumn(k).GetSubVector(k + 1, n - k - 1);
            T xNorm = x.Norm();

            // Skip if the column is already zero
            if (NumOps.LessThan(xNorm, NumOps.FromDouble(1e-14)))
            {
                continue;
            }

            // Compute alpha = -sign(x[0]) * ||x||
            // Use OPPOSITE sign to avoid cancellation: u[0] = x[0] - alpha becomes large, not small
            T alpha = NumOps.LessThan(x[0], NumOps.Zero)
                ? xNorm
                : NumOps.Negate(xNorm);

            // Householder vector: u = x - alpha * e_1
            Vector<T> u = x.Clone();
            u[0] = NumOps.Subtract(x[0], alpha);

            T uNorm = u.Norm();
            if (NumOps.LessThan(uNorm, NumOps.FromDouble(1e-14)))
            {
                continue; // Skip if u is essentially zero
            }
            u = u.Divide(uNorm);

            // VECTORIZED: Construct Householder reflection matrix using outer product
            // P = I - 2*u*u^T (applied to the submatrix)
            Matrix<T> P = Matrix<T>.CreateIdentity(n);
            Matrix<T> uOuter = u.OuterProduct(u).Multiply(NumOps.FromDouble(2));
            for (int i = k + 1; i < n; i++)
            {
                for (int j = k + 1; j < n; j++)
                {
                    P[i, j] = NumOps.Subtract(P[i, j], uOuter[i - k - 1, j - k - 1]);
                }
            }

            // Apply similarity transformation: T = P * T * P
            TMatrix = P.Multiply(TMatrix).Multiply(P);
            QMatrix = QMatrix.Multiply(P);
        }

        // Clean up elements outside the tridiagonal structure
        // These values should be mathematically zero by construction
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (Math.Abs(i - j) > 1)
                {
                    TMatrix[i, j] = NumOps.Zero;
                }
            }
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

        // For each column, zero out elements below the subdiagonal
        for (int i = 0; i < n - 2; i++)
        {
            for (int j = i + 2; j < n; j++)
            {
                T absVal = NumOps.Abs(TMatrix[j, i]);
                if (NumOps.GreaterThan(absVal, NumOps.FromDouble(1e-14)))
                {
                    // Calculate Givens rotation to zero out TMatrix[j, i]
                    T a = TMatrix[i + 1, i];
                    T b = TMatrix[j, i];
                    T r = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(a, a), NumOps.Multiply(b, b)));

                    if (NumOps.LessThan(r, NumOps.FromDouble(1e-14)))
                    {
                        continue;
                    }

                    T c = NumOps.Divide(a, r);
                    T s = NumOps.Divide(b, r);

                    // Apply Givens rotation from the LEFT: T = G^T * T
                    // This transforms rows i+1 and j
                    Vector<T> rowI1 = TMatrix.GetRow(i + 1);
                    Vector<T> rowJ = TMatrix.GetRow(j);
                    Vector<T> newRowI1 = rowI1.Multiply(c).Add(rowJ.Multiply(s));
                    Vector<T> newRowJ = rowI1.Multiply(NumOps.Negate(s)).Add(rowJ.Multiply(c));
                    TMatrix.SetRow(i + 1, newRowI1);
                    TMatrix.SetRow(j, newRowJ);

                    // Apply Givens rotation from the RIGHT: T = T * G
                    // This transforms columns i+1 and j (for similarity transformation)
                    Vector<T> colI1 = TMatrix.GetColumn(i + 1);
                    Vector<T> colJ = TMatrix.GetColumn(j);
                    Vector<T> newColI1 = colI1.Multiply(c).Add(colJ.Multiply(s));
                    Vector<T> newColJ = colI1.Multiply(NumOps.Negate(s)).Add(colJ.Multiply(c));
                    TMatrix.SetColumn(i + 1, newColI1);
                    TMatrix.SetColumn(j, newColJ);

                    // Accumulate Q: Q = Q * G
                    Vector<T> qColI1 = QMatrix.GetColumn(i + 1);
                    Vector<T> qColJ = QMatrix.GetColumn(j);
                    Vector<T> newQColI1 = qColI1.Multiply(c).Add(qColJ.Multiply(s));
                    Vector<T> newQColJ = qColI1.Multiply(NumOps.Negate(s)).Add(qColJ.Multiply(c));
                    QMatrix.SetColumn(i + 1, newQColI1);
                    QMatrix.SetColumn(j, newQColJ);
                }
            }
        }

        // Clean up elements outside the tridiagonal structure
        // These values should be mathematically zero by construction
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (Math.Abs(i - j) > 1)
                {
                    TMatrix[i, j] = NumOps.Zero;
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

        // Initialize with a unit vector
        Vector<T> v = new Vector<T>(n);
        v[0] = NumOps.One;
        Vector<T> vPrev = new Vector<T>(n); // Previous v vector (initially zero)

        // First iteration
        Vector<T> w = A.Multiply(v);
        T alpha = w.DotProduct(v);
        w = w.Subtract(v.Multiply(alpha));
        T beta = w.Norm();

        QMatrix.SetColumn(0, v);
        TMatrix[0, 0] = alpha;

        for (int j = 1; j < n; j++)
        {
            // Check for breakdown
            if (NumOps.LessThan(beta, NumOps.FromDouble(1e-14)))
            {
                // Generate a new orthogonal vector using Gram-Schmidt
                Vector<T> newV = new Vector<T>(n);
                bool found = false;
                for (int k = 0; k < n && !found; k++)
                {
                    newV = new Vector<T>(n);
                    newV[k] = NumOps.One;

                    // Orthogonalize against all previous vectors
                    for (int i = 0; i < j; i++)
                    {
                        Vector<T> qi = QMatrix.GetColumn(i);
                        T dot = newV.DotProduct(qi);
                        newV = newV.Subtract(qi.Multiply(dot));
                    }

                    T newVNorm = newV.Norm();
                    if (NumOps.GreaterThan(newVNorm, NumOps.FromDouble(1e-10)))
                    {
                        v = newV.Divide(newVNorm);
                        found = true;
                    }
                }

                if (!found)
                {
                    break; // Cannot find orthogonal vector, stop
                }
            }
            else
            {
                // Standard Lanczos three-term recurrence
                vPrev = QMatrix.GetColumn(j - 1);
                v = w.Divide(beta);
            }

            QMatrix.SetColumn(j, v);

            // w = A*v - beta*v_{j-1}
            w = A.Multiply(v).Subtract(vPrev.Multiply(beta));

            // alpha = v^T * w
            alpha = w.DotProduct(v);

            // w = w - alpha*v
            w = w.Subtract(v.Multiply(alpha));

            // Re-orthogonalization (important for numerical stability)
            for (int i = 0; i < j; i++)
            {
                Vector<T> qi = QMatrix.GetColumn(i);
                T dot = w.DotProduct(qi);
                w = w.Subtract(qi.Multiply(dot));
            }

            // Compute new beta
            beta = w.Norm();

            // Set tridiagonal elements
            TMatrix[j, j] = alpha;
            if (j > 0)
            {
                T prevBeta = NumOps.GreaterThan(NumOps.Abs(TMatrix[j - 1, j]), NumOps.FromDouble(1e-14))
                    ? TMatrix[j - 1, j]
                    : beta;
                TMatrix[j, j - 1] = prevBeta;
                TMatrix[j - 1, j] = prevBeta;
            }
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
    public override Vector<T> Solve(Vector<T> b)
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
            temp[i] = NumOps.Divide(TMatrix[i, i - 1], d[i - 1]);
            d[i] = NumOps.Subtract(TMatrix[i, i], NumOps.Multiply(temp[i], TMatrix[i - 1, i]));
            x[i] = NumOps.Subtract(b[i], NumOps.Multiply(temp[i], x[i - 1]));
        }

        // Back substitution
        x[n - 1] = NumOps.Divide(x[n - 1], d[n - 1]);
        for (int i = n - 2; i >= 0; i--)
        {
            x[i] = NumOps.Divide(NumOps.Subtract(x[i], NumOps.Multiply(TMatrix[i, i + 1], x[i + 1])), d[i]);
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
    /// reciprocal of 3 (because 3 * 1/3 = 1), the inverse of a matrix A is another matrix that,
    /// when multiplied by A, gives the identity matrix (the matrix equivalent of the number 1).
    /// Finding the inverse is useful for solving multiple systems of equations with the same coefficient matrix.
    /// </para>
    /// </remarks>
    public override Matrix<T> Invert()
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
