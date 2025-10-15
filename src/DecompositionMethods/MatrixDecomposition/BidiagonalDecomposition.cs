namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

public class BidiagonalDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> _numOps = default!;

    /// <summary>
    /// Gets the original matrix that was decomposed.
    /// </summary>
    public Matrix<T> A { get; }

    /// <summary>
    /// Gets the left orthogonal matrix in the decomposition.
    /// In simpler terms, this matrix helps transform the original matrix's columns.
    /// </summary>
    public Matrix<T> U { get; private set; }

    /// <summary>
    /// Gets the bidiagonal matrix in the decomposition.
    /// A bidiagonal matrix is a special matrix where non-zero values appear only on the main diagonal
    /// and the diagonal immediately above it (called the superdiagonal).
    /// </summary>
    public Matrix<T> B { get; private set; }

    /// <summary>
    /// Gets the right orthogonal matrix in the decomposition.
    /// In simpler terms, this matrix helps transform the original matrix's rows.
    /// </summary>
    public Matrix<T> V { get; private set; }

    /// <summary>
    /// Creates a new bidiagonal decomposition of the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for decomposition (default is Householder).</param>
    /// <remarks>
    /// Bidiagonal decomposition breaks down a matrix A into three simpler matrices: U, B, and V,
    /// where A = U*B*V^T. This makes many matrix operations easier to perform.
    /// </remarks>
    public BidiagonalDecomposition(Matrix<T> matrix, BidiagonalAlgorithmType algorithm = BidiagonalAlgorithmType.Householder)
    {
        A = matrix;
        _numOps = MathHelper.GetNumericOperations<T>();
        U = new Matrix<T>(matrix.Rows, matrix.Rows);
        B = new Matrix<T>(matrix.Columns, matrix.Columns);
        V = new Matrix<T>(matrix.Columns, matrix.Columns);
        Decompose(algorithm);
    }

    /// <summary>
    /// Performs the bidiagonal decomposition using the specified algorithm.
    /// </summary>
    /// <param name="algorithm">The algorithm to use for decomposition.</param>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    /// <remarks>
    /// Different algorithms have different performance characteristics and numerical stability.
    /// Householder is generally the most stable and commonly used method.
    /// </remarks>
    public void Decompose(BidiagonalAlgorithmType algorithm = BidiagonalAlgorithmType.Householder)
    {
        switch (algorithm)
        {
            case BidiagonalAlgorithmType.Householder:
                DecomposeHouseholder();
                break;
            case BidiagonalAlgorithmType.Givens:
                DecomposeGivens();
                break;
            case BidiagonalAlgorithmType.Lanczos:
                DecomposeLanczos();
                break;
            default:
                throw new ArgumentException("Unsupported Bidiagonal decomposition algorithm.");
        }
    }

    /// <summary>
    /// Performs bidiagonal decomposition using Householder reflections.
    /// </summary>
    /// <remarks>
    /// Householder reflections are a way to transform vectors by reflecting them across a plane.
    /// This method is numerically stable and commonly used for matrix decompositions.
    /// </remarks>
    private void DecomposeHouseholder()
    {
        int m = A.Rows;
        int n = A.Columns;
        B = A.Clone();
        U = Matrix<T>.CreateIdentity(m);
        V = Matrix<T>.CreateIdentity(n);

        for (int k = 0; k < Math.Min(m - 1, n); k++)
        {
            // Compute Householder vector for column
            Vector<T> x = B.GetColumnSegment(k, k, m - k);
            Vector<T> v = HouseholderVector(x);

            // Apply Householder reflection to B
            Matrix<T> P = Matrix<T>.CreateIdentity(m - k).Subtract(v.OuterProduct(v).Multiply(_numOps.FromDouble(2)));
            Matrix<T> subB = B.GetSubMatrix(k, k, m - k, n - k);
            B.SetSubMatrix(k, k, P.Multiply(subB));

            // Update U
            Matrix<T> subU = U.GetSubMatrix(0, k, m, m - k);
            U.SetSubMatrix(0, k, subU.Multiply(P.Transpose()));

            if (k < n - 2)
            {
                // Compute Householder vector for row
                x = B.GetRowSegment(k, k + 1, n - k - 1);
                v = HouseholderVector(x);

                // Apply Householder reflection to B
                P = Matrix<T>.CreateIdentity(n - k - 1).Subtract(v.OuterProduct(v).Multiply(_numOps.FromDouble(2)));
                subB = B.GetSubMatrix(k, k + 1, m - k, n - k - 1);
                B.SetSubMatrix(k, k + 1, subB.Multiply(P));

                // Update V
                Matrix<T> subV = V.GetSubMatrix(k + 1, 0, n - k - 1, n);
                V.SetSubMatrix(k + 1, 0, P.Multiply(subV));
            }
        }
    }

    /// <summary>
    /// Performs bidiagonal decomposition using Givens rotations.
    /// </summary>
    /// <remarks>
    /// Givens rotations are a way to zero out specific elements in a matrix by rotating
    /// two rows or columns. This method is useful for creating bidiagonal matrices.
    /// </remarks>
    private void DecomposeGivens()
    {
        int m = A.Rows;
        int n = A.Columns;
        B = A.Clone();
        U = Matrix<T>.CreateIdentity(m);
        V = Matrix<T>.CreateIdentity(n);

        for (int k = 0; k < Math.Min(m - 1, n); k++)
        {
            for (int i = m - 1; i > k; i--)
            {
                GivensRotation(B, U, i - 1, i, k, k, true);
            }

            if (k < n - 2)
            {
                for (int j = n - 1; j > k + 1; j--)
                {
                    GivensRotation(B, V, k, k, j - 1, j, false);
                }
            }
        }
    }

    /// <summary>
    /// Performs bidiagonal decomposition using the Lanczos algorithm.
    /// </summary>
    /// <remarks>
    /// The Lanczos algorithm is an iterative method that can efficiently compute
    /// partial decompositions of large matrices. It's particularly useful for
    /// sparse matrices (matrices with mostly zero values).
    /// </remarks>
    private void DecomposeLanczos()
    {
        int m = A.Rows;
        int n = A.Columns;
        B = new Matrix<T>(m, n);
        U = new Matrix<T>(m, m);
        V = new Matrix<T>(n, n);

        Vector<T> u = new Vector<T>(m);
        Vector<T> v = new Vector<T>(n);

        // Initialize with random unit vector
        Random rand = new();
        for (int i = 0; i < n; i++)
        {
            v[i] = _numOps.FromDouble(rand.NextDouble());
        }
        v = v.Divide(v.Norm());

        for (int j = 0; j < Math.Min(m, n); j++)
        {
            u = A.Multiply(v).Subtract(B.GetColumn(j - 1).Multiply(B[j - 1, j]));
            T alpha = u.Norm();
            u = u.Divide(alpha);

            v = A.Transpose().Multiply(u).Subtract(B.GetRow(j).Multiply(alpha));
            T beta = v.Norm();
            v = v.Divide(beta);

            B[j, j] = alpha;
            if (j < n - 1) B[j, j + 1] = beta;

            U.SetColumn(j, u);
            if (j < n) V.SetColumn(j, v);
        }
    }

    /// <summary>
    /// Solves the linear system Ax = b using the bidiagonal decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    /// <exception cref="ArgumentException">Thrown when the length of vector b doesn't match the number of rows in matrix A.</exception>
    /// <remarks>
    /// This method uses the decomposition to efficiently solve the system without
    /// directly inverting the matrix, which is more numerically stable.
    /// </remarks>
    public Vector<T> Solve(Vector<T> b)
    {
        if (b.Length != A.Rows)
            throw new ArgumentException("Vector<double> b must have the same length as the number of rows in matrix A.");

        // Solve Ax = b using U*B*V^T * x = b
        Vector<T> y = U.Transpose().Multiply(b);
        Vector<T> z = SolveBidiagonal(y);
        return V.Multiply(z);
    }

    /// <summary>
    /// Computes the inverse of the original matrix A using the bidiagonal decomposition.
    /// </summary>
    /// <returns>The inverse matrix of A.</returns>
    /// <remarks>
    /// Matrix inversion is computationally expensive and can be numerically unstable.
    /// When possible, use the Solve method instead of explicitly computing the inverse.
    /// </remarks>
    public Matrix<T> Invert()
    {
        int n = A.Columns;
        Matrix<T> inverse = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            Vector<T> ei = new Vector<T>(n);
            ei[i] = _numOps.One;
            inverse.SetColumn(i, Solve(ei));
        }

        return inverse;
    }

    /// <summary>
    /// Computes a Householder reflection vector for the given input vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>A Householder vector that can be used to create a reflection matrix.</returns>
    /// <remarks>
    /// A Householder reflection is a transformation that reflects a vector across a plane.
    /// It's used to zero out specific elements in a matrix during decomposition.
    /// </remarks>
    private Vector<T> HouseholderVector(Vector<T> x)
    {
        T norm = x.Norm();
        Vector<T> v = (Vector<T>)x.Clone();
        v[0] = _numOps.Add(v[0], _numOps.Multiply(_numOps.SignOrZero(x[0]), norm));

        return v.Divide(v.Norm());
    }

        /// <summary>
    /// Applies a Givens rotation to the specified matrices.
    /// </summary>
    /// <param name="M">The matrix to which the rotation is applied.</param>
    /// <param name="Q">The orthogonal matrix that accumulates the rotations.</param>
    /// <param name="i">The first row/column index for the rotation.</param>
    /// <param name="k">The second row/column index for the rotation.</param>
    /// <param name="j">The first column/row index for the rotation.</param>
    /// <param name="l">The second column/row index for the rotation.</param>
    /// <param name="isLeft">If true, applies a left rotation (row operation); otherwise, applies a right rotation (column operation).</param>
    /// <remarks>
    /// A Givens rotation is a simple rotation in a plane spanned by two coordinate axes.
    /// It's used to selectively zero out specific elements in a matrix during decomposition.
    /// </remarks>
    private void GivensRotation(Matrix<T> M, Matrix<T> Q, int i, int k, int j, int l, bool isLeft)
    {
        T a = M[i, j];
        T b = M[k, l];
        T r = _numOps.Sqrt(_numOps.Add(_numOps.Multiply(a, a), _numOps.Multiply(b, b)));
        T c = _numOps.Divide(a, r);
        T s = _numOps.Divide(b, r);

        if (isLeft)
        {
            for (int j2 = 0; j2 < M.Columns; j2++)
            {
                T temp1 = M[i, j2];
                T temp2 = M[k, j2];
                M[i, j2] = _numOps.Add(_numOps.Multiply(c, temp1), _numOps.Multiply(s, temp2));
                M[k, j2] = _numOps.Subtract(_numOps.Multiply(_numOps.Negate(s), temp1), _numOps.Multiply(c, temp2));
            }

            for (int i2 = 0; i2 < Q.Rows; i2++)
            {
                T temp1 = Q[i2, i];
                T temp2 = Q[i2, k];
                Q[i2, i] = _numOps.Add(_numOps.Multiply(c, temp1), _numOps.Multiply(s, temp2));
                Q[i2, k] = _numOps.Subtract(_numOps.Multiply(_numOps.Negate(s), temp1), _numOps.Multiply(c, temp2));
            }
        }
        else
        {
            for (int i2 = 0; i2 < M.Rows; i2++)
            {
                T temp1 = M[i2, j];
                T temp2 = M[i2, l];
                M[i2, j] = _numOps.Add(_numOps.Multiply(c, temp1), _numOps.Multiply(s, temp2));
                M[i2, l] = _numOps.Subtract(_numOps.Multiply(_numOps.Negate(s), temp1), _numOps.Multiply(c, temp2));
            }

            for (int j2 = 0; j2 < Q.Columns; j2++)
            {
                T temp1 = Q[j, j2];
                T temp2 = Q[l, j2];
                Q[j, j2] = _numOps.Add(_numOps.Multiply(c, temp1), _numOps.Multiply(s, temp2));
                Q[l, j2] = _numOps.Subtract(_numOps.Multiply(_numOps.Negate(s), temp1), _numOps.Multiply(c, temp2));
            }
        }
    }

    /// <summary>
    /// Solves a bidiagonal system of linear equations.
    /// </summary>
    /// <param name="y">The right-hand side vector of the equation Bx = y.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// This method efficiently solves a system where the coefficient matrix is bidiagonal
    /// (has non-zero elements only on the main diagonal and the superdiagonal).
    /// It uses back-substitution, which is much faster than general matrix inversion.
    /// </remarks>
    private Vector<T> SolveBidiagonal(Vector<T> y)
    {
        int n = B.Columns;
        Vector<T> x = new(n);

        for (int i = n - 1; i >= 0; i--)
        {
            T sum = y[i];
            if (i < n - 1)
                sum = _numOps.Subtract(sum, _numOps.Multiply(B[i, i + 1], x[i + 1]));
            x[i] = _numOps.Divide(sum, B[i, i]);
        }

        return x;
    }

    /// <summary>
    /// Returns the three factor matrices of the bidiagonal decomposition.
    /// </summary>
    /// <returns>
    /// A tuple containing:
    /// - U: The left orthogonal matrix
    /// - B: The bidiagonal matrix
    /// - V: The right orthogonal matrix
    /// </returns>
    /// <remarks>
    /// The original matrix A can be reconstructed as A = U * B * V^T.
    /// This decomposition is useful for solving linear systems and computing singular values.
    /// </remarks>
    public (Matrix<T> U, Matrix<T> B, Matrix<T> V) GetFactors()
    {
        return (U, B, V);
    }
}