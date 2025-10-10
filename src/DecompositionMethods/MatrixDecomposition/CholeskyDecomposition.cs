namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements the Cholesky decomposition for symmetric positive definite matrices.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
/// <remarks>
/// The Cholesky decomposition breaks down a symmetric positive definite matrix into 
/// the product of a lower triangular matrix and its transpose (A = L * L^T).
/// This is useful for solving linear systems and matrix inversion more efficiently.
/// </remarks>
public class CholeskyDecomposition<T> : IMatrixDecomposition<T>
{
    /// <summary>
    /// Provides numeric operations for the specified type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps = default!;

    /// <summary>
    /// Gets the lower triangular matrix L from the decomposition A = L * L^T.
    /// </summary>
    public Matrix<T> L { get; private set; }

    /// <summary>
    /// Gets the original matrix that was decomposed.
    /// </summary>
    public Matrix<T> A { get; private set; }

    /// <summary>
    /// Initializes a new instance of the Cholesky decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The symmetric positive definite matrix to decompose.</param>
    /// <param name="algorithm">The specific Cholesky algorithm to use (default is Crout).</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the matrix is not square, not symmetric, or not positive definite.
    /// </exception>
    /// <remarks>
    /// A positive definite matrix has all positive eigenvalues, which means it has a unique
    /// Cholesky decomposition. In practical terms, for most matrices you encounter in data
    /// science, this means the matrix must be symmetric with positive values on the diagonal.
    /// </remarks>
    public CholeskyDecomposition(Matrix<T> matrix, CholeskyAlgorithmType algorithm = CholeskyAlgorithmType.Crout)
    {
        A = matrix;
        _numOps = MathHelper.GetNumericOperations<T>();
        L = Decompose(matrix, algorithm);
    }

    /// <summary>
    /// Solves the linear system Ax = b using the Cholesky decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// This method solves the system in two steps:
    /// 1. Forward substitution to solve Ly = b
    /// 2. Back substitution to solve L^Tx = y
    /// </remarks>
    public Vector<T> Solve(Vector<T> b)
    {
        var y = ForwardSubstitution(L, b);
        return BackSubstitution(L.Transpose(), y);
    }

    /// <summary>
    /// Selects and applies the appropriate Cholesky decomposition algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm type to use.</param>
    /// <returns>The lower triangular matrix L from the decomposition.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    private Matrix<T> Decompose(Matrix<T> matrix, CholeskyAlgorithmType algorithm)
    {
        return algorithm switch
        {
            CholeskyAlgorithmType.Crout => ComputeCholeskyCrout(matrix),
            CholeskyAlgorithmType.Banachiewicz => ComputeCholeskyBanachiewicz(matrix),
            CholeskyAlgorithmType.LDL => ComputeCholeskyLDL(matrix),
            CholeskyAlgorithmType.BlockCholesky => ComputeBlockCholesky(matrix),
            _ => throw new ArgumentException("Unsupported Cholesky decomposition algorithm.")
        };
    }

    /// <summary>
    /// Computes the Cholesky decomposition using a default approach.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>The lower triangular matrix L.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the matrix is not square, not symmetric, or not positive definite.
    /// </exception>
    private Matrix<T> ComputeCholeskyDefault(Matrix<T> matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");
        }

        int n = matrix.Rows;
        var L = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                if (i != j && !_numOps.Equals(matrix[i, j], matrix[j, i]))
                {
                    throw new ArgumentException("Matrix must be symmetric for Cholesky decomposition.");
                }

                T sum = _numOps.Zero;

                if (j == i) // Diagonal elements
                {
                    for (int k = 0; k < j; k++)
                    {
                        sum = _numOps.Add(sum, _numOps.Multiply(L[j, k], L[j, k]));
                    }
                    T diagonalValue = _numOps.Subtract(matrix[j, j], sum);
                    if (_numOps.LessThanOrEquals(diagonalValue, _numOps.Zero))
                    {
                        throw new ArgumentException("Matrix<double> is not positive definite.");
                    }
                    L[j, j] = _numOps.Sqrt(diagonalValue);
                }
                else // Lower triangular elements
                {
                    for (int k = 0; k < j; k++)
                    {
                        sum = _numOps.Add(sum, _numOps.Multiply(L[i, k], L[j, k]));
                    }
                    L[i, j] = _numOps.Divide(_numOps.Subtract(matrix[i, j], sum), L[j, j]);
                }
            }
        }

        return L;
    }

    /// <summary>
    /// Computes the Cholesky decomposition using the Crout algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>The lower triangular matrix L.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the matrix is not square or not positive definite.
    /// </exception>
    /// <remarks>
    /// The Crout algorithm computes the decomposition column by column, which can be
    /// more efficient for certain matrix structures.
    /// </remarks>
    private Matrix<T> ComputeCholeskyCrout(Matrix<T> matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");
        }

        int n = matrix.Rows;
        var L = new Matrix<T>(n, n);

        for (int j = 0; j < n; j++)
        {
            T sum = _numOps.Zero;
            for (int k = 0; k < j; k++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(L[j, k], L[j, k]));
            }

            T diagonalValue = _numOps.Subtract(matrix[j, j], sum);
            if (_numOps.LessThanOrEquals(diagonalValue, _numOps.Zero))
            {
                throw new ArgumentException("Matrix<double> is not positive definite.");
            }

            L[j, j] = _numOps.Sqrt(diagonalValue);

            for (int i = j + 1; i < n; i++)
            {
                sum = _numOps.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(L[i, k], L[j, k]));
                }

                L[i, j] = _numOps.Divide(_numOps.Subtract(matrix[i, j], sum), L[j, j]);
            }
        }

        return L;
    }

    /// <summary>
    /// Computes the Cholesky decomposition using the Banachiewicz algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>The lower triangular matrix L.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the matrix is not square or not positive definite.
    /// </exception>
    /// <remarks>
    /// The Banachiewicz algorithm computes the decomposition row by row, which can be
    /// more efficient for certain matrix structures.
    /// </remarks>
    private Matrix<T> ComputeCholeskyBanachiewicz(Matrix<T> matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");
        }

        int n = matrix.Rows;
        var L = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                T sum = _numOps.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(L[i, k], L[j, k]));
                }

                if (i == j)
                {
                    T diagonalValue = _numOps.Subtract(matrix[i, i], sum);
                    if (_numOps.LessThanOrEquals(diagonalValue, _numOps.Zero))
                    {
                        throw new ArgumentException("Matrix<double> is not positive definite.");
                    }
                    L[i, j] = _numOps.Sqrt(diagonalValue);
                }
                else
                {
                    L[i, j] = _numOps.Divide(_numOps.Subtract(matrix[i, j], sum), L[j, j]);
                }
            }
        }

        return L;
    }

        /// <summary>
    /// Computes the LDL' decomposition and converts it to Cholesky form.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>The lower triangular matrix L.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the matrix is not square or not positive definite.
    /// </exception>
    /// <remarks>
    /// The LDL' decomposition is a variant of Cholesky that avoids square roots until the final step.
    /// This can be more numerically stable for certain matrices.
    /// </remarks>
    private Matrix<T> ComputeCholeskyLDL(Matrix<T> matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");
        }

        int n = matrix.Rows;
        var L = Matrix<T>.CreateIdentity(n);
        var D = new Vector<T>(n);

        for (int j = 0; j < n; j++)
        {
            T d = matrix[j, j];
            for (int k = 0; k < j; k++)
            {
                d = _numOps.Subtract(d, _numOps.Multiply(_numOps.Multiply(L[j, k], L[j, k]), D[k]));
            }
            D[j] = d;

            for (int i = j + 1; i < n; i++)
            {
                T sum = matrix[i, j];
                for (int k = 0; k < j; k++)
                {
                    sum = _numOps.Subtract(sum, _numOps.Multiply(_numOps.Multiply(L[i, k], L[j, k]), D[k]));
                }
                L[i, j] = _numOps.Divide(sum, d);
            }
        }

        // Convert LDL' to LL'
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                L[i, j] = _numOps.Multiply(L[i, j], _numOps.Sqrt(D[j]));
            }
        }

        return L;
    }

    /// <summary>
    /// Computes the Cholesky decomposition using a block algorithm for large matrices.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>The lower triangular matrix L.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the matrix is not square or not positive definite.
    /// </exception>
    /// <remarks>
    /// The block Cholesky algorithm divides the matrix into blocks to improve cache efficiency
    /// and potentially leverage parallel processing for large matrices.
    /// </remarks>
    private Matrix<T> ComputeBlockCholesky(Matrix<T> matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");
        }

        int n = matrix.Rows;
        int blockSize = 64; // Adjust this based on your needs
        var L = new Matrix<T>(n, n);

        for (int i = 0; i < n; i += blockSize)
        {
            int size = Math.Min(blockSize, n - i);
            var subMatrix = matrix.SubMatrix(i, i, size, size);
            var subL = ComputeCholeskyCrout(subMatrix);

            for (int r = 0; r < size; r++)
            {
                for (int c = 0; c <= r; c++)
                {
                    L[i + r, i + c] = subL[r, c];
                }
            }

            if (i + size < n)
            {
                var B = matrix.SubMatrix(i + size, i, n - i - size, size);
                var subLCholesky = new CholeskyDecomposition<T>(L.SubMatrix(i, i, size, size));
                var X = subLCholesky.SolveMatrix(B.Transpose());

                for (int r = 0; r < n - i - size; r++)
                {
                    for (int c = 0; c < size; c++)
                    {
                        L[i + size + r, i + c] = X[c, r];
                    }
                }

                var C = matrix.SubMatrix(i + size, i + size, n - i - size, n - i - size);
                for (int r = 0; r < n - i - size; r++)
                {
                    for (int c = 0; c <= r; c++)
                    {
                        T sum = _numOps.Zero;
                        for (int k = 0; k < size; k++)
                        {
                            sum = _numOps.Add(sum, _numOps.Multiply(L[i + size + r, i + k], L[i + size + c, i + k]));
                        }
                        C[r, c] = _numOps.Subtract(C[r, c], sum);
                        C[c, r] = C[r, c];
                    }
                }
            }
        }

        return L;
    }

    /// <summary>
    /// Solves a system of linear equations for multiple right-hand sides.
    /// </summary>
    /// <param name="B">The matrix of right-hand sides.</param>
    /// <returns>The solution matrix X where A*X = B.</returns>
    private Matrix<T> SolveMatrix(Matrix<T> B)
    {
        int columns = B.Columns;
        var X = new Matrix<T>(L.Columns, columns);

        for (int i = 0; i < columns; i++)
        {
            var columnVector = B.GetColumn(i);
            var solutionVector = Solve(columnVector);
            X.SetColumn(i, solutionVector);
        }

        return X;
    }

    /// <summary>
    /// Performs forward substitution to solve L*y = b.
    /// </summary>
    /// <param name="L">The lower triangular matrix.</param>
    /// <param name="b">The right-hand side vector.</param>
    /// <returns>The solution vector y.</returns>
    private Vector<T> ForwardSubstitution(Matrix<T> L, Vector<T> b)
    {
        var y = new Vector<T>(L.Rows);
        for (int i = 0; i < L.Rows; i++)
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
    /// Performs back substitution to solve L^T*x = y.
    /// </summary>
    /// <param name="LT">The transpose of the lower triangular matrix.</param>
    /// <param name="y">The right-hand side vector.</param>
    /// <returns>The solution vector x.</returns>
    private Vector<T> BackSubstitution(Matrix<T> LT, Vector<T> y)
    {
        var x = new Vector<T>(LT.Columns);
        for (int i = LT.Columns - 1; i >= 0; i--)
        {
            T sum = _numOps.Zero;
            for (int j = i + 1; j < LT.Columns; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(LT[i, j], x[j]));
            }

            x[i] = _numOps.Divide(_numOps.Subtract(y[i], sum), LT[i, i]);
        }

        return x;
    }

    /// <summary>
    /// Computes the inverse of the original matrix using the Cholesky decomposition.
    /// </summary>
    /// <returns>The inverse of the original matrix A.</returns>
    /// <remarks>
    /// This method delegates to MatrixHelper to perform the inversion using this decomposition.
    /// </remarks>
    public Matrix<T> Invert()
    {
        return MatrixHelper<T>.InvertUsingDecomposition(this);
    }
}