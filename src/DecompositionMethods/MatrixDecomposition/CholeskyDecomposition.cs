namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements the Cholesky decomposition for symmetric positive definite matrices.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// The Cholesky decomposition breaks down a symmetric positive definite matrix into
/// the product of a lower triangular matrix and its transpose (A = L * L^T).
/// This is useful for solving linear systems and matrix inversion more efficiently.
/// </para>
/// <para>
/// <b>For Beginners:</b> Cholesky decomposition is like taking the square root of a matrix.
/// Just as 25 = 5 × 5, this method breaks down special matrices into L × L^T, where L is
/// a simpler triangular matrix. This makes complex calculations much faster and more accurate.
/// </para>
/// <para>
/// Real-world applications:
/// - Solving systems of linear equations in scientific computing
/// - Covariance matrix decomposition in statistics and machine learning
/// - Efficient simulation of correlated random variables
/// </para>
/// </remarks>
public class CholeskyDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the lower triangular matrix L from the decomposition A = L * L^T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The L matrix has zeros above the main diagonal and contains
    /// all the information needed to reconstruct your original matrix. Think of it as
    /// one of the "building blocks" that combine to recreate the original matrix.
    /// </para>
    /// </remarks>
    public Matrix<T> L { get; private set; } = new Matrix<T>(0, 0);

    private readonly CholeskyAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the Cholesky decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The symmetric positive definite matrix to decompose.</param>
    /// <param name="algorithm">The specific Cholesky algorithm to use (default is Crout).</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the matrix is not square, not symmetric, or not positive definite.
    /// </exception>
    /// <remarks>
    /// <para>
    /// A positive definite matrix has all positive eigenvalues, which means it has a unique
    /// Cholesky decomposition. In practical terms, for most matrices you encounter in data
    /// science, this means the matrix must be symmetric with positive values on the diagonal.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This constructor checks that your matrix meets the requirements
    /// (symmetric and positive definite) and then breaks it down using the selected algorithm.
    /// Different algorithms may have different performance characteristics depending on your matrix.
    /// </para>
    /// </remarks>
    public CholeskyDecomposition(Matrix<T> matrix, CholeskyAlgorithmType algorithm = CholeskyAlgorithmType.Crout)
        : base(matrix)
    {
        _algorithm = algorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the Cholesky decomposition.
    /// </summary>
    protected override void Decompose()
    {
        L = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Solves the linear system Ax = b using the Cholesky decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <para>
    /// This method solves the system in two steps:
    /// 1. Forward substitution to solve Ly = b
    /// 2. Back substitution to solve L^Tx = y
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This solves equations of the form Ax = b, where A is your matrix,
    /// b is known, and x is what you're finding. It uses the decomposition to solve this
    /// efficiently in two simple steps, much faster than directly solving the original equation.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
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
    private Matrix<T> ComputeDecomposition(Matrix<T> matrix, CholeskyAlgorithmType algorithm)
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
    /// <remarks>
    /// <para>
    /// <b>VECTORIZED:</b> This method uses vectorized dot product operations for inner loop sums,
    /// improving performance while maintaining the sequential outer loop structure required by
    /// Cholesky's strong data dependencies.
    /// </para>
    /// </remarks>
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
                if (i != j && !NumOps.Equals(matrix[i, j], matrix[j, i]))
                {
                    throw new ArgumentException("Matrix must be symmetric for Cholesky decomposition.");
                }

                if (j == i) // Diagonal elements
                {
                    // VECTORIZED: compute sum of squares using dot product
                    T sum = NumOps.Zero;
                    if (j > 0)
                    {
                        var rowJ = L.GetRow(j);
                        var rowJSegment = new Vector<T>(rowJ.Take(j));
                        sum = rowJSegment.DotProduct(rowJSegment);
                    }

                    T diagonalValue = NumOps.Subtract(matrix[j, j], sum);
                    if (NumOps.LessThanOrEquals(diagonalValue, NumOps.Zero))
                    {
                        throw new ArgumentException("Matrix is not positive definite.");
                    }
                    L[j, j] = NumOps.Sqrt(diagonalValue);
                }
                else // Lower triangular elements
                {
                    // VECTORIZED: compute dot product of L[i,:j] and L[j,:j]
                    T sum = NumOps.Zero;
                    if (j > 0)
                    {
                        var rowI = L.GetRow(i);
                        var rowJ = L.GetRow(j);
                        var rowISegment = new Vector<T>(rowI.Take(j));
                        var rowJSegment = new Vector<T>(rowJ.Take(j));
                        sum = rowISegment.DotProduct(rowJSegment);
                    }

                    L[i, j] = NumOps.Divide(NumOps.Subtract(matrix[i, j], sum), L[j, j]);
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
    /// <para>
    /// The Crout algorithm computes the decomposition column by column, which can be
    /// more efficient for certain matrix structures.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The Crout method processes the matrix one column at a time,
    /// building up the solution gradually. It's like solving a puzzle by completing
    /// one vertical section before moving to the next.
    /// </para>
    /// <para>
    /// <b>VECTORIZED:</b> This method uses vectorized dot product operations for computing
    /// row sums, replacing scalar loops with efficient vector operations while maintaining
    /// the column-by-column processing order required by the algorithm.
    /// </para>
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
            // VECTORIZED: compute diagonal element using dot product
            T sum = NumOps.Zero;
            if (j > 0)
            {
                var rowJ = L.GetRow(j);
                var rowJSegment = new Vector<T>(rowJ.Take(j));
                sum = rowJSegment.DotProduct(rowJSegment);
            }

            T diagonalValue = NumOps.Subtract(matrix[j, j], sum);
            if (NumOps.LessThanOrEquals(diagonalValue, NumOps.Zero))
            {
                throw new ArgumentException("Matrix is not positive definite.");
            }
            L[j, j] = NumOps.Sqrt(diagonalValue);

            // VECTORIZED: compute off-diagonal elements using dot product
            if (j > 0)
            {
                var rowJ = L.GetRow(j);
                var rowJSegment = new Vector<T>(rowJ.Take(j));

                for (int i = j + 1; i < n; i++)
                {
                    var rowI = L.GetRow(i);
                    var rowISegment = new Vector<T>(rowI.Take(j));
                    sum = rowISegment.DotProduct(rowJSegment);
                    L[i, j] = NumOps.Divide(NumOps.Subtract(matrix[i, j], sum), L[j, j]);
                }
            }
            else
            {
                // First column: no dot product needed
                for (int i = j + 1; i < n; i++)
                {
                    L[i, j] = NumOps.Divide(matrix[i, j], L[j, j]);
                }
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
    /// <para>
    /// The Banachiewicz algorithm computes the decomposition row by row, which can be
    /// more efficient for certain matrix structures.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The Banachiewicz method processes the matrix one row at a time,
    /// building up the solution gradually. It's like solving a puzzle by completing
    /// one horizontal section before moving to the next.
    /// </para>
    /// <para>
    /// <b>VECTORIZED:</b> This method uses vectorized dot product operations to compute
    /// row element contributions, replacing scalar accumulation loops with efficient
    /// vector operations while maintaining row-by-row processing order.
    /// </para>
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
                // VECTORIZED: compute dot product of L[i,:j] and L[j,:j]
                T sum = NumOps.Zero;
                if (j > 0)
                {
                    var rowI = L.GetRow(i);
                    var rowJ = L.GetRow(j);
                    var rowISegment = new Vector<T>(rowI.Take(j));
                    var rowJSegment = new Vector<T>(rowJ.Take(j));
                    sum = rowISegment.DotProduct(rowJSegment);
                }

                if (i == j)
                {
                    T diagonalValue = NumOps.Subtract(matrix[i, i], sum);
                    if (NumOps.LessThanOrEquals(diagonalValue, NumOps.Zero))
                    {
                        throw new ArgumentException("Matrix is not positive definite.");
                    }
                    L[i, j] = NumOps.Sqrt(diagonalValue);
                }
                else
                {
                    L[i, j] = NumOps.Divide(NumOps.Subtract(matrix[i, j], sum), L[j, j]);
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
    /// <para>
    /// The LDL' decomposition is a variant of Cholesky that avoids square roots until the final step.
    /// This can be more numerically stable for certain matrices.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method avoids calculating square roots until the very end,
    /// which can give more accurate results for certain matrices. It's like doing all the
    /// easy arithmetic first and saving the complicated square root calculations for last.
    /// </para>
    /// <para>
    /// <b>VECTORIZED:</b> This method uses vectorized operations with element-wise multiplication
    /// and dot products for computing weighted sums, replacing scalar accumulation loops.
    /// The final conversion to LL' form uses vectorized row operations.
    /// </para>
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
            if (j > 0)
            {
                // VECTORIZED: compute weighted sum using element-wise operations
                var rowJ = L.GetRow(j);
                var rowJSegment = new Vector<T>(rowJ.Take(j));
                var dSegment = new Vector<T>(D.Take(j));

                // Compute L[j,k]^2 * D[k] for k < j
                var squaredRow = new Vector<T>(rowJSegment.Select(x => NumOps.Multiply(x, x)));
                var weighted = new Vector<T>(squaredRow.Zip(dSegment, (l, dVal) => NumOps.Multiply(l, dVal)));
                var weightedSum = weighted.Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));

                d = NumOps.Subtract(d, weightedSum);
            }
            D[j] = d;

            for (int i = j + 1; i < n; i++)
            {
                T sum = matrix[i, j];
                if (j > 0)
                {
                    // VECTORIZED: compute weighted dot product
                    var rowI = L.GetRow(i);
                    var rowJ = L.GetRow(j);
                    var rowISegment = new Vector<T>(rowI.Take(j));
                    var rowJSegment = new Vector<T>(rowJ.Take(j));
                    var dSegment = new Vector<T>(D.Take(j));

                    // Compute sum of L[i,k] * L[j,k] * D[k] for k < j
                    var product = new Vector<T>(rowISegment.Zip(rowJSegment, (li, lj) => NumOps.Multiply(li, lj)));
                    var weighted = new Vector<T>(product.Zip(dSegment, (p, dVal) => NumOps.Multiply(p, dVal)));
                    var weightedSum = weighted.Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));

                    sum = NumOps.Subtract(sum, weightedSum);
                }
                L[i, j] = NumOps.Divide(sum, d);
            }
        }

        // Convert LDL' to LL' - vectorized per row
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                L[i, j] = NumOps.Multiply(L[i, j], NumOps.Sqrt(D[j]));
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
    /// <para>
    /// The block Cholesky algorithm divides the matrix into blocks to improve cache efficiency
    /// and potentially leverage parallel processing for large matrices.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method breaks large matrices into smaller chunks (blocks)
    /// that fit better in your computer's memory cache, making it much faster for big matrices.
    /// Think of it like eating a large meal by breaking it into smaller, manageable bites.
    /// </para>
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
                // VECTORIZED: compute row updates using dot products
                for (int r = 0; r < n - i - size; r++)
                {
                    for (int c = 0; c <= r; c++)
                    {
                        // VECTORIZED: compute dot product of L[i+size+r, i:i+size] and L[i+size+c, i:i+size]
                        T sum = NumOps.Zero;
                        if (size > 0)
                        {
                            var rowR = L.GetRow(i + size + r);
                            var rowC = L.GetRow(i + size + c);
                            var rowRSegment = new Vector<T>(rowR.Skip(i).Take(size));
                            var rowCSegment = new Vector<T>(rowC.Skip(i).Take(size));
                            sum = rowRSegment.DotProduct(rowCSegment);
                        }
                        C[r, c] = NumOps.Subtract(C[r, c], sum);
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
    /// <remarks>
    /// <para>
    /// <b>VECTORIZED:</b> Uses vectorized dot product for computing row contributions,
    /// replacing scalar accumulation loops while maintaining the sequential dependency order.
    /// </para>
    /// </remarks>
    private Vector<T> ForwardSubstitution(Matrix<T> L, Vector<T> b)
    {
        var y = new Vector<T>(L.Rows);
        for (int i = 0; i < L.Rows; i++)
        {
            // VECTORIZED: compute dot product of L[i,:i] and y[:i]
            T sum = NumOps.Zero;
            if (i > 0)
            {
                var rowL = L.GetRow(i);
                var rowLSegment = new Vector<T>(rowL.Take(i));
                var ySegment = new Vector<T>(y.Take(i));
                sum = rowLSegment.DotProduct(ySegment);
            }

            y[i] = NumOps.Divide(NumOps.Subtract(b[i], sum), L[i, i]);
        }

        return y;
    }

    /// <summary>
    /// Performs back substitution to solve L^T*x = y.
    /// </summary>
    /// <param name="LT">The transpose of the lower triangular matrix.</param>
    /// <param name="y">The right-hand side vector.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <para>
    /// <b>VECTORIZED:</b> Uses vectorized dot product for computing row contributions,
    /// replacing scalar accumulation loops while maintaining the sequential dependency order.
    /// </para>
    /// </remarks>
    private Vector<T> BackSubstitution(Matrix<T> LT, Vector<T> y)
    {
        var x = new Vector<T>(LT.Columns);
        for (int i = LT.Columns - 1; i >= 0; i--)
        {
            // VECTORIZED: compute dot product of LT[i, i+1:] and x[i+1:]
            T sum = NumOps.Zero;
            if (i < LT.Columns - 1)
            {
                var rowLT = LT.GetRow(i);
                var rowLTSegment = new Vector<T>(rowLT.Skip(i + 1));
                var xSegment = new Vector<T>(x.Skip(i + 1));
                sum = rowLTSegment.DotProduct(xSegment);
            }

            x[i] = NumOps.Divide(NumOps.Subtract(y[i], sum), LT[i, i]);
        }

        return x;
    }
}
