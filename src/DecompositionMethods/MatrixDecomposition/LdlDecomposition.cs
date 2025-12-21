namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Performs LDL decomposition on a symmetric matrix, factoring it into a lower triangular matrix L
/// and a diagonal matrix D such that A = LDL^T.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// LDL decomposition factors a symmetric matrix A into the product A = LDL^T, where L is a lower
/// triangular matrix with ones on the diagonal, and D is a diagonal matrix. This decomposition is
/// particularly useful for symmetric matrices and avoids computing square roots, making it more
/// numerically stable than Cholesky decomposition in some cases.
/// </para>
/// <para>
/// <b>For Beginners:</b> LDL decomposition breaks down a symmetric matrix into simpler parts:
/// L (a lower triangular matrix with values only on and below the diagonal) and D (a diagonal matrix
/// with values only on the diagonal). This decomposition is useful for solving linear systems,
/// calculating determinants, and inverting matrices more efficiently than working with the original matrix.
/// </para>
/// <para>
/// Real-world applications:
/// - Solving systems of linear equations in optimization
/// - Covariance matrix analysis in statistics
/// - Kalman filtering in signal processing
/// </para>
/// </remarks>
public class LdlDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// The lower triangular matrix L from the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a matrix where all elements above the diagonal are zero.
    /// The diagonal elements are all set to 1.
    /// </remarks>
    public Matrix<T> L { get; private set; }

    /// <summary>
    /// The diagonal matrix D from the decomposition, stored as a vector of the diagonal elements.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Instead of storing a full matrix with zeros everywhere except the diagonal,
    /// we just store the diagonal values in a vector for efficiency.
    /// </remarks>
    public Vector<T> D { get; private set; }

    private readonly LdlAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the LDL decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for decomposition (default is Cholesky).</param>
    /// <exception cref="ArgumentException">Thrown when the matrix is not square.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor takes your original matrix and immediately performs
    /// the decomposition using the specified algorithm.
    /// </remarks>
    public LdlDecomposition(Matrix<T> matrix, LdlAlgorithmType algorithm = LdlAlgorithmType.Cholesky)
        : base(matrix)
    {
        if (!matrix.IsSquareMatrix())
            throw new ArgumentException("Matrix must be square for LDL decomposition.");

        _algorithm = algorithm;
        int n = A.Rows;
        L = new Matrix<T>(n, n);
        D = new Vector<T>(n);
        Decompose();
    }

    /// <summary>
    /// Performs the LDL decomposition.
    /// </summary>
    protected override void Decompose()
    {
        ComputeDecomposition(_algorithm);
    }


    /// <summary>
    /// Performs the actual decomposition computation using the specified algorithm.
    /// </summary>
    /// <param name="algorithm">The algorithm to use for decomposition.</param>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    private void ComputeDecomposition(LdlAlgorithmType algorithm)
    {
        switch (algorithm)
        {
            case LdlAlgorithmType.Cholesky:
                DecomposeCholesky();
                break;
            case LdlAlgorithmType.Crout:
                DecomposeCrout();
                break;
            default:
                throw new ArgumentException("Unsupported LDL decomposition algorithm.");
        }
    }

    /// <summary>
    /// Performs LDL decomposition using the Cholesky-based algorithm.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method implements a variant of the Cholesky decomposition algorithm
    /// that produces the LDL factorization. It works by:
    /// 1. Computing each diagonal element of D
    /// 2. Setting the diagonal of L to 1
    /// 3. Computing each below-diagonal element of L
    /// 
    /// The algorithm is particularly efficient for symmetric, positive-definite matrices.
    /// </remarks>
    private void DecomposeCholesky()
    {
        int n = A.Rows;
        L = new Matrix<T>(n, n);
        D = new Vector<T>(n);

        for (int j = 0; j < n; j++)
        {
            // VECTORIZED: Calculate D[j] using dot product
            T sum = NumOps.Zero;
            if (j > 0)
            {
                var ljRow = new T[j];
                var dSlice = new T[j];
                for (int k = 0; k < j; k++)
                {
                    ljRow[k] = NumOps.Multiply(L[j, k], L[j, k]);
                    dSlice[k] = D[k];
                }
                var ljVec = new Vector<T>(ljRow);
                var dVec = new Vector<T>(dSlice);
                sum = ljVec.DotProduct(dVec);
            }
            D[j] = NumOps.Subtract(A[j, j], sum);

            L[j, j] = NumOps.One;

            for (int i = j + 1; i < n; i++)
            {
                // VECTORIZED: Calculate L[i,j] using dot product
                sum = NumOps.Zero;
                if (j > 0)
                {
                    var liRow = new T[j];
                    var ljRow = new T[j];
                    var dSlice = new T[j];
                    for (int k = 0; k < j; k++)
                    {
                        liRow[k] = L[i, k];
                        ljRow[k] = L[j, k];
                        dSlice[k] = D[k];
                    }
                    // Compute element-wise product then sum
                    for (int k = 0; k < j; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(liRow[k], ljRow[k]), dSlice[k]));
                    }
                }

                L[i, j] = NumOps.Divide(NumOps.Subtract(A[i, j], sum), D[j]);
            }
        }
    }

    /// <summary>
    /// Performs LDL decomposition using the Crout algorithm.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The Crout algorithm is another method for computing the LDL decomposition.
    /// It follows a similar approach to the Cholesky-based algorithm but may be more numerically
    /// stable for certain types of matrices.
    /// 
    /// This implementation computes the L and D matrices column by column, working from left to right.
    /// </remarks>
    private void DecomposeCrout()
    {
        int n = A.Rows;
        L = new Matrix<T>(n, n);
        D = new Vector<T>(n);

        for (int j = 0; j < n; j++)
        {
            // VECTORIZED: Calculate D[j] using dot product
            T sum = NumOps.Zero;
            if (j > 0)
            {
                var ljRow = new T[j];
                var dSlice = new T[j];
                for (int k = 0; k < j; k++)
                {
                    ljRow[k] = NumOps.Multiply(L[j, k], L[j, k]);
                    dSlice[k] = D[k];
                }
                var ljVec = new Vector<T>(ljRow);
                var dVec = new Vector<T>(dSlice);
                sum = ljVec.DotProduct(dVec);
            }

            D[j] = NumOps.Subtract(A[j, j], sum);
            L[j, j] = NumOps.One;

            for (int i = j + 1; i < n; i++)
            {
                // VECTORIZED: Calculate L[i,j] using element-wise product and sum
                sum = NumOps.Zero;
                if (j > 0)
                {
                    var liRow = new T[j];
                    var ljRow = new T[j];
                    var dSlice = new T[j];
                    for (int k = 0; k < j; k++)
                    {
                        liRow[k] = L[i, k];
                        ljRow[k] = L[j, k];
                        dSlice[k] = D[k];
                    }
                    // Compute element-wise product then sum
                    for (int k = 0; k < j; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(liRow[k], ljRow[k]), dSlice[k]));
                    }
                }

                L[i, j] = NumOps.Divide(NumOps.Subtract(A[i, j], sum), D[j]);
            }
        }
    }

    /// <summary>
    /// Solves the linear system Ax = b using the LDL decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector length doesn't match the matrix dimensions.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method finds the values of x that satisfy the equation Ax = b.
    /// It uses the LDL decomposition to solve this in three steps:
    ///
    /// 1. Forward substitution: Solve Ly = b for y
    /// 2. Diagonal scaling: Solve Dz = y for z
    /// 3. Backward substitution: Solve L^T x = z for x
    ///
    /// This approach is much more efficient than directly inverting the matrix A.
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        if (b.Length != A.Rows)
            throw new ArgumentException("Vector b must have the same length as the number of rows in matrix A.");

        // Forward substitution
        Vector<T> y = new(b.Length);
        for (int i = 0; i < b.Length; i++)
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
            y[i] = NumOps.Subtract(b[i], sum);
        }

        // VECTORIZED: Diagonal scaling using vector division
        y = y.ElementwiseDivide(D);

        // Backward substitution
        Vector<T> x = new Vector<T>(b.Length);
        for (int i = b.Length - 1; i >= 0; i--)
        {
            // VECTORIZED: Use dot product for sum computation
            T sum = NumOps.Zero;
            if (i < b.Length - 1)
            {
                int remaining = b.Length - i - 1;
                var colSlice = new T[remaining];
                var xSlice = new T[remaining];
                for (int k = 0; k < remaining; k++)
                {
                    colSlice[k] = L[i + 1 + k, i];
                    xSlice[k] = x[i + 1 + k];
                }
                var colVec = new Vector<T>(colSlice);
                var xVec = new Vector<T>(xSlice);
                sum = colVec.DotProduct(xVec);
            }
            x[i] = NumOps.Subtract(y[i], sum);
        }

        return x;
    }

    /// <summary>
    /// Calculates the inverse of the original matrix using the LDL decomposition.
    /// </summary>
    /// <returns>The inverse of the original matrix A.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The inverse of a matrix A is another matrix A⁻¹ such that when multiplied
    /// together, they give the identity matrix (A * A⁻¹ = I).
    ///
    /// This method computes the inverse by:
    /// 1. Creating a set of unit vectors (vectors with a single 1 and the rest 0s)
    /// 2. Solving the system Ax = e_i for each unit vector e_i
    /// 3. Combining these solutions as columns to form the inverse matrix
    ///
    /// Using LDL decomposition for this process is more efficient and numerically stable
    /// than directly computing the inverse through other methods.
    /// </remarks>
    public override Matrix<T> Invert()
    {
        int n = A.Rows;
        Matrix<T> inverse = new(n, n);

        for (int i = 0; i < n; i++)
        {
            Vector<T> ei = new(n)
            {
                [i] = NumOps.One
            };
            Vector<T> column = Solve(ei);
            for (int j = 0; j < n; j++)
            {
                inverse[j, i] = column[j];
            }
        }

        return inverse;
    }

    /// <summary>
    /// Returns the L and D factors from the decomposition.
    /// </summary>
    /// <returns>A tuple containing the lower triangular matrix L and the diagonal vector D.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method provides direct access to the two components of the LDL decomposition:
    /// - L: The lower triangular matrix with 1s on the diagonal
    /// - D: The diagonal elements stored as a vector
    /// 
    /// These components can be used for various mathematical operations or to reconstruct
    /// the original matrix using the formula A = LDL^T.
    /// </remarks>
    public (Matrix<T> L, Vector<T> D) GetFactors()
    {
        return (L, D);
    }
}
