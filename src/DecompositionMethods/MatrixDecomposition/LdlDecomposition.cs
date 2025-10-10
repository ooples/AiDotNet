namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Performs LDL decomposition on a symmetric matrix, factoring it into a lower triangular matrix L
/// and a diagonal matrix D such that A = LDL^T.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> LDL decomposition breaks down a symmetric matrix into simpler parts:
/// - L: A lower triangular matrix (has values only on and below the diagonal)
/// - D: A diagonal matrix (has values only on the diagonal)
/// 
/// This decomposition is useful for solving linear systems, calculating determinants,
/// and inverting matrices more efficiently than working with the original matrix.
/// </remarks>
public class LdlDecomposition<T> : IMatrixDecomposition<T>
{
    /// <summary>
    /// Provides operations for the numeric type T (addition, multiplication, etc.)
    /// </summary>
    private readonly INumericOperations<T> _numOps = default!;

    /// <summary>
    /// The original matrix being decomposed.
    /// </summary>
    public Matrix<T> A { get; }
    
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
    {
        if (!matrix.IsSquareMatrix())
            throw new ArgumentException("Matrix must be square for LDL decomposition.");

        _numOps = MathHelper.GetNumericOperations<T>();
        A = matrix;
        int n = A.Rows;
        L = new Matrix<T>(n, n);
        D = new Vector<T>(n);
        Decompose(algorithm);
    }

    /// <summary>
    /// Performs the LDL decomposition using the specified algorithm.
    /// </summary>
    /// <param name="algorithm">The algorithm to use for decomposition (default is Cholesky).</param>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method does the actual work of breaking down the original matrix
    /// into the L and D components. Different algorithms may be more efficient for different
    /// types of matrices.
    /// </remarks>
    public void Decompose(LdlAlgorithmType algorithm = LdlAlgorithmType.Cholesky)
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
            T sum = _numOps.Zero;
            for (int k = 0; k < j; k++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_numOps.Multiply(L[j, k], L[j, k]), D[k]));
            }
            D[j] = _numOps.Subtract(A[j, j], sum);

            L[j, j] = _numOps.One;

            for (int i = j + 1; i < n; i++)
            {
                sum = _numOps.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(_numOps.Multiply(L[i, k], L[j, k]), D[k]));
                }

                L[i, j] = _numOps.Divide(_numOps.Subtract(A[i, j], sum), D[j]);
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
            T sum = _numOps.Zero;
            for (int k = 0; k < j; k++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(_numOps.Multiply(L[j, k], L[j, k]), D[k]));
            }

            D[j] = _numOps.Subtract(A[j, j], sum);
            L[j, j] = _numOps.One;

            for (int i = j + 1; i < n; i++)
            {
                sum = _numOps.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum = _numOps.Add(sum, _numOps.Multiply(_numOps.Multiply(L[i, k], L[j, k]), D[k]));
                }

                L[i, j] = _numOps.Divide(_numOps.Subtract(A[i, j], sum), D[j]);
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
    public Vector<T> Solve(Vector<T> b)
    {
        if (b.Length != A.Rows)
            throw new ArgumentException("Vector<double> b must have the same length as the number of rows in matrix A.");

        // Forward substitution
        Vector<T> y = new(b.Length);
        for (int i = 0; i < b.Length; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < i; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(L[i, j], y[j]));
            }
            y[i] = _numOps.Subtract(b[i], sum);
        }

        // Diagonal scaling
        for (int i = 0; i < b.Length; i++)
        {
            y[i] = _numOps.Divide(y[i], D[i]);
        }

        // Backward substitution
        Vector<T> x = new Vector<T>(b.Length);
        for (int i = b.Length - 1; i >= 0; i--)
        {
            T sum = _numOps.Zero;
            for (int j = i + 1; j < b.Length; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(L[j, i], x[j]));
            }
            x[i] = _numOps.Subtract(y[i], sum);
        }

        return x;
    }

    /// <summary>
    /// Calculates the inverse of the original matrix using the LDL decomposition.
    /// </summary>
    /// <returns>The inverse of the original matrix A.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The inverse of a matrix A is another matrix A⁻¹ such that when multiplied 
    /// together, they give the identity matrix (A × A⁻¹ = I).
    /// 
    /// This method computes the inverse by:
    /// 1. Creating a set of unit vectors (vectors with a single 1 and the rest 0s)
    /// 2. Solving the system Ax = e_i for each unit vector e_i
    /// 3. Combining these solutions as columns to form the inverse matrix
    /// 
    /// Using LDL decomposition for this process is more efficient and numerically stable
    /// than directly computing the inverse through other methods.
    /// </remarks>
    public Matrix<T> Invert()
    {
        int n = A.Rows;
        Matrix<T> inverse = new(n, n);

        for (int i = 0; i < n; i++)
        {
            Vector<T> ei = new(n)
            {
                [i] = _numOps.One  // Fixed to use _numOps instead of NumOps
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