namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements Hessenberg decomposition, which transforms a matrix into a form that is almost triangular.
/// </summary>
/// <remarks>
/// A Hessenberg matrix has zeros below the first subdiagonal, making it easier to work with for
/// many numerical algorithms. This decomposition is often used as a preprocessing step for
/// eigenvalue calculations and solving linear systems.
/// 
/// <b>For Beginners:</b> Think of Hessenberg decomposition as a way to simplify a matrix by making
/// most elements below the diagonal equal to zero, which makes further calculations much faster.
/// </remarks>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
public class HessenbergDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Gets the resulting Hessenberg matrix after decomposition.
    /// </summary>
    /// <remarks>
    /// A Hessenberg matrix has zeros in all positions below the first subdiagonal.
    /// This simplified form makes many matrix operations more efficient.
    /// </remarks>
    public Matrix<T> HessenbergMatrix { get; private set; }

    /// <summary>
    /// Gets the original matrix that was decomposed.
    /// </summary>
    public Matrix<T> A { get; private set; }

    /// <summary>
    /// Initializes a new instance of the Hessenberg decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for the decomposition (default is Householder).</param>
    /// <remarks>
    /// Different algorithms have different performance characteristics:
    /// - Householder: Generally the most stable and efficient for dense matrices
    /// - Givens: Good for sparse matrices or when only a few elements need to be zeroed
    /// - ElementaryTransformations: Simpler to understand but less numerically stable
    /// - ImplicitQR: Combines Hessenberg reduction with QR iteration
    /// - Lanczos: Efficient for large, sparse matrices
    /// 
    /// For beginners, the default Householder algorithm is recommended as it provides
    /// a good balance of stability and performance.
    /// </remarks>
    public HessenbergDecomposition(Matrix<T> matrix, HessenbergAlgorithmType algorithm = HessenbergAlgorithmType.Householder)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        A = matrix;
        HessenbergMatrix = Decompose(matrix, algorithm);
    }

    /// <summary>
    /// Selects and applies the appropriate decomposition algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for decomposition.</param>
    /// <returns>The resulting Hessenberg matrix.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    private Matrix<T> Decompose(Matrix<T> matrix, HessenbergAlgorithmType algorithm)
    {
        return algorithm switch
        {
            HessenbergAlgorithmType.Householder => ComputeHessenbergHouseholder(matrix),
            HessenbergAlgorithmType.Givens => ComputeHessenbergGivens(matrix),
            HessenbergAlgorithmType.ElementaryTransformations => ComputeHessenbergElementaryTransformations(matrix),
            HessenbergAlgorithmType.ImplicitQR => ComputeHessenbergImplicitQR(matrix),
            HessenbergAlgorithmType.Lanczos => ComputeHessenbergLanczos(matrix),
            _ => throw new ArgumentException("Unsupported Hessenberg decomposition algorithm.")
        };
    }

    /// <summary>
    /// Computes the Hessenberg form using Householder reflections.
    /// </summary>
    /// <param name="matrix">The matrix to transform.</param>
    /// <returns>The Hessenberg matrix.</returns>
    /// <remarks>
    /// Householder reflections are a way to zero out multiple elements of a matrix at once.
    /// 
    /// <b>For Beginners:</b> A Householder reflection is like holding up a mirror to a vector
    /// in such a way that it reflects to align with a target direction. This method:
    /// 1. Takes each column of the matrix
    /// 2. Creates a reflection that zeros out all elements below the first subdiagonal
    /// 3. Applies this reflection to the entire matrix
    /// 
    /// This is generally the most stable and efficient algorithm for Hessenberg decomposition.
    /// </remarks>
    private Matrix<T> ComputeHessenbergHouseholder(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = matrix.Clone();

        for (int k = 0; k < n - 2; k++)
        {
            var x = new Vector<T>(n - k - 1);
            for (int i = 0; i < n - k - 1; i++)
            {
                x[i] = H[k + 1 + i, k];
            }

            var v = MatrixHelper<T>.CreateHouseholderVector(x);
            H = MatrixHelper<T>.ApplyHouseholderTransformation(H, v, k);
        }

        return H;
    }

        /// <summary>
    /// Computes the Hessenberg form using Givens rotations.
    /// </summary>
    /// <param name="matrix">The matrix to transform.</param>
    /// <returns>The Hessenberg matrix.</returns>
    /// <remarks>
    /// Givens rotations are a way to zero out one element at a time by rotating in a plane.
    /// 
    /// <b>For Beginners:</b> A Givens rotation is like turning a 2D coordinate system to make
    /// one coordinate zero. This method:
    /// 1. For each column, starts from the bottom row
    /// 2. Applies rotations to zero out elements one by one, moving upward
    /// 3. Continues until only the first subdiagonal and above have non-zero elements
    /// 
    /// Givens rotations are useful when you only need to zero out a few specific elements.
    /// </remarks>
    private Matrix<T> ComputeHessenbergGivens(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = matrix.Clone();

        for (int k = 0; k < n - 2; k++)
        {
            for (int i = n - 1; i > k + 1; i--)
            {
                var (c, s) = MatrixHelper<T>.ComputeGivensRotation(H[i - 1, k], H[i, k]);
                MatrixHelper<T>.ApplyGivensRotation(H, c, s, i - 1, i, k, n);
            }
        }

        return H;
    }

    /// <summary>
    /// Computes the Hessenberg form using elementary row transformations.
    /// </summary>
    /// <param name="matrix">The matrix to transform.</param>
    /// <returns>The Hessenberg matrix.</returns>
    /// <remarks>
    /// Elementary transformations modify one row at a time by adding multiples of other rows.
    /// 
    /// <b>For Beginners:</b> This method is similar to Gaussian elimination that you might have
    /// learned in basic linear algebra. It works by:
    /// 1. Taking each column from left to right
    /// 2. For each element below the first subdiagonal, calculating a factor
    /// 3. Using that factor to add a multiple of one row to another to create zeros
    /// 
    /// This is the most straightforward algorithm to understand but may be less numerically
    /// stable than Householder or Givens methods.
    /// </remarks>
    private Matrix<T> ComputeHessenbergElementaryTransformations(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = matrix.Clone();

        for (int k = 0; k < n - 2; k++)
        {
            for (int i = k + 2; i < n; i++)
            {
                if (!_numOps.Equals(H[i, k], _numOps.Zero))
                {
                    T factor = _numOps.Divide(H[i, k], H[k + 1, k]);
                    for (int j = k; j < n; j++)
                    {
                        H[i, j] = _numOps.Subtract(H[i, j], _numOps.Multiply(factor, H[k + 1, j]));
                    }
                    H[i, k] = _numOps.Zero;
                }
            }
        }

        return H;
    }

    /// <summary>
    /// Computes the Hessenberg form using the implicit QR algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to transform.</param>
    /// <returns>The Hessenberg matrix.</returns>
    /// <remarks>
    /// The implicit QR algorithm combines Hessenberg reduction with QR iteration.
    /// 
    /// <b>For Beginners:</b> This advanced method:
    /// 1. Starts with an initial approximation
    /// 2. Iteratively refines the matrix using Givens rotations
    /// 3. Continues until the matrix converges to Hessenberg form
    /// 
    /// This method is particularly useful when you need both the Hessenberg form
    /// and eigenvalues, as it makes progress toward both simultaneously.
    /// </remarks>
    private Matrix<T> ComputeHessenbergImplicitQR(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = matrix.Clone();
        var Q = Matrix<T>.CreateIdentity(n);

        for (int iter = 0; iter < 100; iter++) // Max iterations
        {
            for (int k = 0; k < n - 1; k++)
            {
                var (c, s) = MatrixHelper<T>.ComputeGivensRotation(H[k, k], H[k + 1, k]);
                MatrixHelper<T>.ApplyGivensRotation(H, c, s, k, k + 1, k, n);
                MatrixHelper<T>.ApplyGivensRotation(Q, c, s, k, k + 1, 0, n);
            }

            if (MatrixHelper<T>.IsUpperHessenberg(H, _numOps.FromDouble(1e-10)))
            {
                break;
            }
        }

        return H;
    }

    /// <summary>
    /// Computes the Hessenberg form using the Lanczos algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to transform.</param>
    /// <returns>The Hessenberg matrix.</returns>
    /// <remarks>
    /// The Lanczos algorithm is an iterative method that is particularly efficient for large, sparse matrices.
    /// 
    /// <b>For Beginners:</b> This method:
    /// 1. Starts with a single vector
    /// 2. Repeatedly multiplies by the original matrix to generate new vectors
    /// 3. Orthogonalizes these vectors to create a basis
    /// 4. Represents the original matrix in this new basis, resulting in a Hessenberg form
    /// 
    /// This approach is especially useful when dealing with very large matrices where
    /// other methods would be too computationally expensive.
    /// </remarks>
    private Matrix<T> ComputeHessenbergLanczos(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = new Matrix<T>(n, n);
        var v = new Vector<T>(n)
        {
            [0] = _numOps.One
        };

        for (int j = 0; j < n; j++)
        {
            var w = matrix.Multiply(v);
            if (j > 0)
            {
                w = w.Subtract(v.Multiply(H[j - 1, j]));
            }

            H[j, j] = w.DotProduct(v);
            w = w.Subtract(v.Multiply(H[j, j]));
            if (j < n - 1)
            {
                H[j, j + 1] = H[j + 1, j] = w.Norm();
                v = w.Divide(H[j, j + 1]);
            }
        }

        return H;
    }

    /// <summary>
    /// Solves a linear system Ax = b using the Hessenberg decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// This method solves the linear system by using the special structure of the Hessenberg matrix.
    /// 
    /// <b>For Beginners:</b> Solving a linear system means finding values for x that satisfy the equation Ax = b.
    /// This method:
    /// 1. First performs forward substitution (working from top to bottom) to solve an intermediate system
    /// 2. Then performs backward substitution (working from bottom to top) to find the final solution
    /// 
    /// The Hessenberg form makes this process more efficient than solving with the original matrix.
    /// </remarks>
    public Vector<T> Solve(Vector<T> b)
    {
        var n = A.Rows;
        var y = new Vector<T>(n);

        // Forward substitution
        for (int i = 0; i < n; i++)
        {
            var sum = _numOps.Zero;
            for (int j = Math.Max(0, i - 1); j < i; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(HessenbergMatrix[i, j], y[j]));
            }

            y[i] = _numOps.Divide(_numOps.Subtract(b[i], sum), HessenbergMatrix[i, i]);
        }

        // Backward substitution
        var x = new Vector<T>(n);
        for (int i = n - 1; i >= 0; i--)
        {
            var sum = _numOps.Zero;
            for (int j = i + 1; j < n; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(HessenbergMatrix[i, j], x[j]));
            }

            x[i] = _numOps.Subtract(y[i], sum);
        }

        return x;
    }

    /// <summary>
    /// Calculates the inverse of the original matrix using the Hessenberg decomposition.
    /// </summary>
    /// <returns>The inverse of the original matrix.</returns>
    /// <remarks>
    /// Matrix inversion finds a matrix A?� such that A � A?� = I (identity matrix).
    /// 
    /// <b>For Beginners:</b> The inverse of a matrix is like the reciprocal of a number.
    /// Just as 5 � (1/5) = 1, a matrix multiplied by its inverse gives the identity matrix.
    /// 
    /// This method uses the MatrixHelper class to efficiently compute the inverse
    /// based on the Hessenberg decomposition, which is generally faster than
    /// directly inverting the original matrix.
    /// </remarks>
    public Matrix<T> Invert()
    {
        return MatrixHelper<T>.InvertUsingDecomposition(this);
    }
}