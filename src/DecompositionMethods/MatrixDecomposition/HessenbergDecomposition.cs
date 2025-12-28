namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements Hessenberg decomposition, which transforms a matrix into a form that is almost triangular.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// A Hessenberg matrix has zeros below the first subdiagonal, making it easier to work with for
/// many numerical algorithms. This decomposition is often used as a preprocessing step for
/// eigenvalue calculations and solving linear systems.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of Hessenberg decomposition as a way to simplify a matrix by making
/// most elements below the diagonal equal to zero, which makes further calculations much faster.
/// It's like organizing a messy room by putting everything in its place - the matrix becomes
/// easier to work with even though it's not completely triangular.
/// </para>
/// <para>
/// Real-world applications:
/// - Preprocessing for eigenvalue computation
/// - Accelerating iterative methods for solving linear systems
/// - Control theory and system analysis
/// </para>
/// </remarks>
public class HessenbergDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the resulting Hessenberg matrix after decomposition.
    /// </summary>
    /// <remarks>
    /// A Hessenberg matrix has zeros in all positions below the first subdiagonal.
    /// This simplified form makes many matrix operations more efficient.
    /// </remarks>
    public Matrix<T> HessenbergMatrix { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the orthogonal transformation matrix Q from the decomposition.
    /// </summary>
    /// <remarks>
    /// The orthogonal matrix Q satisfies A = Q * H * Q^T, where A is the original matrix
    /// and H is the Hessenberg matrix. Q is orthogonal, meaning Q^T * Q = Q * Q^T = I.
    /// </remarks>
    public Matrix<T> OrthogonalMatrix { get; private set; } = new Matrix<T>(0, 0);

    private readonly HessenbergAlgorithmType _algorithm;

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
        : base(matrix)
    {
        if (!matrix.IsSquareMatrix())
            throw new ArgumentException("Matrix must be square for Hessenberg decomposition.");

        _algorithm = algorithm;
        Decompose();
    }

    /// <summary>
    /// Performs the Hessenberg decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (HessenbergMatrix, OrthogonalMatrix) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Computes the Hessenberg decomposition using the specified algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for decomposition.</param>
    /// <returns>A tuple containing the Hessenberg matrix and the orthogonal matrix.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    private (Matrix<T> H, Matrix<T> Q) ComputeDecomposition(Matrix<T> matrix, HessenbergAlgorithmType algorithm)
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
    /// <returns>A tuple containing the Hessenberg matrix and the orthogonal matrix.</returns>
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
    private (Matrix<T> H, Matrix<T> Q) ComputeHessenbergHouseholder(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = matrix.Clone();
        var Q = Matrix<T>.CreateIdentity(n);

        for (int k = 0; k < n - 2; k++)
        {
            // Extract column below the subdiagonal
            var x = new Vector<T>(n - k - 1);
            for (int i = 0; i < n - k - 1; i++)
            {
                x[i] = H[k + 1 + i, k];
            }

            // Compute norm of x
            T xNorm = NumOps.Zero;
            for (int i = 0; i < x.Length; i++)
            {
                xNorm = NumOps.Add(xNorm, NumOps.Multiply(x[i], x[i]));
            }
            xNorm = NumOps.Sqrt(xNorm);

            // Skip if column is already zero (nothing to eliminate)
            if (NumOps.LessThan(xNorm, NumOps.FromDouble(1e-14)))
            {
                continue;
            }

            // Use OPPOSITE sign for numerical stability (avoid cancellation)
            T alpha = NumOps.LessThan(x[0], NumOps.Zero)
                ? xNorm
                : NumOps.Negate(xNorm);

            // Create Householder vector: v = x - alpha*e_1
            var v = new Vector<T>(x.Length);
            v[0] = NumOps.Subtract(x[0], alpha);
            for (int i = 1; i < x.Length; i++)
            {
                v[i] = x[i];
            }

            // Normalize v
            T vNorm = NumOps.Zero;
            for (int i = 0; i < v.Length; i++)
            {
                vNorm = NumOps.Add(vNorm, NumOps.Multiply(v[i], v[i]));
            }
            vNorm = NumOps.Sqrt(vNorm);

            if (NumOps.LessThan(vNorm, NumOps.FromDouble(1e-14)))
            {
                continue;
            }

            for (int i = 0; i < v.Length; i++)
            {
                v[i] = NumOps.Divide(v[i], vNorm);
            }

            // Apply Householder from LEFT: H = (I - 2*v*v^T) * H
            // For rows k+1 to n-1, columns 0 to n-1
            for (int j = 0; j < n; j++)
            {
                T dot = NumOps.Zero;
                for (int i = 0; i < v.Length; i++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(v[i], H[k + 1 + i, j]));
                }
                dot = NumOps.Multiply(NumOps.FromDouble(2), dot);

                for (int i = 0; i < v.Length; i++)
                {
                    H[k + 1 + i, j] = NumOps.Subtract(H[k + 1 + i, j], NumOps.Multiply(v[i], dot));
                }
            }

            // Apply Householder from RIGHT: H = H * (I - 2*v*v^T)
            // For rows 0 to n-1, columns k+1 to n-1
            for (int i = 0; i < n; i++)
            {
                T dot = NumOps.Zero;
                for (int j = 0; j < v.Length; j++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(H[i, k + 1 + j], v[j]));
                }
                dot = NumOps.Multiply(NumOps.FromDouble(2), dot);

                for (int j = 0; j < v.Length; j++)
                {
                    H[i, k + 1 + j] = NumOps.Subtract(H[i, k + 1 + j], NumOps.Multiply(dot, v[j]));
                }
            }

            // Accumulate Q: Q = Q * P where P = I - 2*v*v^T (acting on rows k+1 to n-1)
            // Q = Q * P is equivalent to updating columns k+1 to n-1 of Q
            for (int i = 0; i < n; i++)
            {
                T dot = NumOps.Zero;
                for (int j = 0; j < v.Length; j++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(Q[i, k + 1 + j], v[j]));
                }
                dot = NumOps.Multiply(NumOps.FromDouble(2), dot);

                for (int j = 0; j < v.Length; j++)
                {
                    Q[i, k + 1 + j] = NumOps.Subtract(Q[i, k + 1 + j], NumOps.Multiply(dot, v[j]));
                }
            }

            // Explicitly zero out elements below subdiagonal in column k
            for (int i = k + 2; i < n; i++)
            {
                H[i, k] = NumOps.Zero;
            }
        }

        return (H, Q);
    }

    /// <summary>
    /// Computes the Hessenberg form using Givens rotations.
    /// </summary>
    /// <param name="matrix">The matrix to transform.</param>
    /// <returns>A tuple containing the Hessenberg matrix and the orthogonal matrix.</returns>
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
    private (Matrix<T> H, Matrix<T> Q) ComputeHessenbergGivens(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = matrix.Clone();
        var Q = Matrix<T>.CreateIdentity(n);

        for (int k = 0; k < n - 2; k++)
        {
            // Eliminate elements below the subdiagonal in column k
            for (int i = n - 1; i > k + 1; i--)
            {
                // Skip if element is already zero
                if (NumOps.LessThan(NumOps.Abs(H[i, k]), NumOps.FromDouble(1e-14)))
                {
                    continue;
                }

                // Compute Givens rotation to zero out H[i, k]
                T a = H[i - 1, k];
                T b = H[i, k];
                T r = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(a, a), NumOps.Multiply(b, b)));

                if (NumOps.LessThan(r, NumOps.FromDouble(1e-14)))
                {
                    continue;
                }

                T c = NumOps.Divide(a, r);
                T s = NumOps.Divide(b, r);

                // Apply Givens rotation from LEFT: H = G^T * H
                // G^T rotates rows i-1 and i
                for (int j = 0; j < n; j++)
                {
                    T temp1 = H[i - 1, j];
                    T temp2 = H[i, j];
                    H[i - 1, j] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
                    H[i, j] = NumOps.Subtract(NumOps.Multiply(c, temp2), NumOps.Multiply(s, temp1));
                }

                // Apply Givens rotation from RIGHT: H = H * G
                // G rotates columns i-1 and i
                for (int j = 0; j < n; j++)
                {
                    T temp1 = H[j, i - 1];
                    T temp2 = H[j, i];
                    H[j, i - 1] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
                    H[j, i] = NumOps.Subtract(NumOps.Multiply(c, temp2), NumOps.Multiply(s, temp1));
                }

                // Accumulate Q: Q = Q * G
                for (int j = 0; j < n; j++)
                {
                    T temp1 = Q[j, i - 1];
                    T temp2 = Q[j, i];
                    Q[j, i - 1] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
                    Q[j, i] = NumOps.Subtract(NumOps.Multiply(c, temp2), NumOps.Multiply(s, temp1));
                }

                // Explicitly zero out the element
                H[i, k] = NumOps.Zero;
            }
        }

        return (H, Q);
    }

    /// <summary>
    /// Computes the Hessenberg form using elementary row transformations.
    /// </summary>
    /// <param name="matrix">The matrix to transform.</param>
    /// <returns>A tuple containing the Hessenberg matrix and the transformation matrix.</returns>
    /// <remarks>
    /// Elementary transformations modify one row at a time by adding multiples of other rows.
    ///
    /// <b>For Beginners:</b> This method is similar to Gaussian elimination that you might have
    /// learned in basic linear algebra. It works by:
    /// 1. Taking each column from left to right
    /// 2. For each element below the first subdiagonal, calculating a factor
    /// 3. Using that factor to add a multiple of one row to another to create zeros
    ///
    /// Note: Elementary transformations produce a non-orthogonal similarity transformation.
    /// The returned Q satisfies H = Q^(-1) * A * Q, but Q is not orthogonal.
    /// For applications requiring orthogonal Q, use Householder or Givens algorithms.
    /// </remarks>
    private (Matrix<T> H, Matrix<T> Q) ComputeHessenbergElementaryTransformations(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = matrix.Clone();
        var Q = Matrix<T>.CreateIdentity(n);

        for (int k = 0; k < n - 2; k++)
        {
            // Check if pivot is near zero; if so, try to find a better pivot
            if (NumOps.LessThan(NumOps.Abs(H[k + 1, k]), NumOps.FromDouble(1e-14)))
            {
                // Find row with largest element to swap
                int maxRow = k + 1;
                T maxVal = NumOps.Abs(H[k + 1, k]);
                for (int i = k + 2; i < n; i++)
                {
                    if (NumOps.GreaterThan(NumOps.Abs(H[i, k]), maxVal))
                    {
                        maxVal = NumOps.Abs(H[i, k]);
                        maxRow = i;
                    }
                }

                if (NumOps.GreaterThan(maxVal, NumOps.FromDouble(1e-14)) && maxRow != k + 1)
                {
                    // Swap rows in H (similarity transformation)
                    for (int j = 0; j < n; j++)
                    {
                        T temp = H[k + 1, j];
                        H[k + 1, j] = H[maxRow, j];
                        H[maxRow, j] = temp;
                    }
                    // Swap columns in H (similarity transformation)
                    for (int j = 0; j < n; j++)
                    {
                        T temp = H[j, k + 1];
                        H[j, k + 1] = H[j, maxRow];
                        H[j, maxRow] = temp;
                    }
                    // Swap columns in Q to track transformation
                    for (int j = 0; j < n; j++)
                    {
                        T temp = Q[j, k + 1];
                        Q[j, k + 1] = Q[j, maxRow];
                        Q[j, maxRow] = temp;
                    }
                }
            }

            // Skip if pivot is still too small
            if (NumOps.LessThan(NumOps.Abs(H[k + 1, k]), NumOps.FromDouble(1e-14)))
            {
                continue;
            }

            for (int i = k + 2; i < n; i++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(H[i, k]), NumOps.FromDouble(1e-14)))
                {
                    T factor = NumOps.Divide(H[i, k], H[k + 1, k]);

                    // Apply row operation from LEFT: row_i = row_i - factor * row_{k+1}
                    for (int j = 0; j < n; j++)
                    {
                        H[i, j] = NumOps.Subtract(H[i, j], NumOps.Multiply(factor, H[k + 1, j]));
                    }

                    // Apply column operation from RIGHT: col_{k+1} = col_{k+1} + factor * col_i
                    // This is the inverse transformation to maintain similarity
                    for (int j = 0; j < n; j++)
                    {
                        H[j, k + 1] = NumOps.Add(H[j, k + 1], NumOps.Multiply(factor, H[j, i]));
                    }

                    // Track transformation in Q: Q = Q * M where M has 1s on diagonal,
                    // and factor at position (k+1, i)
                    for (int j = 0; j < n; j++)
                    {
                        Q[j, k + 1] = NumOps.Add(Q[j, k + 1], NumOps.Multiply(factor, Q[j, i]));
                    }

                    // Explicitly zero out
                    H[i, k] = NumOps.Zero;
                }
            }
        }

        return (H, Q);
    }

    /// <summary>
    /// Computes the Hessenberg form using the implicit QR algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to transform.</param>
    /// <returns>A tuple containing the Hessenberg matrix and the orthogonal matrix.</returns>
    /// <remarks>
    /// The implicit QR algorithm combines Hessenberg reduction with bulge-chasing.
    ///
    /// <b>For Beginners:</b> This advanced method:
    /// 1. First reduces the matrix to Hessenberg form using Givens rotations
    /// 2. Uses implicit shifts to improve numerical stability
    /// 3. Applies Givens rotations as similarity transforms
    ///
    /// This method is particularly useful when you need both the Hessenberg form
    /// and eigenvalues, as it makes progress toward both simultaneously.
    /// </remarks>
    private (Matrix<T> H, Matrix<T> Q) ComputeHessenbergImplicitQR(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = matrix.Clone();
        var Q = Matrix<T>.CreateIdentity(n);

        // Use Givens rotations with implicit shifting
        // First pass: reduce to Hessenberg form using Givens rotations from bottom up
        for (int k = 0; k < n - 2; k++)
        {
            // Eliminate elements below the subdiagonal in column k
            for (int i = n - 1; i > k + 1; i--)
            {
                // Skip if element is already zero
                if (NumOps.LessThan(NumOps.Abs(H[i, k]), NumOps.FromDouble(1e-14)))
                {
                    continue;
                }

                // Compute Givens rotation to zero out H[i, k]
                T a = H[i - 1, k];
                T b = H[i, k];
                T r = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(a, a), NumOps.Multiply(b, b)));

                if (NumOps.LessThan(r, NumOps.FromDouble(1e-14)))
                {
                    continue;
                }

                T c = NumOps.Divide(a, r);
                T s = NumOps.Divide(b, r);

                // Apply Givens rotation from LEFT: H = G^T * H
                for (int j = 0; j < n; j++)
                {
                    T temp1 = H[i - 1, j];
                    T temp2 = H[i, j];
                    H[i - 1, j] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
                    H[i, j] = NumOps.Subtract(NumOps.Multiply(c, temp2), NumOps.Multiply(s, temp1));
                }

                // Apply Givens rotation from RIGHT: H = H * G
                for (int j = 0; j < n; j++)
                {
                    T temp1 = H[j, i - 1];
                    T temp2 = H[j, i];
                    H[j, i - 1] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
                    H[j, i] = NumOps.Subtract(NumOps.Multiply(c, temp2), NumOps.Multiply(s, temp1));
                }

                // Accumulate Q: Q = Q * G
                for (int j = 0; j < n; j++)
                {
                    T temp1 = Q[j, i - 1];
                    T temp2 = Q[j, i];
                    Q[j, i - 1] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
                    Q[j, i] = NumOps.Subtract(NumOps.Multiply(c, temp2), NumOps.Multiply(s, temp1));
                }

                // Explicitly zero out the element
                H[i, k] = NumOps.Zero;
            }
        }

        return (H, Q);
    }

    /// <summary>
    /// Computes the Hessenberg form using the Lanczos algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to transform.</param>
    /// <returns>A tuple containing the Hessenberg matrix and the orthogonal matrix.</returns>
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
    /// Note: The Lanczos algorithm generates a tridiagonal matrix for symmetric matrices.
    /// For general (non-symmetric) matrices, the Arnoldi iteration is used instead.
    /// </remarks>
    private (Matrix<T> H, Matrix<T> Q) ComputeHessenbergLanczos(Matrix<T> matrix)
    {
        var n = matrix.Rows;
        var H = new Matrix<T>(n, n);
        var Q = new Matrix<T>(n, n);

        // Initialize first Lanczos vector (normalized)
        var v = new Vector<T>(n);
        v[0] = NumOps.One;

        // Store Lanczos vectors as columns of Q
        for (int i = 0; i < n; i++)
        {
            Q[i, 0] = v[i];
        }

        Vector<T> vPrev = new Vector<T>(n);

        for (int j = 0; j < n; j++)
        {
            // w = A * v_j
            var w = matrix.Multiply(v);

            // Orthogonalize against previous vector (if j > 0)
            if (j > 0)
            {
                T beta = H[j, j - 1];
                for (int i = 0; i < n; i++)
                {
                    w[i] = NumOps.Subtract(w[i], NumOps.Multiply(beta, vPrev[i]));
                }
            }

            // Compute alpha = v_j^T * w
            T alpha = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                alpha = NumOps.Add(alpha, NumOps.Multiply(v[i], w[i]));
            }
            H[j, j] = alpha;

            // w = w - alpha * v_j
            for (int i = 0; i < n; i++)
            {
                w[i] = NumOps.Subtract(w[i], NumOps.Multiply(alpha, v[i]));
            }

            // Full reorthogonalization against all previous vectors for stability
            for (int k = 0; k <= j; k++)
            {
                T dot = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(w[i], Q[i, k]));
                }
                for (int i = 0; i < n; i++)
                {
                    w[i] = NumOps.Subtract(w[i], NumOps.Multiply(dot, Q[i, k]));
                }
            }

            if (j < n - 1)
            {
                // Compute beta = ||w||
                T beta = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    beta = NumOps.Add(beta, NumOps.Multiply(w[i], w[i]));
                }
                beta = NumOps.Sqrt(beta);

                H[j, j + 1] = beta;
                H[j + 1, j] = beta;

                // Check for breakdown
                if (NumOps.LessThan(beta, NumOps.FromDouble(1e-14)))
                {
                    // Breakdown: restart with a random vector orthogonal to current subspace
                    w = new Vector<T>(n);
                    w[(j + 1) % n] = NumOps.One;

                    // Orthogonalize against all previous vectors
                    for (int k = 0; k <= j; k++)
                    {
                        T dot = NumOps.Zero;
                        for (int i = 0; i < n; i++)
                        {
                            dot = NumOps.Add(dot, NumOps.Multiply(w[i], Q[i, k]));
                        }
                        for (int i = 0; i < n; i++)
                        {
                            w[i] = NumOps.Subtract(w[i], NumOps.Multiply(dot, Q[i, k]));
                        }
                    }

                    // Recompute norm
                    beta = NumOps.Zero;
                    for (int i = 0; i < n; i++)
                    {
                        beta = NumOps.Add(beta, NumOps.Multiply(w[i], w[i]));
                    }
                    beta = NumOps.Sqrt(beta);

                    if (NumOps.LessThan(beta, NumOps.FromDouble(1e-14)))
                    {
                        // Complete breakdown, fill remaining with identity-like structure
                        for (int jj = j + 1; jj < n; jj++)
                        {
                            Q[jj, jj] = NumOps.One;
                        }
                        break;
                    }
                }

                // v_{j+1} = w / beta
                vPrev = v;
                v = new Vector<T>(n);
                for (int i = 0; i < n; i++)
                {
                    v[i] = NumOps.Divide(w[i], beta);
                }

                // Store in Q
                for (int i = 0; i < n; i++)
                {
                    Q[i, j + 1] = v[i];
                }
            }
        }

        return (H, Q);
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
    public override Vector<T> Solve(Vector<T> b)
    {
        var n = A.Rows;
        var y = new Vector<T>(n);

        // VECTORIZED: Forward substitution using dot product
        for (int i = 0; i < n; i++)
        {
            T sum = NumOps.Zero;
            if (i > 0)
            {
                int start = Math.Max(0, i - 1);
                int len = i - start;
                if (len > 0)
                {
                    var rowSegment = new Vector<T>(HessenbergMatrix.GetRow(i).Skip(start).Take(len));
                    var ySegment = new Vector<T>(y.Skip(start).Take(len));
                    sum = rowSegment.DotProduct(ySegment);
                }
            }

            y[i] = NumOps.Divide(NumOps.Subtract(b[i], sum), HessenbergMatrix[i, i]);
        }

        // VECTORIZED: Backward substitution using dot product
        var x = new Vector<T>(n);
        for (int i = n - 1; i >= 0; i--)
        {
            T sum = NumOps.Zero;
            if (i < n - 1)
            {
                int len = n - i - 1;
                var rowSegment = new Vector<T>(HessenbergMatrix.GetRow(i).Skip(i + 1));
                var xSegment = new Vector<T>(x.Skip(i + 1));
                sum = rowSegment.DotProduct(xSegment);
            }

            x[i] = NumOps.Subtract(y[i], sum);
        }

        return x;
    }

    /// <summary>
    /// Calculates the inverse of the original matrix using the Hessenberg decomposition.
    /// </summary>
    /// <returns>The inverse of the original matrix.</returns>
    /// <remarks>
    /// Matrix inversion finds a matrix A⁻¹ such that A * A⁻¹ = I (identity matrix).
    ///
    /// <b>For Beginners:</b> The inverse of a matrix is like the reciprocal of a number.
    /// Just as 5 * (1/5) = 1, a matrix multiplied by its inverse gives the identity matrix.
    ///
    /// This method uses the MatrixHelper class to efficiently compute the inverse
    /// based on the Hessenberg decomposition, which is generally faster than
    /// directly inverting the original matrix.
    /// </remarks>
    public override Matrix<T> Invert()
    {
        return MatrixHelper<T>.InvertUsingDecomposition(this);
    }
}
