namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Performs eigenvalue decomposition of a matrix, breaking it down into its eigenvalues and eigenvectors.
/// </summary>
/// <typeparam name="T">The numeric data type used in calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Eigenvalue decomposition is a way to factorize a square matrix into a set of eigenvectors and eigenvalues.
/// This is useful in many applications including principal component analysis, vibration analysis,
/// and solving systems of differential equations.
/// </para>
/// <para>
/// <b>For Beginners:</b> Eigenvalue decomposition finds special directions (eigenvectors) in which a matrix
/// acts like simple scaling. When you multiply the matrix by an eigenvector, you get the same vector back
/// but scaled by a number (the eigenvalue). Think of it like finding the "natural directions" of a transformation.
/// </para>
/// <para>
/// Real-world applications:
/// - Principal Component Analysis (PCA) for data dimensionality reduction
/// - Vibration analysis in mechanical engineering
/// - Google's PageRank algorithm
/// </para>
/// </remarks>
public class EigenDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the eigenvectors of the decomposed matrix.
    /// </summary>
    /// <remarks>
    /// Eigenvectors are special vectors that, when multiplied by the original matrix,
    /// result in a vector that points in the same direction but may be scaled.
    /// Each eigenvector corresponds to an eigenvalue in the EigenValues property.
    /// </remarks>
    public Matrix<T> EigenVectors { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the eigenvalues of the decomposed matrix.
    /// </summary>
    /// <remarks>
    /// Eigenvalues are special scalars that, when the original matrix multiplies its corresponding eigenvector,
    /// the result is the same as scaling the eigenvector by this value.
    /// Each eigenvalue corresponds to an eigenvector in the EigenVectors property.
    /// </remarks>
    public Vector<T> EigenValues { get; private set; } = new Vector<T>(0);

    private readonly EigenAlgorithmType _algorithm;

    /// <summary>
    /// Creates a new eigenvalue decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for eigenvalue decomposition. Defaults to QR algorithm.</param>
    /// <remarks>
    /// Different algorithms have different trade-offs in terms of speed, accuracy, and applicability:
    /// - QR: Generally robust and accurate for most matrices.
    /// - PowerIteration: Simpler but may be slower for finding all eigenvalues.
    /// - Jacobi: Works well for symmetric matrices.
    /// </remarks>
    public EigenDecomposition(Matrix<T> matrix, EigenAlgorithmType algorithm = EigenAlgorithmType.QR)
        : base(matrix)
    {
        ValidateMatrix(matrix, requireSquare: true);
        _algorithm = algorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the eigenvalue decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (EigenValues, EigenVectors) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Selects and applies the appropriate eigenvalue decomposition algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for eigenvalue decomposition.</param>
    /// <returns>A tuple containing the eigenvalues and eigenvectors of the matrix.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeDecomposition(Matrix<T> matrix, EigenAlgorithmType algorithm)
    {
        return algorithm switch
        {
            EigenAlgorithmType.QR => ComputeEigenQR(matrix),
            EigenAlgorithmType.PowerIteration => ComputeEigenPowerIteration(matrix),
            EigenAlgorithmType.Jacobi => ComputeEigenJacobi(matrix),
            _ => throw new ArgumentException("Unsupported eigenvalue decomposition algorithm.")
        };
    }

    /// <summary>
    /// Computes eigenvalues and eigenvectors using the Power Iteration method.
    /// </summary>
    /// <remarks>
    /// The Power Iteration method works by repeatedly multiplying a vector by the matrix
    /// and normalizing the result. This process converges to the dominant eigenvector
    /// (the one with the largest eigenvalue in absolute terms).
    /// 
    /// This implementation uses deflation to find multiple eigenvalues and eigenvectors.
    /// After finding each eigenvector, we modify the matrix to remove the influence of
    /// that eigenvector, allowing us to find the next most dominant one.
    /// </remarks>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the eigenvalues and eigenvectors.</returns>
    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEigenPowerIteration(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Vector<T> eigenValues = new(n);
        Matrix<T> eigenVectors = Matrix<T>.CreateIdentity(n);

        Matrix<T> deflated = matrix.Clone();
        int maxIterations = 2000;
        T tolerance = NumOps.FromDouble(1e-14);

        for (int i = 0; i < n; i++)
        {
            // Start with a vector that has all ones - more stable than random
            Vector<T> v = new Vector<T>(n);
            for (int j = 0; j < n; j++)
            {
                v[j] = NumOps.One;
            }
            // Normalize the initial vector
            T initNorm = v.Norm();
            if (NumOps.GreaterThan(initNorm, NumOps.FromDouble(1e-14)))
            {
                v = v.Divide(initNorm);
            }

            T eigenValue = NumOps.Zero;
            bool converged = false;

            // Phase 1: Standard power iteration
            for (int iter = 0; iter < maxIterations && !converged; iter++)
            {
                // Orthogonalize against previously found eigenvectors
                for (int j = 0; j < i; j++)
                {
                    Vector<T> prevV = eigenVectors.GetColumn(j);
                    T projection = v.DotProduct(prevV);
                    v = v.Subtract(prevV.Multiply(projection));
                }
                T vNorm = v.Norm();
                if (NumOps.LessThan(vNorm, NumOps.FromDouble(1e-14)))
                {
                    // Vector became zero, reinitialize
                    for (int j = 0; j < n; j++) v[j] = NumOps.FromDouble(j + 1);
                    v = v.Divide(v.Norm());
                    continue;
                }
                v = v.Divide(vNorm);

                // Power iteration step: v = A*v / ||A*v||
                Vector<T> w = deflated.Multiply(v);
                T norm = w.Norm();

                // Check for zero norm (indicates zero eigenvalue or numerical issues)
                if (NumOps.LessThan(norm, NumOps.FromDouble(1e-14)))
                {
                    eigenValues[i] = NumOps.Zero;
                    converged = true;
                    break;
                }

                Vector<T> vNew = w.Divide(norm);

                // Compute Rayleigh quotient: λ = v^T * A * v
                Vector<T> Av = deflated.Multiply(vNew);
                T newEigenValue = vNew.DotProduct(Av);

                // Check convergence: ||Av - λv|| should be small
                Vector<T> residual = Av.Subtract(vNew.Multiply(newEigenValue));
                T residualNorm = residual.Norm();

                if (NumOps.LessThan(residualNorm, tolerance))
                {
                    v = vNew;
                    eigenValue = newEigenValue;
                    eigenValues[i] = eigenValue;
                    converged = true;
                    break;
                }

                v = vNew;
                eigenValue = newEigenValue;
                eigenValues[i] = eigenValue;
            }

            // Phase 2: Rayleigh quotient iteration for refinement (if not converged)
            if (!converged)
            {
                for (int iter = 0; iter < 200; iter++)
                {
                    Vector<T> Av = deflated.Multiply(v);
                    eigenValue = v.DotProduct(Av);

                    // Solve (A - λI)w = v using shifted matrix
                    // For better convergence, we use the residual direction
                    Vector<T> residual = Av.Subtract(v.Multiply(eigenValue));
                    T residualNorm = residual.Norm();

                    if (NumOps.LessThan(residualNorm, tolerance))
                    {
                        eigenValues[i] = eigenValue;
                        break;
                    }

                    // Update v in the direction of the residual (gradient descent on Rayleigh quotient)
                    // v_new = (A*v) / ||A*v||
                    T avNorm = Av.Norm();
                    if (NumOps.GreaterThan(avNorm, NumOps.FromDouble(1e-14)))
                    {
                        v = Av.Divide(avNorm);
                    }
                    eigenValues[i] = eigenValue;
                }
            }

            eigenVectors.SetColumn(i, v);

            // Deflate: remove the found eigenvalue/eigenvector from the matrix
            // A_new = A - λ * v * v^T
            deflated = deflated.Subtract(MatrixHelper<T>.OuterProduct(v, v).Multiply(eigenValues[i]));
        }

        return (eigenValues, eigenVectors);
    }

    /// <summary>
    /// Computes eigenvalues and eigenvectors using the QR algorithm with Wilkinson shift.
    /// </summary>
    /// <remarks>
    /// The QR algorithm is an iterative method that works by repeatedly factoring the matrix
    /// into a product Q*R (where Q is orthogonal and R is upper triangular), and then
    /// recombining as R*Q. This process eventually converges to a matrix where the eigenvalues
    /// appear on the diagonal.
    ///
    /// This implementation uses Wilkinson shift for faster convergence on symmetric matrices.
    /// </remarks>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the eigenvalues and eigenvectors.</returns>
    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEigenQR(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Matrix<T> A = matrix.Clone();
        Matrix<T> Q = Matrix<T>.CreateIdentity(n);

        T tolerance = NumOps.FromDouble(1e-12);
        int maxIterations = 200;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Check if already diagonal (for symmetric) or upper triangular
            bool converged = true;
            for (int i = 1; i < n && converged; i++)
            {
                for (int j = 0; j < i && converged; j++)
                {
                    if (NumOps.GreaterThan(NumOps.Abs(A[i, j]), tolerance))
                    {
                        converged = false;
                    }
                }
            }
            if (converged) break;

            // Wilkinson shift: compute shift from bottom 2x2 submatrix
            T shift = NumOps.Zero;
            if (n >= 2)
            {
                T a = A[n - 2, n - 2];
                T b = A[n - 2, n - 1];
                T c = A[n - 1, n - 2];
                T d = A[n - 1, n - 1];

                // Compute eigenvalue of 2x2 block closest to d
                T delta = NumOps.Divide(NumOps.Subtract(a, d), NumOps.FromDouble(2.0));
                T signDelta = NumOps.GreaterThanOrEquals(delta, NumOps.Zero) ? NumOps.One : NumOps.FromDouble(-1.0);
                T bc = NumOps.Multiply(b, c);
                T sqrtTerm = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(delta, delta), bc));

                if (NumOps.GreaterThan(NumOps.Abs(sqrtTerm), NumOps.FromDouble(1e-14)))
                {
                    shift = NumOps.Subtract(d, NumOps.Divide(NumOps.Multiply(signDelta, bc),
                        NumOps.Add(NumOps.Abs(delta), sqrtTerm)));
                }
                else
                {
                    shift = d;
                }
            }

            // Shifted QR step: A - shift*I = Q*R, then A = R*Q + shift*I
            Matrix<T> shiftedA = A.Clone();
            for (int i = 0; i < n; i++)
            {
                shiftedA[i, i] = NumOps.Subtract(shiftedA[i, i], shift);
            }

            var qrDecomp = new QrDecomposition<T>(shiftedA);
            (var q, var r) = (qrDecomp.Q, qrDecomp.R);

            A = r.Multiply(q);
            for (int i = 0; i < n; i++)
            {
                A[i, i] = NumOps.Add(A[i, i], shift);
            }

            Q = Q.Multiply(q);
        }

        Vector<T> eigenValues = MatrixHelper<T>.ExtractDiagonal(A);
        return (eigenValues, Q);
    }

    /// <summary>
    /// Computes eigenvalues and eigenvectors using the Jacobi method.
    /// </summary>
    /// <remarks>
    /// The Jacobi method is an iterative algorithm particularly effective for symmetric matrices.
    /// It works by performing a series of rotations (Jacobi rotations) that gradually make the
    /// off-diagonal elements of the matrix smaller, eventually converging to a diagonal matrix
    /// where the diagonal elements are the eigenvalues.
    /// 
    /// In simple terms, this method systematically eliminates the largest off-diagonal element
    /// in each iteration until all off-diagonal elements are close to zero.
    /// </remarks>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the eigenvalues and eigenvectors of the matrix.</returns>
    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEigenJacobi(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Matrix<T> A = matrix.Clone();
        Matrix<T> V = Matrix<T>.CreateIdentity(n);

        for (int iter = 0; iter < 100; iter++)
        {
            // VECTORIZED: Find the largest off-diagonal element using row operations
            T maxOffDiagonal = NumOps.Zero;
            int p = 0, q = 0;

            for (int i = 0; i < n - 1; i++)
            {
                // VECTORIZED: Extract upper triangular portion of row and find max
                Vector<T> rowSegment = new Vector<T>(A.GetRow(i).Skip(i + 1).Take(n - i - 1));
                for (int k = 0; k < rowSegment.Length; k++)
                {
                    T absValue = NumOps.Abs(rowSegment[k]);
                    if (NumOps.GreaterThan(absValue, maxOffDiagonal))
                    {
                        maxOffDiagonal = absValue;
                        p = i;
                        q = i + 1 + k;
                    }
                }
            }

            // Check if we've reached the desired precision
            if (NumOps.LessThan(maxOffDiagonal, NumOps.FromDouble(1e-6)))
                break;

            // Calculate the Jacobi rotation parameters
            T theta = NumOps.Divide(NumOps.Subtract(A[q, q], A[p, p]), NumOps.Multiply(NumOps.FromDouble(2), A[p, q]));
            T t = NumOps.Divide(NumOps.SignOrZero(theta), NumOps.Add(NumOps.Abs(theta), NumOps.Sqrt(NumOps.Add(NumOps.One, NumOps.Multiply(theta, theta)))));
            T c = NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.Add(NumOps.One, NumOps.Multiply(t, t))));
            T s = NumOps.Multiply(t, c);

            // Create the Jacobi rotation matrix
            Matrix<T> J = Matrix<T>.CreateIdentity(n);
            J[p, p] = c; J[q, q] = c;
            J[p, q] = s; J[q, p] = NumOps.Negate(s);

            // Apply the rotation to A and accumulate in V
            A = J.Transpose().Multiply(A).Multiply(J);
            V = V.Multiply(J);
        }

        Vector<T> eigenValues = MatrixHelper<T>.ExtractDiagonal(A);
        return (eigenValues, V);
    }

    /// <summary>
    /// Solves a system of linear equations Ax = b using the eigenvalue decomposition.
    /// </summary>
    /// <remarks>
    /// This method uses the eigenvalue decomposition to solve the system of equations.
    /// It works by transforming the problem into the eigenvector basis, where the system
    /// becomes diagonal and easy to solve, then transforming back to the original basis.
    ///
    /// The solution is computed as: x = V * D⁻¹ * V^T * b
    /// where V is the matrix of eigenvectors, D is a diagonal matrix of eigenvalues,
    /// and V^T is the transpose of V.
    /// </remarks>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    public override Vector<T> Solve(Vector<T> b)
    {
        Matrix<T> D = Matrix<T>.CreateDiagonal(EigenValues);
        return EigenVectors.Multiply(D.InvertDiagonalMatrix()).Multiply(EigenVectors.Transpose()).Multiply(b);
    }

    /// <summary>
    /// Computes the inverse of the original matrix using the eigenvalue decomposition.
    /// </summary>
    /// <remarks>
    /// This method uses the eigenvalue decomposition to compute the inverse of the matrix.
    /// The inverse is calculated as: A⁻¹ = V * D⁻¹ * V^T
    /// where V is the matrix of eigenvectors, D is a diagonal matrix of eigenvalues,
    /// and V^T is the transpose of V.
    ///
    /// This approach can be more numerically stable than directly inverting the matrix,
    /// especially for matrices that are nearly singular (close to having no inverse).
    /// </remarks>
    /// <returns>The inverse of the original matrix.</returns>
    public override Matrix<T> Invert()
    {
        Matrix<T> D = Matrix<T>.CreateDiagonal(EigenValues);
        return EigenVectors.Multiply(D.InvertDiagonalMatrix()).Multiply(EigenVectors.Transpose());
    }
}
