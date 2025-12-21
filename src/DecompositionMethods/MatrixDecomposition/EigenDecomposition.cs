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

        for (int i = 0; i < n; i++)
        {
            // Replace CreateRandom with a method to create a random vector
            Vector<T> v = Vector<T>.CreateRandom(n);
            for (int iter = 0; iter < 100; iter++)
            {
                Vector<T> w = matrix.Multiply(v);
                T eigenValue = NumOps.Divide(w.DotProduct(v), v.DotProduct(v));
                v = w.Divide(w.Norm());

                if (iter > 0 && NumOps.LessThan(NumOps.Abs(NumOps.Subtract(eigenValue, eigenValues[i])), NumOps.FromDouble(1e-6)))
                {
                    break;
                }
                eigenValues[i] = eigenValue;
            }

            eigenVectors.SetColumn(i, v);
            // Fix the Multiply operation
            matrix = matrix.Subtract(MatrixHelper<T>.OuterProduct(v, v).Multiply(eigenValues[i]));
        }

        return (eigenValues, eigenVectors);
    }

    /// <summary>
    /// Computes eigenvalues and eigenvectors using the QR algorithm.
    /// </summary>
    /// <remarks>
    /// The QR algorithm is an iterative method that works by repeatedly factoring the matrix
    /// into a product Q*R (where Q is orthogonal and R is upper triangular), and then
    /// recombining as R*Q. This process eventually converges to a matrix where the eigenvalues
    /// appear on the diagonal.
    /// 
    /// This is one of the most widely used methods for computing all eigenvalues and eigenvectors
    /// of a matrix, as it is generally stable and accurate.
    /// </remarks>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the eigenvalues and eigenvectors.</returns>
    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEigenQR(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Matrix<T> A = matrix.Clone();
        Matrix<T> Q = Matrix<T>.CreateIdentity(n);

        for (int iter = 0; iter < 100; iter++)
        {
            var qrDecomp = new QrDecomposition<T>(A);
            (var q, var r) = (qrDecomp.Q, qrDecomp.R);
            A = r.Multiply(q);
            Q = Q.Multiply(q);

            if (A.IsUpperTriangularMatrix(NumOps.FromDouble(1e-6)))
                break;
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
