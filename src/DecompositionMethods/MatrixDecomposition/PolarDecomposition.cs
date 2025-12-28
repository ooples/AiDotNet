namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements the Polar Decomposition of a matrix, which factors a matrix A into the product of
/// an orthogonal matrix U and a positive semi-definite matrix P.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrices and vectors.</typeparam>
/// <remarks>
/// <para>
/// The Polar Decomposition expresses a matrix A as A = UP, where U is orthogonal (U^T * U = I)
/// and P is positive semi-definite.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of Polar Decomposition as breaking down a transformation into two simpler steps:
/// first a rotation/reflection (U), and then a stretching/scaling (P). This is similar to how polar
/// coordinates break down a point into an angle and a distance.
/// </para>
/// </remarks>
public class PolarDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the orthogonal factor of the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This matrix represents the rotation/reflection part of the transformation.
    /// It preserves angles and distances when applied to vectors.
    /// </remarks>
    public Matrix<T> U { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the positive semi-definite factor of the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This matrix represents the stretching/scaling part of the transformation.
    /// It may change the length of vectors but in a symmetric way.
    /// </remarks>
    public Matrix<T> P { get; private set; } = new Matrix<T>(0, 0);

    private readonly PolarAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the PolarDecomposition class.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for the decomposition.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This creates a new polar decomposition of your matrix using the specified algorithm.
    /// If you're not sure which algorithm to use, the default (SVD) is generally reliable but may be slower.
    /// </remarks>
    public PolarDecomposition(Matrix<T> matrix, PolarAlgorithmType algorithm = PolarAlgorithmType.SVD)
        : base(matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Polar decomposition requires a square matrix.", nameof(matrix));
        }
        _algorithm = algorithm;
        Decompose();
    }

    /// <summary>
    /// Performs the polar decomposition.
    /// </summary>
    protected override void Decompose()
    {
        ComputeDecomposition(_algorithm);
    }

    /// <summary>
    /// Computes the polar decomposition using the specified algorithm.
    /// </summary>
    /// <param name="algorithm">The algorithm to use for the decomposition.</param>
    /// <remarks>
    /// <para>
    /// Different algorithms have different trade-offs in terms of speed, accuracy, and stability.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the U and P matrices using one of several methods.
    /// Each method has its strengths and weaknesses:
    /// - SVD: Most accurate but can be slow
    /// - Newton-Schulz: Fast for matrices close to orthogonal
    /// - Halley: Faster convergence than Newton-Schulz but more complex
    /// - QR Iteration: Good balance of speed and accuracy
    /// - Scaling and Squaring: Good for matrices with large condition numbers
    /// </para>
    /// </remarks>
    private void ComputeDecomposition(PolarAlgorithmType algorithm)
    {
        switch (algorithm)
        {
            case PolarAlgorithmType.SVD:
                DecomposeSVD();
                break;
            case PolarAlgorithmType.NewtonSchulz:
                DecomposeNewtonSchulz();
                break;
            case PolarAlgorithmType.HalleyIteration:
                DecomposeHalleyIteration();
                break;
            case PolarAlgorithmType.QRIteration:
                DecomposeQRIteration();
                break;
            case PolarAlgorithmType.ScalingAndSquaring:
                DecomposeScalingAndSquaring();
                break;
            default:
                throw new ArgumentException("Unsupported Polar decomposition algorithm.");
        }
    }

    /// <summary>
    /// Performs polar decomposition using Singular Value Decomposition (SVD).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method computes U = U_svd * V_svd^T and P = V_svd * S * V_svd^T.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the most reliable method for polar decomposition. It works by first
    /// breaking down the matrix using another technique (SVD) and then recombining the pieces to
    /// get our U and P matrices. It's like taking apart a watch, organizing the pieces, and then
    /// reassembling them in a different way.
    /// </para>
    /// </remarks>
    private void DecomposeSVD()
    {
        var svd = new SvdDecomposition<T>(A);

        // VECTORIZED: Uses matrix multiplication and transpose operations which are already vectorized
        // Polar decomposition from SVD: A = U_svd * Σ * V^T
        // U_polar = U_svd * V^T, P = V * Σ * V^T
        // Since svd.Vt = V^T, we have: U_polar = svd.U * svd.Vt
        U = svd.U.Multiply(svd.Vt);
        var sigma = Matrix<T>.CreateDiagonal(svd.S);
        P = svd.Vt.Transpose().Multiply(sigma).Multiply(svd.Vt);
    }

    /// <summary>
    /// Performs polar decomposition using the Newton-Schulz iterative algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method uses an iterative approach that converges quadratically when the starting matrix
    /// is close to orthogonal.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method finds U and P through a series of approximations that get better
    /// with each step. It's like homing in on a target by making adjustments based on how far off
    /// you are. This method works well when your matrix is already close to being a pure rotation.
    /// </para>
    /// </remarks>
    private void DecomposeNewtonSchulz()
    {
        int n = A.Rows;

        // Scale the matrix to have spectral norm close to 1 for convergence
        T norm = MatrixHelper<T>.SpectralNorm(A);
        Matrix<T> X = A.Multiply(NumOps.Divide(NumOps.One, norm));

        T tolerance = NumOps.FromDouble(1e-10);
        int maxIterations = 100;

        for (int i = 0; i < maxIterations; i++)
        {
            // Standard Newton-Schulz iteration: X_{k+1} = (1/2) * X_k * (3*I - X_k^T * X_k)
            Matrix<T> XtX = X.Transpose().Multiply(X);
            Matrix<T> I = Matrix<T>.CreateIdentityMatrix(n);

            // Compute 3*I - X^T * X
            Matrix<T> term = I.Multiply(NumOps.FromDouble(3.0)).Subtract(XtX);

            // X_{k+1} = (1/2) * X * term
            Matrix<T> nextX = X.Multiply(term).Multiply(NumOps.FromDouble(0.5));

            T error = nextX.Subtract(X).FrobeniusNorm();

            if (NumOps.LessThan(error, tolerance))
            {
                X = nextX;
                break;
            }

            X = nextX;

            // Check for divergence
            if (NumOps.GreaterThan(error, NumOps.FromDouble(1e6)))
            {
                throw new InvalidOperationException("Newton-Schulz iteration diverged.");
            }
        }

        U = X;
        P = U.Transpose().Multiply(A);
    }

    /// <summary>
    /// Performs polar decomposition using Halley's iterative method.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Halley's method offers cubic convergence, which is faster than Newton-Schulz but requires
    /// matrix inversions at each step.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is another iterative method like Newton-Schulz, but it converges faster
    /// (gets to the right answer in fewer steps). The trade-off is that each step is more complex
    /// and computationally expensive. It's like taking bigger, more carefully calculated steps
    /// toward your destination.
    /// </para>
    /// </remarks>
    private void DecomposeHalleyIteration()
    {
        int n = A.Rows;

        // Scale the matrix to have spectral norm close to 1 for convergence
        T norm = MatrixHelper<T>.SpectralNorm(A);

        // Handle case where matrix is zero or nearly zero
        if (NumOps.LessThan(norm, NumOps.FromDouble(1e-14)))
        {
            U = Matrix<T>.CreateIdentity(n);
            P = new Matrix<T>(n, n);
            return;
        }

        Matrix<T> X = A.Multiply(NumOps.Divide(NumOps.One, norm));

        T tolerance = NumOps.FromDouble(1e-10);
        int maxIterations = 100;
        Matrix<T> I = Matrix<T>.CreateIdentityMatrix(n);

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Correct Halley iteration: X_{k+1} = X_k * (X_k^T*X_k + 3*I) * (3*X_k^T*X_k + I)^{-1}
            // Note: Order matters! It's X * numerator * (denominator)^{-1}
            Matrix<T> XtX = X.Transpose().Multiply(X);

            // numerator = X^T*X + 3*I
            Matrix<T> numerator = XtX.Add(I.Multiply(NumOps.FromDouble(3.0)));

            // denominator = 3*X^T*X + I
            Matrix<T> denominator = XtX.Multiply(NumOps.FromDouble(3.0)).Add(I);

            // Check invertibility with regularization if needed
            if (!MatrixHelper<T>.IsInvertible(denominator))
            {
                // Add small regularization for numerical stability
                denominator = denominator.Add(I.Multiply(NumOps.FromDouble(1e-10)));
            }

            // X_{k+1} = X * numerator * denominator^{-1}
            Matrix<T> nextX = X.Multiply(numerator).Multiply(denominator.Inverse());

            T error = nextX.Subtract(X).FrobeniusNorm();

            if (NumOps.LessThan(error, tolerance))
            {
                X = nextX;
                break;
            }

            X = nextX;

            // Check for divergence - increase threshold
            T xNorm = X.FrobeniusNorm();
            if (NumOps.GreaterThan(xNorm, NumOps.FromDouble(1e10)))
            {
                // Fall back to SVD method if Halley diverges
                DecomposeSVD();
                return;
            }
        }

        U = X;
        P = U.Transpose().Multiply(A);
    }

    /// <summary>
    /// Performs polar decomposition using QR iteration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method uses successive QR decompositions to iteratively compute the polar factors.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method finds U and P by repeatedly breaking down the matrix using a technique
    /// called QR decomposition, then recombining the pieces in a special way. With each iteration, we get
    /// closer to the correct answer. It's like refining a rough sketch into a detailed drawing through
    /// multiple passes.
    /// </para>
    /// </remarks>
    private void DecomposeQRIteration()
    {
        // QR-based polar iteration: X_{k+1} = Q_k where X_k = Q_k * R_k
        // This converges to U such that A = U * P
        Matrix<T> X = A.Clone();
        T tolerance = NumOps.FromDouble(1e-10);
        int maxIterations = 100;

        for (int i = 0; i < maxIterations; i++)
        {
            var qr = new QrDecomposition<T>(X);
            Matrix<T> Q = qr.Q;

            T error = Q.Subtract(X).FrobeniusNorm();

            if (NumOps.LessThan(error, tolerance))
            {
                X = Q;
                break;
            }

            X = Q;

            // Check for divergence (Q should always be bounded)
            T xNorm = X.FrobeniusNorm();
            if (NumOps.GreaterThan(xNorm, NumOps.FromDouble(1e6)))
            {
                throw new InvalidOperationException("QR iteration diverged.");
            }
        }

        U = X;
        P = U.Transpose().Multiply(A);
    }

    /// <summary>
    /// Performs polar decomposition using the scaling and squaring method.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method scales the matrix to improve numerical stability, applies an iterative process,
    /// and then reverses the scaling through squaring.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method handles matrices that might be difficult to decompose directly.
    /// First, we "scale down" the matrix to make it easier to work with (like zooming out on a map).
    /// Then we perform our calculations on this simpler version. Finally, we "scale back up" to get
    /// our answer for the original matrix (zooming back in). This approach is especially good for
    /// matrices with very large or very small values.
    /// </para>
    /// </remarks>
    private void DecomposeScalingAndSquaring()
    {
        // Standard Newton iteration for polar decomposition: X_{k+1} = (1/2)(X_k + X_k^{-T})
        // This converges to the orthogonal factor U when starting from X_0 = A (scaled)
        // Reference: Higham, "Computing the polar decomposition—with applications"

        Matrix<T> X = A.Clone();

        // Scale the matrix to improve convergence
        // Use Frobenius norm scaling: X = A / ||A||_F
        T frobNorm = X.FrobeniusNorm();
        if (NumOps.GreaterThan(frobNorm, NumOps.FromDouble(1e-14)))
        {
            X = X.Multiply(NumOps.Divide(NumOps.One, frobNorm));
        }

        T tolerance = NumOps.FromDouble(1e-12);
        int maxIterations = 50;

        for (int i = 0; i < maxIterations; i++)
        {
            // Newton iteration: X_{k+1} = (1/2)(X_k + X_k^{-T})
            if (!MatrixHelper<T>.IsInvertible(X))
            {
                // Fall back to SVD-based method for singular matrices
                DecomposeSVD();
                return;
            }

            Matrix<T> Xinv = X.Inverse();
            Matrix<T> XinvT = Xinv.Transpose();
            Matrix<T> Xnew = X.Add(XinvT).Multiply(NumOps.FromDouble(0.5));

            // Check convergence
            T error = X.Subtract(Xnew).FrobeniusNorm();

            X = Xnew;

            if (NumOps.LessThan(error, tolerance))
            {
                break;
            }

            // Check for divergence
            if (NumOps.GreaterThan(error, NumOps.FromDouble(1e6)))
            {
                // Fall back to SVD-based method
                DecomposeSVD();
                return;
            }
        }

        // X has converged to an orthogonal matrix (the unitary factor)
        U = X;

        // P = U^T * A (the positive semi-definite factor)
        P = U.Transpose().Multiply(A);

        // Ensure P is symmetric by averaging: P = (P + P^T) / 2
        P = P.Add(P.Transpose()).Multiply(NumOps.FromDouble(0.5));
    }

    /// <summary>
    /// Solves the linear system Ax = b using the polar decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector.</param>
    /// <returns>The solution vector x such that Ax = b.</returns>
    /// <remarks>
    /// <para>
    /// This method solves the system by first solving Px = b, then computing y = U^T * x.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When you have an equation Ax = b and want to find x, this method uses the
    /// polar decomposition to break this into simpler steps. Since A = UP, we can rewrite the equation
    /// as UPx = b. We first solve Px = b' for x, then compute the final answer using U. This is often
    /// more stable than solving the original equation directly.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        // Solve Px = b
        var x = MatrixSolutionHelper.SolveLinearSystem(P, b, MatrixDecompositionType.Polar);

        // Compute y = U^T * x (equivalent to solving Uy = x)
        return U.Transpose().Multiply(x);
    }

    /// <summary>
    /// Computes the inverse of the original matrix A using its polar decomposition.
    /// </summary>
    /// <returns>The inverse matrix A⁻¹.</returns>
    /// <remarks>
    /// <para>
    /// Since A = UP, the inverse is A⁻¹ = P⁻¹ * U^T.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The inverse of a matrix is like the reciprocal of a number - when you multiply
    /// a matrix by its inverse, you get the identity matrix (similar to how 5 * 1/5 = 1). This method
    /// finds the inverse by using the special properties of the polar decomposition, which makes the
    /// calculation more reliable than directly inverting the original matrix.
    /// </para>
    /// </remarks>
    public override Matrix<T> Invert()
    {
        var invP = P.Inverse();
        var invU = U.Transpose();

        return invP.Multiply(invU);
    }

    /// <summary>
    /// Returns the factors U and P of the polar decomposition.
    /// </summary>
    /// <returns>A tuple containing the orthogonal factor U and the positive semi-definite factor P.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method simply gives you back the two matrices that make up the polar
    /// decomposition. You can use these matrices separately for various calculations or to better
    /// understand the properties of your original matrix.
    /// </remarks>
    public (Matrix<T> U, Matrix<T> P) GetFactors()
    {
        return (U, P);
    }
}
