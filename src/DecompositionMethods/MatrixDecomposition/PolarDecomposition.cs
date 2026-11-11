using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Matrix Computations", "https://doi.org/10.56021/9781421407944")]
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
        var x = Tensor<T>.FromMatrix(A.Multiply(NumOps.Divide(NumOps.One, norm)));

        T tolerance = NumOps.FromDouble(1e-10);
        T divergence = NumOps.FromDouble(1e6);
        T half = NumOps.FromDouble(0.5);
        var threeI = Engine.TensorMultiplyScalar(Engine.TensorEye<T>(n), NumOps.FromDouble(3.0));
        int maxIterations = 100;

        for (int i = 0; i < maxIterations; i++)
        {
            // Standard Newton-Schulz iteration: X_{k+1} = (1/2) * X_k * (3*I - X_k^T * X_k)
            var xtx = Engine.TensorMatMul(Engine.TensorTranspose(x), x);
            var nextX = Engine.TensorMultiplyScalar(Engine.TensorMatMul(x, Engine.TensorSubtract(threeI, xtx)), half);

            var error = Engine.TensorNorm(Engine.TensorSubtract(nextX, x));
            x = nextX;

            if (AnyBelow(error, tolerance))
            {
                break;
            }

            // Check for divergence
            if (AnyAbove(error, divergence))
            {
                throw new InvalidOperationException("Newton-Schulz iteration diverged.");
            }
        }

        U = x.ToMatrix();
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

        var x = Tensor<T>.FromMatrix(A.Multiply(NumOps.Divide(NumOps.One, norm)));

        T tolerance = NumOps.FromDouble(1e-10);
        T divergence = NumOps.FromDouble(1e10);
        T three = NumOps.FromDouble(3.0);
        var identity = Engine.TensorEye<T>(n);
        var threeI = Engine.TensorMultiplyScalar(identity, three);
        var regularization = Engine.TensorMultiplyScalar(identity, NumOps.FromDouble(1e-10));
        int maxIterations = 100;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Halley iteration: X_{k+1} = X_k * (X_k^T*X_k + 3*I) * (3*X_k^T*X_k + I)^{-1}
            var xtx = Engine.TensorMatMul(Engine.TensorTranspose(x), x);
            var numerator = Engine.TensorAdd(xtx, threeI);
            var denominator = Engine.TensorAdd(Engine.TensorMultiplyScalar(xtx, three), identity);

            // Regularize a singular denominator for numerical stability
            if (!MatrixHelper<T>.IsInvertible(denominator.ToMatrix()))
            {
                denominator = Engine.TensorAdd(denominator, regularization);
            }

            var nextX = Engine.TensorMatMul(Engine.TensorMatMul(x, numerator), Linalg.Inv(denominator));
            var error = Engine.TensorNorm(Engine.TensorSubtract(nextX, x));
            x = nextX;

            if (AnyBelow(error, tolerance))
            {
                break;
            }

            // Fall back to SVD method if Halley diverges
            if (AnyAbove(Engine.TensorNorm(x), divergence))
            {
                DecomposeSVD();
                return;
            }
        }

        U = x.ToMatrix();
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
        T divergence = NumOps.FromDouble(1e6);
        int maxIterations = 100;

        for (int i = 0; i < maxIterations; i++)
        {
            var qr = new QrDecomposition<T>(X);
            Matrix<T> Q = qr.Q;

            var error = Engine.TensorNorm(Engine.TensorSubtract(Tensor<T>.FromMatrix(Q), Tensor<T>.FromMatrix(X)));
            X = Q;

            if (AnyBelow(error, tolerance))
            {
                break;
            }

            // Check for divergence (Q should always be bounded)
            if (AnyAbove(Engine.TensorNorm(Tensor<T>.FromMatrix(X)), divergence))
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
        var x = Tensor<T>.FromMatrix(A.Clone());

        // Scale the matrix to improve convergence: X = A / ||A||_F
        T frobNorm = FrobeniusNorm(A);
        if (NumOps.GreaterThan(frobNorm, NumOps.FromDouble(1e-14)))
        {
            x = Engine.TensorMultiplyScalar(x, NumOps.Divide(NumOps.One, frobNorm));
        }

        T tolerance = NumOps.FromDouble(1e-12);
        T divergence = NumOps.FromDouble(1e6);
        T half = NumOps.FromDouble(0.5);
        int maxIterations = 50;

        for (int i = 0; i < maxIterations; i++)
        {
            // Newton iteration: X_{k+1} = (1/2)(X_k + X_k^{-T})
            if (!MatrixHelper<T>.IsInvertible(x.ToMatrix()))
            {
                // Fall back to SVD-based method for singular matrices
                DecomposeSVD();
                return;
            }

            var inverseTransposed = Engine.TensorTranspose(Linalg.Inv(x));
            var xNew = Engine.TensorMultiplyScalar(Engine.TensorAdd(x, inverseTransposed), half);

            // Check convergence
            var error = Engine.TensorNorm(Engine.TensorSubtract(x, xNew));
            x = xNew;

            if (AnyBelow(error, tolerance))
            {
                break;
            }

            // Check for divergence: fall back to the SVD-based method
            if (AnyAbove(error, divergence))
            {
                DecomposeSVD();
                return;
            }
        }

        // X has converged to an orthogonal matrix (the unitary factor)
        U = x.ToMatrix();

        // P = U^T * A (the positive semi-definite factor), symmetrized: P = (P + P^T) / 2
        P = U.Transpose().Multiply(A);
        P = P.Add(P.Transpose()).Multiply(half);
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
