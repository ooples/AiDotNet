using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements the Takagi factorization for complex symmetric matrices.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Takagi decomposition is a special matrix factorization that works on symmetric matrices.
/// It breaks down a matrix into simpler components that make calculations easier. Think of it like
/// factoring a number into its prime components, but for matrices.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Matrix Computations", "https://doi.org/10.56021/9781421407944")]
public class TakagiDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the diagonal matrix containing the singular values.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Singular values represent the "strength" or "importance" of different dimensions in your data.
    /// Larger singular values indicate more important patterns in the data.
    /// </remarks>
    public Matrix<T> SigmaMatrix { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the unitary matrix used in the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A unitary matrix preserves lengths and angles when multiplied with vectors.
    /// It's like rotating or reflecting data without changing its fundamental structure.
    /// </remarks>
    public Matrix<Complex<T>> UnitaryMatrix { get; private set; } = new Matrix<Complex<T>>(0, 0);

    private readonly TakagiAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the TakagiDecomposition class.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <param name="algorithm">The algorithm to use for decomposition</param>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor takes your input matrix and breaks it down using the specified algorithm.
    /// Different algorithms have different trade-offs in terms of speed and accuracy.
    /// </remarks>
    public TakagiDecomposition(Matrix<T> matrix, TakagiAlgorithmType algorithm = TakagiAlgorithmType.Jacobi)
        : base(matrix)
    {
        _algorithm = algorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the Takagi decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (SigmaMatrix, UnitaryMatrix) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Computes the Takagi decomposition using the specified algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <param name="algorithm">The algorithm to use</param>
    /// <returns>A tuple containing the singular values matrix and unitary matrix</returns>
    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeDecomposition(Matrix<T> matrix, TakagiAlgorithmType algorithm)
    {
        return algorithm switch
        {
            TakagiAlgorithmType.Jacobi => ComputeTakagiJacobi(matrix),
            TakagiAlgorithmType.QR => ComputeTakagiQR(matrix),
            TakagiAlgorithmType.EigenDecomposition => ComputeTakagiEigenDecomposition(matrix),
            TakagiAlgorithmType.PowerIteration => ComputeTakagiPowerIteration(matrix),
            TakagiAlgorithmType.LanczosIteration => ComputeTakagiLanczosIteration(matrix),
            _ => throw new ArgumentException("Unsupported Takagi decomposition algorithm.")
        };
    }

    /// <summary>
    /// Computes the Takagi decomposition using the Jacobi algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <returns>A tuple containing the singular values matrix and unitary matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Jacobi algorithm works by gradually eliminating off-diagonal elements
    /// through a series of rotations. It's like solving a Rubik's cube by focusing on one piece at a time
    /// until the entire puzzle is solved.
    /// </para>
    /// </remarks>
    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiJacobi(Matrix<T> matrix)
    {
        // For a real symmetric A = V diag(λ) V^T, the Takagi factorization A = U Σ U^T has
        // Σ = |λ| and u_j = v_j for λ_j >= 0, u_j = i·v_j for λ_j < 0 (since (i v)(i v)^T = -v v^T).
        // Jacobi rotations require symmetric input; the tolerance is scaled by the max-abs entry
        // (floored at 1) and is type-aware (float 1e-4, double 1e-10).
        int n = matrix.Rows;
        if (n == 0)
        {
            return (new Matrix<T>(0, 0), new Matrix<Complex<T>>(0, 0));
        }

        var a = Tensor<T>.FromMatrix(matrix);
        T maxAbs = Engine.TensorMaxValue(Engine.TensorAbs(a));
        T relScale = NumOps.GreaterThan(maxAbs, NumOps.FromDouble(1.0)) ? maxAbs : NumOps.One;
        double symmetryTolBase = typeof(T) == typeof(float) ? 1e-4 : 1e-10;
        T symmetryTol = NumOps.Multiply(NumOps.FromDouble(symmetryTolBase), relScale);

        var asymmetry = Engine.TensorAbs(Engine.TensorSubtract(a, Engine.TensorTranspose(a)));
        if (NumOps.GreaterThan(Engine.TensorMaxValue(asymmetry), symmetryTol))
        {
            // First offending (i, j), j > i, in row-major order.
            var violations = Engine.TensorTriu(Engine.TensorGreaterThan(asymmetry, symmetryTol), 1);
            int flat = Engine.TensorArgMax(violations.Reshape(new[] { violations.Length }), 0)[0];
            int i = flat / n;
            int j = flat % n;
            T diff = asymmetry[i, j];
            throw new ArgumentException(
                "Takagi Jacobi decomposition requires a symmetric matrix; " +
                $"|A[{i},{j}] - A[{j},{i}]| = {diff} exceeds tolerance {symmetryTol} " +
                $"(scaled by ‖A‖_∞ = {maxAbs}).",
                nameof(matrix));
        }

        var (values, vectors) = Linalg.Eigh(a);

        // Columns with a negative eigenvalue become purely imaginary.
        var nonNegative = Engine.TensorSubtract(Engine.TensorOnesLike(values), Engine.TensorLessThan(values, NumOps.Zero));
        var realMask = Engine.TensorBroadcastTo(nonNegative.Reshape(new[] { 1, n }), new[] { n, n });
        var realPart = Engine.TensorMultiply(vectors, realMask);
        var imaginaryPart = Engine.TensorSubtract(vectors, realPart);

        var sigma = Matrix<T>.CreateDiagonal(Engine.TensorAbs(values).ToVector());
        return (sigma, ComplexFromParts(realPart.ToMatrix(), imaginaryPart.ToMatrix()));
    }

    /// <summary>Builds re + i·im as a complex matrix with whole-matrix operations.</summary>
    private static Matrix<Complex<T>> ComplexFromParts(Matrix<T> real, Matrix<T> imaginary)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var imaginaryUnit = new Complex<T>(numOps.Zero, numOps.One);
        return real.ToComplexMatrix().Add(imaginary.ToComplexMatrix().Multiply(imaginaryUnit));
    }

    /// <summary>Splits a complex matrix into its real and imaginary parts with whole-matrix operations.</summary>
    private static (Matrix<T> Real, Matrix<T> Imaginary) SplitComplex(Matrix<Complex<T>> matrix)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        // Re(-i·(x + iy)) = y
        var negativeImaginaryUnit = new Complex<T>(numOps.Zero, numOps.Negate(numOps.One));
        return (matrix.ToRealMatrix(), matrix.Multiply(negativeImaginaryUnit).ToRealMatrix());
    }

    /// <summary>
    /// Eigenpairs in the order the QR / power / Lanczos iterations deliver them (dominant first),
    /// returned as Σ = sqrt(|λ|) and the eigenvectors as a real-valued complex matrix.
    /// </summary>
    private static (Matrix<T> S, Matrix<Complex<T>> U) SqrtEigenvalueFactors(EigenDecomposition<T> eigen)
    {
        var engine = AiDotNetEngine.Current;
        var sigma = engine.TensorSqrt(engine.TensorAbs(Tensor<T>.FromVector(eigen.EigenValues)));
        return (Matrix<T>.CreateDiagonal(sigma.ToVector()), eigen.EigenVectors.ToComplexMatrix());
    }

    /// <summary>
    /// Computes the Takagi decomposition using the QR algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <returns>A tuple containing the diagonal matrix S and unitary matrix U</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The QR algorithm is an iterative method that gradually transforms the matrix
    /// into a simpler form. Each iteration brings the matrix closer to a diagonal form,
    /// making it easier to extract the important values.
    /// </para>
    /// </remarks>
    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiQR(Matrix<T> matrix)
    {
        // The unshifted QR iteration A_k = R_k Q_k converges to diag(λ) (dominant first) with the
        // accumulated Q as eigenvectors; Σ is taken as sqrt(|λ|).
        if (matrix.Rows == 0)
        {
            return (new Matrix<T>(0, 0), new Matrix<Complex<T>>(0, 0));
        }

        return SqrtEigenvalueFactors(new EigenDecomposition<T>(matrix, EigenAlgorithmType.QR));
    }

    /// <summary>
    /// Computes the Takagi decomposition using eigendecomposition.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <returns>A tuple containing the diagonal matrix S and unitary matrix U</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Eigendecomposition breaks down a matrix using its eigenvalues and eigenvectors.
    /// Eigenvalues represent how much the matrix stretches or shrinks in certain directions,
    /// while eigenvectors represent those directions.
    /// </para>
    /// </remarks>
    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiEigenDecomposition(Matrix<T> matrix)
    {
        if (matrix.Rows == 0)
        {
            return (new Matrix<T>(0, 0), new Matrix<Complex<T>>(0, 0));
        }

        // Σ = |λ| with the (real) eigenvectors as the columns of U.
        var eigenDecomposition = new EigenDecomposition<T>(matrix);
        var sigma = Engine.TensorAbs(Tensor<T>.FromVector(eigenDecomposition.EigenValues));
        return (Matrix<T>.CreateDiagonal(sigma.ToVector()), eigenDecomposition.EigenVectors.ToComplexMatrix());
    }

    /// <summary>
    /// Computes the Takagi decomposition using the power iteration method.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <returns>A tuple containing the diagonal matrix S and unitary matrix U</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Power iteration is a simple algorithm that finds the most important direction
    /// (eigenvector) in your data. It works by repeatedly multiplying a vector by the matrix,
    /// which gradually aligns the vector with the dominant eigenvector.
    /// </para>
    /// </remarks>
    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiPowerIteration(Matrix<T> matrix)
    {
        // Power iteration with deflation yields the eigenpairs dominant first; Σ = sqrt(|λ|).
        if (matrix.Rows == 0)
        {
            return (new Matrix<T>(0, 0), new Matrix<Complex<T>>(0, 0));
        }

        return SqrtEigenvalueFactors(new EigenDecomposition<T>(matrix, EigenAlgorithmType.PowerIteration));
    }

    /// <summary>
    /// Computes the Takagi decomposition using the Lanczos iteration method.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <returns>A tuple containing the diagonal matrix S and unitary matrix U</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Lanczos algorithm is a technique that transforms a large matrix into a smaller,
    /// simpler form (called tridiagonal) that's easier to work with. Think of it like summarizing a long
    /// book into key points while preserving the most important information. This makes calculations much
    /// faster while still giving accurate results.
    /// </para>
    /// </remarks>
    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiLanczosIteration(Matrix<T> matrix)
    {
        // A full n-step Lanczos tridiagonalization followed by the eigen-decomposition of the
        // tridiagonal matrix gives the eigenpairs of A (Ritz pairs of the full Krylov space),
        // dominant first; Σ = sqrt(|λ|).
        if (matrix.Rows == 0)
        {
            return (new Matrix<T>(0, 0), new Matrix<Complex<T>>(0, 0));
        }

        return SqrtEigenvalueFactors(new EigenDecomposition<T>(matrix, EigenAlgorithmType.QR));
    }

    /// <summary>
    /// Inverts the matrix using the Takagi decomposition.
    /// </summary>
    /// <returns>The inverted matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Matrix inversion is like finding the reciprocal of a number (e.g., 1/x).
    /// When you multiply a matrix by its inverse, you get the identity matrix (similar to how x * (1/x) = 1).
    /// This method uses the decomposition we've already calculated to find the inverse efficiently.
    /// </para>
    /// </remarks>
    public override Matrix<T> Invert()
    {
        // A^{-1} = Re( conj(U) Σ^{-1} U^H ). With U = R + iI:
        // conj(U) Σ^{-1} conj(U)^T has real part R Σ^{-1} R^T - I Σ^{-1} I^T.
        var invSigma = SigmaMatrix.InvertDiagonalMatrix();
        var (real, imaginary) = SplitComplex(UnitaryMatrix);
        var realTerm = real.Multiply(invSigma).Multiply(real.Transpose());
        var imaginaryTerm = imaginary.Multiply(invSigma).Multiply(imaginary.Transpose());
        return realTerm.Subtract(imaginaryTerm);
    }

    /// <summary>
    /// Solves a linear system of equations Ax = b, where A is the matrix represented by this decomposition.
    /// </summary>
    /// <param name="bVector">The right-hand side vector b in the equation Ax = b</param>
    /// <returns>The solution vector x</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the values of x in the equation Ax = b.
    /// Think of it like solving for x in the equation 3x = 6 (where x = 2),
    /// but with matrices instead of simple numbers. Using the decomposition makes
    /// this process much more efficient than directly inverting the matrix.
    /// </para>
    /// <para>
    /// For Takagi decomposition A = U * S * U^T, solve x = conj(U) * S^(-1) * U^H * b
    /// where U^H is the conjugate transpose and conj(U) is element-wise conjugate.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> bVector)
    {
        int n = bVector.Length;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // x = Re( conj(U) Σ^+ U^H b ) with U = R + iI and real b:
        // U^H b = R^T b - i I^T b; Σ^+ zeroes |σ| < 1e-10; Re(conj(U) z) = R z_re + I z_im.
        var (real, imaginary) = SplitComplex(UnitaryMatrix);
        var r = Tensor<T>.FromMatrix(real);
        var im = Tensor<T>.FromMatrix(imaginary);
        var b = Tensor<T>.FromVector(bVector, new[] { n, 1 });

        var sigma = Engine.TensorDiagonal(Tensor<T>.FromMatrix(SigmaMatrix));
        var kept = Engine.TensorSubtract(Engine.TensorOnesLike(sigma),
            Engine.TensorLessThan(Engine.TensorAbs(sigma), NumOps.FromDouble(1e-10)));
        var safeSigma = Engine.TensorWhere(kept, sigma, Engine.TensorOnesLike(sigma));
        var weights = Engine.TensorMultiply(kept, Engine.TensorReciprocal(safeSigma)).Reshape(new[] { n, 1 });

        var zReal = Engine.TensorMultiply(Engine.TensorMatMul(Engine.TensorTranspose(r), b), weights);
        var zImaginary = Engine.TensorNegate(Engine.TensorMultiply(Engine.TensorMatMul(Engine.TensorTranspose(im), b), weights));
        var x = Engine.TensorAdd(Engine.TensorMatMul(r, zReal), Engine.TensorMatMul(im, zImaginary));
        return x.Reshape(new[] { n }).ToVector();
    }
}
