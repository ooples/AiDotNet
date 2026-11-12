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
        ValidateSymmetric(a, "Takagi Jacobi decomposition", nameof(matrix));

        var (values, vectors) = LinalgDispatch<T>.Eigh(a);
        return TakagiFromEigen(values, vectors);
    }

    /// <summary>
    /// Builds the Takagi factors of a real symmetric A = V diag(λ) V^T: Σ = |λ| and
    /// u_j = v_j for λ_j &gt;= 0, u_j = i·v_j for λ_j &lt; 0, so that U Σ U^T = A.
    /// </summary>
    private (Matrix<T> S, Matrix<Complex<T>> U) TakagiFromEigen(Tensor<T> values, Tensor<T> vectors)
    {
        int n = values.Length;

        // Columns with a negative eigenvalue become purely imaginary.
        var nonNegative = Engine.TensorSubtract(Engine.TensorOnesLike(values), Engine.TensorLessThan(values, NumOps.Zero));
        var realMask = Engine.TensorBroadcastTo(nonNegative.Reshape(new[] { 1, n }), new[] { n, n });
        var realPart = Engine.TensorMultiply(vectors, realMask);
        var imaginaryPart = Engine.TensorSubtract(vectors, realPart);

        var sigma = Matrix<T>.CreateDiagonal(Engine.TensorAbs(values).ToVector());
        return (sigma, ComplexFromParts(realPart.ToMatrix(), imaginaryPart.ToMatrix()));
    }

    /// <summary>
    /// Validates that <paramref name="matrix"/> is symmetric, runs the selected eigensolver and builds
    /// the Takagi factors from its eigenpairs (dominant first for the QR and power options).
    /// </summary>
    private (Matrix<T> S, Matrix<Complex<T>> U) TakagiFromEigenSolver(Matrix<T> matrix, EigenAlgorithmType eigenAlgorithm, string algorithmName)
    {
        if (matrix.Rows == 0)
        {
            return (new Matrix<T>(0, 0), new Matrix<Complex<T>>(0, 0));
        }

        ValidateSymmetric(Tensor<T>.FromMatrix(matrix), algorithmName, nameof(matrix));
        var eigen = new EigenDecomposition<T>(matrix, eigenAlgorithm);
        return TakagiFromEigen(Tensor<T>.FromVector(eigen.EigenValues), Tensor<T>.FromMatrix(eigen.EigenVectors));
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
        // The QR iteration delivers the eigenpairs of the symmetric A dominant first; the Takagi
        // factors are Σ = |λ| and U = V with i·v_j for negative λ_j.
        return TakagiFromEigenSolver(matrix, EigenAlgorithmType.QR, "Takagi QR decomposition");
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
        // Σ = |λ|; eigenvectors of negative eigenvalues are multiplied by i so that U Σ U^T = A.
        return TakagiFromEigenSolver(matrix, EigenAlgorithmType.QR, "Takagi eigendecomposition");
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
        // Power iteration with deflation yields the eigenpairs dominant first; Σ = |λ|.
        return TakagiFromEigenSolver(matrix, EigenAlgorithmType.PowerIteration, "Takagi power iteration");
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
        // dominant first; Σ = |λ|.
        return TakagiFromEigenSolver(matrix, EigenAlgorithmType.QR, "Takagi Lanczos iteration");
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
