using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Matrix Computations", "https://doi.org/10.56021/9781421407944")]
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
    /// Computes eigenvalues and eigenvectors in order of decreasing magnitude (the
    /// <see cref="EigenAlgorithmType.PowerIteration"/> option).
    /// </summary>
    /// <remarks>
    /// This option is an alias of <see cref="EigenAlgorithmType.QR"/>: both make one engine eigensolver
    /// call (<c>Linalg.Eigh</c> for symmetric input, otherwise <c>Linalg.Eig</c> keeping real parts) and
    /// sort the pairs dominant-first, the order power iteration with deflation delivers them. No power
    /// iteration or deflation runs, so choosing this option does not change accuracy or convergence.
    /// </remarks>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the eigenvalues and eigenvectors.</returns>
    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEigenPowerIteration(Matrix<T> matrix)
    {
        // Power iteration with deflation finds the eigenpairs in order of decreasing magnitude.
        // The engine eigensolver returns all pairs at once; reorder them dominant-first.
        return ComputeEngineEigen(matrix, dominantFirst: true);
    }

    /// <summary>
    /// Computes eigenvalues and eigenvectors in order of decreasing magnitude (the
    /// <see cref="EigenAlgorithmType.QR"/> option).
    /// </summary>
    /// <remarks>
    /// One engine eigensolver call (<c>Linalg.Eigh</c> for symmetric input, otherwise <c>Linalg.Eig</c>
    /// keeping real parts), with the pairs sorted dominant-first as a deflating QR iteration leaves
    /// them. No explicit shifted QR iteration runs; <see cref="EigenAlgorithmType.PowerIteration"/> is an
    /// alias of this option.
    /// </remarks>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the eigenvalues and eigenvectors.</returns>
    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEigenQR(Matrix<T> matrix)
    {
        // The shifted QR iteration deflates the dominant eigenvalues into the leading diagonal
        // positions, and callers (whitening, SSA, Takagi) take the leading pairs as the largest.
        return ComputeEngineEigen(matrix, dominantFirst: true);
    }

    /// <summary>
    /// Computes all eigenpairs with one engine eigensolver call: the symmetric solver
    /// (<c>Linalg.Eigh</c>, orthonormal eigenvectors) for symmetric input, otherwise the general
    /// solver (<c>Linalg.Eig</c>) keeping the real parts, as the real-arithmetic QR iteration did.
    /// </summary>
    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEngineEigen(Matrix<T> matrix, bool dominantFirst)
    {
        int n = matrix.Rows;
        if (n == 0)
        {
            return (new Vector<T>(0), new Matrix<T>(0, 0));
        }

        var a = Tensor<T>.FromMatrix(matrix);
        Tensor<T> values;
        Tensor<T> vectors;
        if (NumOps.LessThanOrEquals(MaxAsymmetry(a), SymmetryTolerance(a)))
        {
            (values, vectors) = LinalgDispatch<T>.Eigh(a);
        }
        else
        {
            var (complexValues, complexVectors) = LinalgDispatch<T>.Eig(a);
            values = Engine.TensorSelect(complexValues, 1, 0);
            vectors = Engine.TensorSelect(complexVectors, 2, 0);
        }

        if (dominantFirst)
        {
            var (_, order) = Engine.TensorSort(Engine.TensorAbs(values), 0, descending: true);
            values = Engine.TensorIndexSelect(values, order, 0);
            vectors = Engine.TensorIndexSelect(vectors, order, 1);
        }

        return (values.ToVector(), vectors.ToMatrix());
    }

    /// <summary>
    /// Computes eigenvalues and eigenvectors of a symmetric matrix (the <see cref="EigenAlgorithmType.Jacobi"/> option).
    /// </summary>
    /// <remarks>
    /// The matrix must be symmetric; other input throws <see cref="ArgumentException"/>. The result is
    /// the symmetric eigendecomposition A = V diag(λ) V^T with orthonormal V, which is what converged
    /// Jacobi rotations produce. It is computed with one call to the engine symmetric eigensolver
    /// (<c>Linalg.Eigh</c>), not with explicit Jacobi rotations, and the eigenvalues keep the solver's
    /// order. All three <see cref="EigenAlgorithmType"/> options use the engine eigensolver.
    /// </remarks>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the eigenvalues and eigenvectors of the matrix.</returns>
    private (Vector<T> eigenValues, Matrix<T> eigenVectors) ComputeEigenJacobi(Matrix<T> matrix)
    {
        // Jacobi rotations solve the symmetric eigenproblem; a non-symmetric matrix would silently
        // solve a different problem, so it is rejected at the API boundary. Tolerance is scaled by
        // the max-abs entry (floored at 1) and is type-aware (float 1e-4, double 1e-10).
        int n = matrix.Rows;
        if (n == 0)
        {
            return (new Vector<T>(0), new Matrix<T>(0, 0));
        }

        var a = Tensor<T>.FromMatrix(matrix);
        ValidateSymmetric(a, "Jacobi eigen decomposition", nameof(matrix));

        if (n == 1)
        {
            return (MatrixHelper<T>.ExtractDiagonal(matrix), Matrix<T>.CreateIdentity(1));
        }

        // The converged Jacobi iteration is the symmetric eigendecomposition A = V diag(λ) V^T
        // with orthonormal V; the engine symmetric eigensolver computes it in one call.
        var (values, vectors) = LinalgDispatch<T>.Eigh(a);
        return (values.ToVector(), vectors.ToMatrix());
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
