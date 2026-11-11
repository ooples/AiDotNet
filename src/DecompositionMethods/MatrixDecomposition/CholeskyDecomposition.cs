using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements the Cholesky decomposition for symmetric positive definite matrices.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// The Cholesky decomposition breaks down a symmetric positive definite matrix into
/// the product of a lower triangular matrix and its transpose (A = L * L^T).
/// This is useful for solving linear systems and matrix inversion more efficiently.
/// </para>
/// <para>
/// <b>For Beginners:</b> Cholesky decomposition is like taking the square root of a matrix.
/// Just as 25 = 5 × 5, this method breaks down special matrices into L × L^T, where L is
/// a simpler triangular matrix. This makes complex calculations much faster and more accurate.
/// </para>
/// <para>
/// Real-world applications:
/// - Solving systems of linear equations in scientific computing
/// - Covariance matrix decomposition in statistics and machine learning
/// - Efficient simulation of correlated random variables
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Matrix Computations", "https://doi.org/10.56021/9781421407944")]
public class CholeskyDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the lower triangular matrix L from the decomposition A = L * L^T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The L matrix has zeros above the main diagonal and contains
    /// all the information needed to reconstruct your original matrix. Think of it as
    /// one of the "building blocks" that combine to recreate the original matrix.
    /// </para>
    /// </remarks>
    public Matrix<T> L { get; private set; } = new Matrix<T>(0, 0);

    private readonly CholeskyAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the Cholesky decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The symmetric positive definite matrix to decompose.</param>
    /// <param name="algorithm">The specific Cholesky algorithm to use (default is Crout).</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the matrix is not square, not symmetric, or not positive definite.
    /// </exception>
    /// <remarks>
    /// <para>
    /// A positive definite matrix has all positive eigenvalues, which means it has a unique
    /// Cholesky decomposition. In practical terms, for most matrices you encounter in data
    /// science, this means the matrix must be symmetric with positive values on the diagonal.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This constructor checks that your matrix meets the requirements
    /// (symmetric and positive definite) and then breaks it down using the selected algorithm.
    /// Different algorithms may have different performance characteristics depending on your matrix.
    /// </para>
    /// </remarks>
    public CholeskyDecomposition(Matrix<T> matrix, CholeskyAlgorithmType algorithm = CholeskyAlgorithmType.Crout)
        : base(matrix)
    {
        _algorithm = algorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the Cholesky decomposition.
    /// </summary>
    protected override void Decompose()
    {
        L = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Solves the linear system Ax = b using the Cholesky decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <para>
    /// This method solves the system in two steps:
    /// 1. Forward substitution to solve Ly = b
    /// 2. Back substitution to solve L^Tx = y
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This solves equations of the form Ax = b, where A is your matrix,
    /// b is known, and x is what you're finding. It uses the decomposition to solve this
    /// efficiently in two simple steps, much faster than directly solving the original equation.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        var y = SolveTriangularSystem(L, b, upper: false);
        return SolveTriangularSystem(L.Transpose(), y, upper: true);
    }

    /// <summary>
    /// Selects and applies the appropriate Cholesky decomposition algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm type to use.</param>
    /// <returns>The lower triangular matrix L from the decomposition.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    private Matrix<T> ComputeDecomposition(Matrix<T> matrix, CholeskyAlgorithmType algorithm)
    {
        // Validate matrix is symmetric (required for all Cholesky algorithms)
        ValidateSymmetric(matrix);

        // Crout (column order), Banachiewicz (row order), LDL' (square roots last) and the blocked
        // variant all produce the same unique lower factor of an SPD matrix; the engine's Cholesky
        // kernel computes it in one whole-matrix call on every backend.
        return algorithm switch
        {
            CholeskyAlgorithmType.Crout
                or CholeskyAlgorithmType.Banachiewicz
                or CholeskyAlgorithmType.LDL
                or CholeskyAlgorithmType.BlockCholesky => ComputeCholeskyFactor(matrix),
            _ => throw new ArgumentException("Unsupported Cholesky decomposition algorithm.")
        };
    }

    /// <summary>
    /// Validates that the matrix is symmetric.
    /// </summary>
    /// <param name="matrix">The matrix to validate.</param>
    /// <exception cref="ArgumentException">Thrown when the matrix is not symmetric.</exception>
    private void ValidateSymmetric(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n == 0 || matrix.Columns != n)
        {
            // Non-square input is rejected by the factorization with its own message.
            return;
        }

        // Use relative tolerance: scale by the largest diagonal magnitude so that
        // matrices with large entries (e.g., kernel matrices) aren't falsely rejected.
        var values = Tensor<T>.FromMatrix(matrix);
        var transposed = Engine.TensorTranspose(values);
        T diagScale = Engine.TensorMaxValue(Engine.TensorAbs(Engine.TensorDiagonal(values)));
        T relativeTolerance = MathHelper.Max(
            NumOps.Multiply(diagScale, NumOps.FromDouble(1e-6)),
            NumOps.FromDouble(1e-10));
        T threshold = NumOps.Multiply(relativeTolerance, NumOps.FromDouble(1000));

        var asymmetry = Engine.TensorAbs(Engine.TensorSubtract(values, transposed));
        if (NumOps.GreaterThan(Engine.TensorMaxValue(asymmetry), threshold))
        {
            // Report the first offending (i, j), j > i, in row-major order, as the scan did.
            var violations = Engine.TensorTriu(Engine.TensorGreaterThan(asymmetry, threshold), 1);
            int flat = Engine.TensorArgMax(violations.Reshape(new[] { violations.Length }), 0)[0];
            int i = flat / n;
            int j = flat % n;
            T diff = asymmetry[i, j];
            throw new ArgumentException(
                $"Matrix must be symmetric for Cholesky decomposition. " +
                $"Asymmetry at [{i},{j}]: {diff:E3} exceeds tolerance {threshold:E3}.");
        }

        // Symmetrize in place: (A + A^T) / 2 leaves the diagonal unchanged and averages each pair.
        var symmetric = Engine.TensorMultiplyScalar(Engine.TensorAdd(values, transposed), NumOps.FromDouble(0.5));
        matrix.SetSubMatrix(0, 0, symmetric.ToMatrix());
    }

    /// <summary>
    /// Computes the lower Cholesky factor with one engine factorization.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>The lower triangular matrix L.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the matrix is not square or not positive definite.
    /// </exception>
    private Matrix<T> ComputeCholeskyFactor(Matrix<T> matrix)
    {
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");
        }

        if (matrix.Rows == 0)
        {
            return new Matrix<T>(0, 0);
        }

        var (factor, info) = LinalgDispatch<T>.CholeskyEx(Tensor<T>.FromMatrix(matrix), upper: false);
        if (Engine.TensorCountNonzero(info) > 0)
        {
            throw new ArgumentException("Matrix is not positive definite.");
        }

        return factor.ToMatrix();
    }
}
