using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Performs LQ decomposition on a matrix, factoring it into a lower triangular matrix L and an orthogonal matrix Q.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// LQ decomposition factors a matrix A into the product A = LQ, where L is a lower triangular matrix
/// and Q is an orthogonal matrix. This decomposition is the transpose version of QR decomposition and
/// is particularly useful when working with matrices that have more columns than rows.
/// </para>
/// <para>
/// <b>For Beginners:</b> LQ decomposition breaks down a matrix A into two components:
/// L (a lower triangular matrix with values only on and below the diagonal) and
/// Q (an orthogonal matrix with columns that are perpendicular to each other).
/// This decomposition is useful for solving linear systems, least squares problems,
/// and other numerical linear algebra tasks.
/// </para>
/// <para>
/// Real-world applications:
/// - Solving underdetermined systems of equations
/// - Least squares problems with wide matrices
/// - Numerical stability in various computations
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Matrix Computations", "https://doi.org/10.56021/9781421407944")]
public class LqDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the lower triangular matrix L from the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The L matrix contains values only on and below the diagonal.
    /// It represents one part of the factorization of the original matrix.
    /// </remarks>
    public Matrix<T> L { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the orthogonal matrix Q from the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> An orthogonal matrix has columns that are perpendicular to each other
    /// and have unit length. This means Q^T * Q = I (the identity matrix).
    /// </remarks>
    public Matrix<T> Q { get; private set; } = new Matrix<T>(0, 0);

    private readonly LqAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the LqDecomposition class and performs the decomposition.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for the decomposition (default is Householder).</param>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor takes your input matrix and immediately performs
    /// the LQ decomposition using the specified algorithm. After creating this object,
    /// you can access the L and Q matrices or use methods like Solve() to work with the decomposition.
    /// </remarks>
    public LqDecomposition(Matrix<T> matrix, LqAlgorithmType algorithm = LqAlgorithmType.Householder)
        : base(matrix)
    {
        _algorithm = algorithm;
        Decompose();
    }

    /// <summary>
    /// Performs the LQ decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (L, Q) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Solves the linear system Ax = b using the LQ decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method finds the values of x that satisfy the equation Ax = b.
    /// It uses the LQ decomposition to solve this in two steps:
    ///
    /// 1. Forward substitution: Solve Ly = b for y
    /// 2. Multiply by Q^T: x = Q^T * y
    ///
    /// This approach is more efficient than directly inverting the matrix A.
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        int m = L.Rows;
        int n = Q.Rows;
        if (b.Length != m)
        {
            throw new ArgumentException(
                $"Vector b must have length {m} (the number of rows of A); got {b.Length}.", nameof(b));
        }

        if (m == 0 || n == 0)
        {
            return new Vector<T>(n);
        }

        if (m <= n)
        {
            // A = L[:, :m] Q[:m, :]: solve the leading m x m lower-triangular block, then
            // x = Q[:m, :]^T y is the minimum-norm solution of the (under)determined system.
            var y = ForwardSubstitution(L, b);
            var leadingQ = Q.Rows == m ? Q : Q.GetSubMatrix(0, 0, m, n);
            return leadingQ.Transpose().Multiply(y);
        }

        // Tall A (m > n): A = L Q with Q orthogonal, so the least-squares x is Q^T z where z
        // minimises ||L z - b||; z comes from the reduced QR of the m x n trapezoidal L.
        var (qL, rL) = LinalgDispatch<T>.QR(Tensor<T>.FromMatrix(L), "reduced");
        var projected = Engine.TensorMatMul(Engine.TensorTranspose(qL), Tensor<T>.FromVector(b, new[] { m, 1 }));
        var z = SolveTriangularSystem(rL.ToMatrix(), projected.Reshape(new[] { n }).ToVector(), upper: true);
        return Q.Transpose().Multiply(z);
    }

    /// <summary>
    /// Selects and applies the appropriate decomposition algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for decomposition.</param>
    /// <returns>A tuple containing the L and Q matrices.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    private (Matrix<T> L, Matrix<T> Q) ComputeDecomposition(Matrix<T> matrix, LqAlgorithmType algorithm)
    {
        return algorithm switch
        {
            LqAlgorithmType.Householder => ComputeLqHouseholder(matrix),
            LqAlgorithmType.GramSchmidt => ComputeLqGramSchmidt(matrix),
            LqAlgorithmType.Givens => ComputeLqGivens(matrix),
            _ => throw new ArgumentException("Unsupported LQ decomposition algorithm."),
        };
    }

    /// <summary>
    /// Computes the LQ decomposition using the Householder reflection method.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the L and Q matrices.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The Householder method uses reflections to transform the matrix
    /// into lower triangular form. It's numerically stable and efficient.
    ///
    /// LQ decomposition of A is computed via QR decomposition of A^T:
    /// If A^T = Q_qr * R, then A = R^T * Q_qr^T = L * Q
    /// where L = R^T (lower triangular) and Q = Q_qr^T (orthogonal).
    /// </remarks>
    private (Matrix<T> L, Matrix<T> Q) ComputeLqHouseholder(Matrix<T> matrix)
    {
        // A = LQ is the transpose of the complete Householder QR of A^T: L = R^T, Q = Q_qr^T.
        return ComputeLqFromEngineQr(matrix, positiveDiagonal: false);
    }

    /// <summary>
    /// Computes A = LQ from one complete engine QR factorization of A^T (A^T = Q_qr R).
    /// </summary>
    /// <param name="matrix">The m x n matrix to factor.</param>
    /// <param name="positiveDiagonal">
    /// True to scale the factors so that L has a non-negative diagonal (the Gram-Schmidt and Givens
    /// conventions); false keeps the Householder reflector signs.
    /// </param>
    /// <returns>L (m x n, lower trapezoidal) and Q (n x n, orthogonal).</returns>
    private (Matrix<T> L, Matrix<T> Q) ComputeLqFromEngineQr(Matrix<T> matrix, bool positiveDiagonal)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        if (m == 0 || n == 0)
        {
            return (new Matrix<T>(m, n), Matrix<T>.CreateIdentity(n));
        }

        var (q, r) = LinalgDispatch<T>.QR(Engine.TensorTranspose(Tensor<T>.FromMatrix(matrix)), "complete");

        if (positiveDiagonal)
        {
            // Row j of R and column j of Q_qr flip together, so A^T = Q_qr R is preserved.
            var signs = Engine.TensorSign(Engine.TensorDiagonal(r));
            signs = Engine.TensorAdd(signs, Engine.TensorSubtract(Engine.TensorOnesLike(signs), Engine.TensorAbs(signs)));
            int k = signs.Length;
            if (k < n)
            {
                signs = Engine.TensorSetSlice(
                    Engine.TensorFullLike(new Tensor<T>(new[] { n }), NumOps.One), signs, new[] { 0 });
            }

            var signMatrix = Engine.TensorDiagEmbed(signs);
            q = Engine.TensorMatMul(q, signMatrix);
            r = Engine.TensorMatMul(signMatrix, r);
        }

        return (Engine.TensorTranspose(r).ToMatrix(), Engine.TensorTranspose(q).ToMatrix());
    }

    /// <summary>
    /// Computes the LQ decomposition using the Gram-Schmidt orthogonalization process.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the L and Q matrices.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The Gram-Schmidt process transforms a set of vectors into a set of
    /// orthogonal vectors (vectors that are perpendicular to each other).
    ///
    /// LQ decomposition of A is computed via QR decomposition of A^T using Gram-Schmidt:
    /// If A^T = Q_qr * R, then A = R^T * Q_qr^T = L * Q
    /// where L = R^T (lower triangular) and Q = Q_qr^T (orthogonal).
    /// </remarks>
    private (Matrix<T> L, Matrix<T> Q) ComputeLqGramSchmidt(Matrix<T> matrix)
    {
        // Gram-Schmidt on the columns of A^T (completed to a full basis) is the complete QR of A^T
        // with a non-negative R diagonal.
        return ComputeLqFromEngineQr(matrix, positiveDiagonal: true);
    }

    /// <summary>
    /// Computes the LQ decomposition using Givens rotations.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the L and Q matrices.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Givens rotations are a way to zero out specific elements in a matrix
    /// by rotating two rows or columns at a time.
    ///
    /// LQ decomposition of A is computed via QR decomposition of A^T using Givens:
    /// If A^T = Q_qr * R, then A = R^T * Q_qr^T = L * Q
    /// where L = R^T (lower triangular) and Q = Q_qr^T (orthogonal).
    /// </remarks>
    private (Matrix<T> L, Matrix<T> Q) ComputeLqGivens(Matrix<T> matrix)
    {
        // Bottom-up Givens rotations leave R[j,j] = sqrt(a^2 + b^2) >= 0: the complete QR of A^T
        // with a non-negative R diagonal.
        return ComputeLqFromEngineQr(matrix, positiveDiagonal: true);
    }

    /// <summary>
    /// Solves a linear system Lx = b where L is a lower triangular matrix.
    /// </summary>
    /// <param name="L">The lower triangular matrix.</param>
    /// <param name="b">The right-hand side vector.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Forward substitution is a method to solve equations when the matrix
    /// is lower triangular (has non-zero values only on and below the diagonal).
    ///
    /// The process works by:
    /// 1. Solving for the first variable directly (since there are no other variables in the first equation)
    /// 2. Using that value to solve for the second variable
    /// 3. Continuing this pattern for all variables
    ///
    /// This is much faster than general matrix inversion because we can solve for each
    /// variable one at a time, working from top to bottom.
    /// </remarks>
    private Vector<T> ForwardSubstitution(Matrix<T> L, Vector<T> b)
    {
        int n = Math.Min(L.Rows, L.Columns);
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // Only the leading min(m, n) square lower-triangular block and the matching entries of b
        // take part in the substitution.
        var square = (L.Rows == n && L.Columns == n) ? L : L.GetSubMatrix(0, 0, n, n);
        var rhs = b.Length == n ? b : b.GetSubVector(0, n);
        return SolveTriangularSystem(square, rhs, upper: false);
    }
}
