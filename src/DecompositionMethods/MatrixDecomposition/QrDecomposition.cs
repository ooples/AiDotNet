using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Performs QR decomposition on a matrix, factoring it into an orthogonal matrix Q and an upper triangular matrix R.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// QR decomposition factors a matrix A into the product of an orthogonal matrix Q and an upper triangular
/// matrix R. This decomposition is widely used in solving linear systems, computing eigenvalues, and least
/// squares problems. Multiple algorithms are available, each with different performance characteristics.
/// </para>
/// <para>
/// <b>For Beginners:</b> QR decomposition breaks down a matrix into two parts - Q (which has perpendicular columns with length 1)
/// and R (which is triangular with zeros below the diagonal). This is useful for solving equations and other matrix operations.
/// Think of it like factoring a number into its prime components, but for matrices.
/// </para>
/// <para>
/// Real-world applications:
/// - Solving systems of linear equations
/// - Computing eigenvalues and eigenvectors
/// - Least squares regression in statistics
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Matrix Computations", "https://doi.org/10.56021/9781421407944")]
public class QrDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the orthogonal matrix Q from the decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Q matrix has special properties - its columns are perpendicular to each other
    /// (orthogonal) and each column has a length of 1. This makes it useful for many calculations.
    /// </para>
    /// </remarks>
    public Matrix<T> Q { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the upper triangular matrix R from the decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The R matrix is upper triangular, which means it has numbers on and above the diagonal,
    /// and zeros below the diagonal. This structure makes solving equations much easier.
    /// </para>
    /// </remarks>
    public Matrix<T> R { get; private set; } = new Matrix<T>(0, 0);

    private readonly QrAlgorithmType _algorithm;

    /// <summary>
    /// Creates a new QR decomposition of the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="qrAlgorithm">The algorithm to use for QR decomposition (default is Householder).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different algorithms can be used to perform QR decomposition. Each has advantages in terms of
    /// speed, accuracy, or memory usage. Householder is generally a good default choice for most applications.
    /// </para>
    /// </remarks>
    public QrDecomposition(Matrix<T> matrix, QrAlgorithmType qrAlgorithm = QrAlgorithmType.Householder)
        : base(matrix)
    {
        _algorithm = qrAlgorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the QR decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (Q, R) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Solves the linear system Ax = b using the QR decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector.</param>
    /// <returns>The solution vector x such that Ax * b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method solves equations of the form Ax = b, where A is a matrix, and x and b are vectors.
    /// It finds the values of x that make the equation true. Using QR decomposition often gives more accurate results
    /// than other methods, especially for complex matrices.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        var y = Q.Transpose().Multiply(b);
        return BackSubstitution(R, y);
    }

    /// <summary>
    /// Selects and applies the appropriate QR decomposition algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    private (Matrix<T> Q, Matrix<T> R) ComputeDecomposition(Matrix<T> matrix, QrAlgorithmType algorithm)
    {
        return algorithm switch
        {
            QrAlgorithmType.GramSchmidt => ComputeQrGramSchmidt(matrix),
            QrAlgorithmType.Householder => ComputeQrHouseholder(matrix),
            QrAlgorithmType.Givens => ComputeQrGivens(matrix),
            QrAlgorithmType.ModifiedGramSchmidt => ComputeQrModifiedGramSchmidt(matrix),
            QrAlgorithmType.IterativeGramSchmidt => ComputeQrIterativeGramSchmidt(matrix),
            _ => throw new ArgumentException("Unsupported QR decomposition algorithm.")
        };
    }

    /// <summary>
    /// Computes QR decomposition using the classical Gram-Schmidt process.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Gram-Schmidt process transforms a set of vectors into a set of perpendicular vectors.
    /// It works by taking each column of the matrix and removing any components that point in the same direction
    /// as previous columns. This is like ensuring that each new direction is completely different from all previous ones.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrGramSchmidt(Matrix<T> matrix)
    {
        // Classical Gram-Schmidt: the thin QR with R[j,j] = ‖v_j‖ >= 0.
        return ComputeSignNormalizedQr(matrix, complete: false, nonNegativeDiagonal: true);
    }

    /// <summary>
    /// Computes QR decomposition using Householder reflections.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Householder reflections use special matrices that act like mirrors, reflecting vectors
    /// across a plane. This method is more numerically stable than Gram-Schmidt, meaning it gives more accurate
    /// results, especially for large or complex matrices. It works by systematically creating zeros below the diagonal
    /// of the matrix.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrHouseholder(Matrix<T> matrix)
    {
        // Complete Householder QR. Each column is reflected with u = x + ‖x‖ e1, which maps it to
        // -‖x‖ e1, so R[k,k] <= 0.
        return ComputeSignNormalizedQr(matrix, complete: true, nonNegativeDiagonal: false);
    }

    /// <summary>
    /// Computes QR decomposition using Givens rotations.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Givens rotations work by rotating the matrix in a plane to create zeros in specific positions.
    /// Think of it like carefully turning a dial to make certain values become zero. This method is particularly useful
    /// for sparse matrices (matrices with many zeros) because it can target specific elements without changing others.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrGivens(Matrix<T> matrix)
    {
        // Complete QR by bottom-up Givens rotations; each rotation leaves r = sqrt(a^2 + b^2) >= 0
        // on the diagonal.
        return ComputeSignNormalizedQr(matrix, complete: true, nonNegativeDiagonal: true);
    }

    /// <summary>
    /// Computes QR decomposition using the Modified Gram-Schmidt process, which is more numerically stable.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Modified Gram-Schmidt process is an improved version of the classical Gram-Schmidt method.
    /// It performs the same task (creating perpendicular vectors) but does it in a way that reduces rounding errors
    /// in calculations. This makes it more reliable for complex problems or when high precision is needed.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrModifiedGramSchmidt(Matrix<T> matrix)
    {
        // Modified Gram-Schmidt computes the same thin factors as the classical variant.
        return ComputeSignNormalizedQr(matrix, complete: false, nonNegativeDiagonal: true);
    }

    /// <summary>
    /// Computes QR decomposition using the Iterative Gram-Schmidt process for enhanced numerical stability.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Iterative Gram-Schmidt method repeats the orthogonalization process multiple times
    /// to achieve better accuracy. Think of it like double-checking your work to make sure the vectors are truly
    /// perpendicular to each other. This method is useful when working with matrices that might cause precision problems.
    /// </para>
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeQrIterativeGramSchmidt(Matrix<T> matrix)
    {
        // Re-orthogonalized Gram-Schmidt computes the same thin factors with less loss of
        // orthogonality; the engine's Householder QR is orthogonal to working precision.
        return ComputeSignNormalizedQr(matrix, complete: false, nonNegativeDiagonal: true);
    }

    /// <summary>
    /// Solves a system of linear equations using back substitution on an upper triangular matrix.
    /// </summary>
    /// <param name="R">The upper triangular matrix.</param>
    /// <param name="y">The right-hand side vector.</param>
    /// <returns>The solution vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Back substitution is a simple way to solve equations when your matrix is triangular.
    /// It works by starting at the bottom row (which has only one unknown) and solving for that value.
    /// Then it moves up row by row, using the values already found to solve for each new unknown.
    /// This is like solving a puzzle where each piece you place helps you figure out where the next one goes.
    /// </para>
    /// </remarks>
    private Vector<T> BackSubstitution(Matrix<T> R, Vector<T> y)
    {
        int n = R.Columns;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // Only the leading n x n upper-triangular block and the first n entries of y take part.
        var square = R.Rows == n ? R : R.GetSubMatrix(0, 0, n, n);
        var rhs = y.Length == n ? y : y.GetSubVector(0, n);
        return SolveTriangularSystem(square, rhs, upper: true);
    }
}
