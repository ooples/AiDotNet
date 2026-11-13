using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements the Gram-Schmidt orthogonalization process to decompose a matrix into an orthogonal matrix Q and an upper triangular matrix R.
/// </summary>
/// <typeparam name="T">The numeric data type used in calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Gram-Schmidt process transforms a set of vectors into a set of orthogonal vectors (vectors that are perpendicular to each other).
/// This decomposition is useful for solving linear systems, computing least squares solutions, and other numerical applications.
/// The result is a QR factorization where A = Q * R.
/// </para>
/// <para>
/// <b>For Beginners:</b> This class takes a matrix and breaks it down into two special matrices:
/// Q (a matrix with perpendicular columns) and R (an upper triangular matrix with values only on and above the diagonal).
/// Think of it like organizing a messy set of vectors into a neat, perpendicular coordinate system.
/// Together, these matrices can be multiplied to get back the original matrix: A = Q * R.
/// </para>
/// <para>
/// Real-world applications:
/// - Solving systems of linear equations
/// - Computing least squares solutions in regression analysis
/// - Numerical stability improvements in various algorithms
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Matrix Computations", "https://doi.org/10.56021/9781421407944")]
public class GramSchmidtDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the orthogonal matrix Q from the decomposition.
    /// </summary>
    /// <remarks>
    /// The Q matrix has columns that are orthogonal to each other (perpendicular).
    /// Each column has a length (norm) of 1, making it an "orthonormal" matrix.
    /// This property makes Q useful for transforming vectors while preserving their length.
    /// </remarks>
    public Matrix<T> Q { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the upper triangular matrix R from the decomposition.
    /// </summary>
    /// <remarks>
    /// The R matrix is upper triangular, meaning it only has non-zero values on and above the diagonal.
    /// This structure makes it easy to solve systems of equations through back-substitution.
    /// </remarks>
    public Matrix<T> R { get; private set; } = new Matrix<T>(0, 0);

    private readonly GramSchmidtAlgorithmType _algorithm;

    /// <summary>
    /// Creates a new Gram-Schmidt decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The type of Gram-Schmidt algorithm to use (Classical or Modified).</param>
    /// <remarks>
    /// The constructor initializes the decomposition by computing the Q and R matrices.
    /// Two algorithm variants are available:
    /// - Classical: The original algorithm, which can suffer from numerical instability
    /// - Modified: An improved version with better numerical stability for most applications
    /// </remarks>
    public GramSchmidtDecomposition(Matrix<T> matrix, GramSchmidtAlgorithmType algorithm = GramSchmidtAlgorithmType.Classical)
        : base(matrix)
    {
        _algorithm = algorithm;
        Decompose();
    }

    /// <summary>
    /// Performs the Gram-Schmidt decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (Q, R) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Selects and applies the appropriate Gram-Schmidt algorithm based on the specified type.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The type of Gram-Schmidt algorithm to use.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm type is specified.</exception>
    private (Matrix<T> Q, Matrix<T> R) ComputeDecomposition(Matrix<T> matrix, GramSchmidtAlgorithmType algorithm)
    {
        return algorithm switch
        {
            GramSchmidtAlgorithmType.Classical => ComputeClassicalGramSchmidt(matrix),
            GramSchmidtAlgorithmType.Modified => ComputeModifiedGramSchmidt(matrix),
            _ => throw new ArgumentException("Unsupported Gram-Schmidt algorithm.")
        };
    }

    /// <summary>
    /// Computes the QR decomposition using the Classical Gram-Schmidt algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// The Classical Gram-Schmidt algorithm works as follows:
    /// 1. For each column in the input matrix:
    ///    a. Take the current column vector
    ///    b. Subtract its projections onto all previous orthogonal vectors
    ///    c. Normalize the resulting vector to have length 1
    /// 
    /// While simpler to understand, this method can suffer from numerical instability
    /// when dealing with nearly linearly dependent columns.
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeClassicalGramSchmidt(Matrix<T> matrix)
    {
        // Classical Gram-Schmidt yields the unique thin QR factorization with R[j,j] = ‖v_j‖ > 0;
        // the engine QR computes the same factors in one whole-matrix call.
        return ComputePositiveDiagonalQr(matrix);
    }

    /// <summary>
    /// Computes the QR decomposition using the Modified Gram-Schmidt algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// The Modified Gram-Schmidt algorithm is a more numerically stable version that:
    /// 1. Takes each column vector one by one
    /// 2. Immediately updates the vector after each projection subtraction
    /// 3. This reduces accumulated rounding errors compared to the classical method
    /// 
    /// This method is generally preferred for practical applications as it produces
    /// more accurate results, especially with nearly linearly dependent columns.
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeModifiedGramSchmidt(Matrix<T> matrix)
    {
        // Modified Gram-Schmidt computes the same factors as the classical variant (with better
        // rounding); the engine's Householder QR is at least as stable.
        return ComputePositiveDiagonalQr(matrix);
    }

    /// <summary>
    /// Solves a system of linear equations Ax = b using the QR decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// This method solves the system in two steps:
    /// 1. First, it solves Qy = b for y by computing y = Q^T * b
    ///    (This works because Q is orthogonal, so Q^T * Q = I)
    /// 2. Then, it solves Rx = y using back-substitution
    ///    (This is efficient because R is upper triangular)
    ///
    /// The result is the solution vector x that satisfies Ax = b.
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        // Solve Qy = b by computing y = Q^T * b
        var y = Q.Transpose().Multiply(b);

        // Solve Rx = y with one engine triangular solve
        return SolveTriangularSystem(R, y, upper: true);
    }

    /// <summary>
    /// Calculates the inverse of the original matrix A using the QR decomposition.
    /// </summary>
    /// <returns>The inverse matrix of A.</returns>
    /// <remarks>
    /// This method finds the inverse by solving the equation AX = I, where:
    /// - A is the original matrix
    /// - I is the identity matrix
    /// - X is the inverse of A
    ///
    /// The process works by:
    /// 1. Creating an identity matrix of the same size as A
    /// 2. Solving the system AX = I for each column of X separately
    /// 3. Combining the solution vectors to form the complete inverse matrix
    ///
    /// The inverse exists only if A is square and has full rank (all columns are linearly independent).
    /// Using QR decomposition for matrix inversion is numerically stable compared to some other methods.
    /// </remarks>
    public override Matrix<T> Invert()
    {
        int n = A.Rows;
        var identity = Matrix<T>.CreateIdentity(n);
        var inverse = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            inverse.SetColumn(i, Solve(identity.GetColumn(i)));
        }

        return inverse;
    }
}
