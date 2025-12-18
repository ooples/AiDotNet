namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements the Normal Equation method for solving linear systems, especially useful for overdetermined systems.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrices and vectors.</typeparam>
/// <remarks>
/// <para>
/// The Normal Equation transforms a potentially non-square system Ax = b into a square system (A^T)Ax = (A^T)b,
/// which can then be solved using Cholesky decomposition for efficiency and stability.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you have more equations than unknowns (like having 10 data points but only 
/// wanting to find the slope and intercept of a line), this method helps find the "best fit" solution.
/// It works by converting your original problem into a simpler one that can be solved more easily.
/// Think of it like finding the average of several measurements - you're finding the solution that
/// minimizes the overall error.
/// </para>
/// </remarks>
public class NormalDecomposition<T> : MatrixDecompositionBase<T>
{
    // A property is inherited from MatrixDecompositionBase<T>

    /// <summary>
    /// The product of A-transpose and A, forming a square, symmetric matrix.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a transformed version of your original data that makes it easier to solve.
    /// It's always square (same number of rows and columns) which makes it more manageable.
    /// </remarks>
    private Matrix<T> _aTA { get; set; }

    /// <summary>
    /// The Cholesky decomposition of the A^T*A matrix, used to efficiently solve the system.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a special way of breaking down the transformed matrix into simpler parts
    /// that makes solving the equations much faster.
    /// </remarks>
    private CholeskyDecomposition<T> _choleskyDecomposition;

    /// <summary>
    /// Initializes a new instance of the NormalDecomposition class.
    /// </summary>
    /// <param name="matrix">The matrix A in the system Ax = b.</param>
    /// <remarks>
    /// <para>
    /// This constructor computes A^T*A and performs Cholesky decomposition on it.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This sets up everything needed to solve your problem. It takes your original data,
    /// transforms it into a more solvable form, and prepares the special decomposition that will make
    /// solving quick and accurate.
    /// </para>
    /// </remarks>
    public NormalDecomposition(Matrix<T> matrix) : base(matrix)
    {
        _aTA = A.Transpose().Multiply(A);
        _choleskyDecomposition = new CholeskyDecomposition<T>(_aTA);
    }

    /// <summary>
    /// Decomposition is performed in the constructor via Cholesky decomposition.
    /// </summary>
    protected override void Decompose()
    {
        // Normal equation decomposition is handled in the constructor
        // by computing A^T*A and creating a Cholesky decomposition
    }

    /// <summary>
    /// Solves the system Ax = b using the normal equation method.
    /// </summary>
    /// <param name="b">The right-hand side vector in the system Ax = b.</param>
    /// <returns>The solution vector x that best satisfies the system.</returns>
    /// <remarks>
    /// <para>
    /// This method solves (A^T)Ax = (A^T)b using Cholesky decomposition.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where the actual solving happens. Given your data points (matrix A) and 
    /// target values (vector b), this finds the values of your unknowns (vector x) that best fit your data.
    /// For example, if you're trying to find the best-fit line through several points, this would give you
    /// the slope and intercept of that line.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        // VECTORIZED: Uses matrix transpose and multiplication operations which are already vectorized
        var aTb = A.Transpose().Multiply(b);
        return _choleskyDecomposition.Solve(aTb);
    }

    /// <summary>
    /// Calculates the inverse of the original matrix using the normal equation method.
    /// </summary>
    /// <returns>The pseudo-inverse of matrix A.</returns>
    /// <remarks>
    /// <para>
    /// For non-square matrices, this returns the Moore-Penrose pseudo-inverse.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The inverse of a matrix is like the reciprocal of a number - when you multiply them together,
    /// you get the identity (like 1 in regular multiplication). For non-square matrices, we can't find a true inverse,
    /// so we calculate what's called a "pseudo-inverse" that's the next best thing. This is useful for many
    /// machine learning algorithms and data transformations.
    /// </para>
    /// <para>
    /// Note: Computing the inverse is generally less numerically stable than directly solving the system.
    /// When possible, use the Solve method instead.
    /// </para>
    /// </remarks>
    public override Matrix<T> Invert()
    {
        var invAta = _choleskyDecomposition.Invert();
        return invAta.Multiply(A.Transpose());
    }
}
