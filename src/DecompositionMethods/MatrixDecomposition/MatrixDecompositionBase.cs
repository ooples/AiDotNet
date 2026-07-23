namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Base class for matrix decomposition algorithms that break down matrices into simpler components.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Matrix decomposition is like factoring a number, but for matrices.
/// Just as we can break down 12 into 3 * 4, we can break down complex matrices into simpler
/// matrices that are easier to work with. Different decomposition methods have different strengths
/// and are used for different purposes in machine learning and numerical computing.
/// </para>
/// <para>
/// This base class provides common functionality that all matrix decompositions need, such as:
/// - Storing the original matrix
/// - Providing numeric operations
/// - Implementing matrix inversion
/// - Solving linear systems
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
public abstract class MatrixDecompositionBase<T> : IMatrixDecomposition<T>
{
    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper object knows how to perform math operations
    /// on the specific number type you're using (like float or double). It ensures
    /// that mathematical operations work correctly regardless of the numeric type.
    /// </para>
    /// </remarks>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// The original matrix that was decomposed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the matrix you started with before breaking it down
    /// into simpler components. It's kept here so you can refer back to it if needed.
    /// </para>
    /// </remarks>
    public Matrix<T> A { get; protected set; }

    /// <summary>
    /// Initializes a new instance of the matrix decomposition class.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the common parts that all matrix
    /// decompositions need. It stores the original matrix and initializes the numeric
    /// operations helper.
    /// </para>
    /// </remarks>
    protected MatrixDecompositionBase(Matrix<T> matrix)
    {
        Guard.NotNull(matrix);
        A = matrix;
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Performs the actual decomposition of the matrix into its components.
    /// This method must be implemented by derived classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where the actual "work" of breaking down the matrix happens.
    /// Each specific decomposition method (LU, QR, SVD, NMF, ICA, etc.) has its own unique
    /// algorithm, which is why each derived class implements this method differently.
    /// </para>
    /// </remarks>
    protected abstract void Decompose();

    /// <summary>
    /// Solves a linear system of equations Ax = b, where A is the decomposed matrix.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method solves equations of the form "Ax = b" where:
    /// - A is your matrix (the one that was decomposed)
    /// - b is a known vector (like a set of results or outcomes)
    /// - x is what you're trying to find (like the unknown variables)
    /// </para>
    /// <para>
    /// Real-world example: If you're calculating the best mix of ingredients for a recipe:
    /// - Matrix A represents how each ingredient affects taste, texture, and nutrition
    /// - Vector b represents your desired taste, texture, and nutrition targets
    /// - Vector x (the solution) tells you how much of each ingredient to use
    /// </para>
    /// <para>
    /// Using the decomposition to solve the system is much faster and more accurate
    /// than directly calculating the inverse of A.
    /// </para>
    /// </remarks>
    public abstract Vector<T> Solve(Vector<T> b);

    /// <summary>
    /// Calculates the inverse of the original matrix A using the decomposition.
    /// </summary>
    /// <returns>The inverse of the matrix A.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The inverse of a matrix is like the reciprocal of a number.
    /// Just as 5 * (1/5) = 1, a matrix multiplied by its inverse gives the identity matrix
    /// (which is like the number 1 in matrix form).
    /// </para>
    /// <para>
    /// For example, if matrix A represents a transformation (like rotating or scaling),
    /// then A-¹ represents the opposite transformation that "undoes" the original:
    /// - If A rotates data clockwise, A-¹ rotates it counterclockwise
    /// - If A scales data up by 2x, A-¹ scales it down by 1/2
    /// </para>
    /// <para>
    /// This default implementation uses the decomposition to compute the inverse efficiently.
    /// Some derived classes may override this method if they have a more efficient
    /// algorithm specific to their decomposition type.
    /// </para>
    /// </remarks>
    public virtual Matrix<T> Invert()
    {
        return MatrixHelper<T>.InvertUsingDecomposition(this);
    }

    /// <summary>
    /// Validates that the matrix meets the requirements for this decomposition method.
    /// </summary>
    /// <param name="matrix">The matrix to validate.</param>
    /// <param name="requireSquare">Whether the matrix must be square.</param>
    /// <param name="requireNonNegative">Whether all elements must be non-negative.</param>
    /// <param name="requireSymmetric">Whether the matrix must be symmetric.</param>
    /// <exception cref="ArgumentException">Thrown when the matrix doesn't meet the requirements.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different decomposition methods have different requirements.
    /// For example:
    /// - Some methods only work on square matrices (same number of rows and columns)
    /// - Some require all values to be non-negative (zero or positive)
    /// - Some require the matrix to be symmetric (mirror image across the diagonal)
    /// </para>
    /// <para>
    /// This helper method checks these requirements and gives a clear error message
    /// if something is wrong, rather than producing incorrect results.
    /// </para>
    /// </remarks>
    protected void ValidateMatrix(
        Matrix<T> matrix,
        bool requireSquare = false,
        bool requireNonNegative = false,
        bool requireSymmetric = false)
    {
        if (requireSquare && matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException(
                $"Matrix must be square for this decomposition. Got {matrix.Rows}x{matrix.Columns} matrix.",
                nameof(matrix));
        }

        if (requireNonNegative)
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    if (NumOps.LessThan(matrix[i, j], NumOps.Zero))
                    {
                        throw new ArgumentException(
                            $"Matrix must contain only non-negative values for this decomposition. " +
                            $"Found negative value {matrix[i, j]} at position ({i}, {j}).",
                            nameof(matrix));
                    }
                }
            }
        }

        if (requireSymmetric)
        {
            if (matrix.Rows != matrix.Columns)
            {
                throw new ArgumentException(
                    "Matrix must be square to be symmetric.",
                    nameof(matrix));
            }

            T tolerance = NumOps.FromDouble(1e-10);
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = i + 1; j < matrix.Columns; j++)
                {
                    T diff = NumOps.Abs(NumOps.Subtract(matrix[i, j], matrix[j, i]));
                    if (NumOps.GreaterThan(diff, tolerance))
                    {
                        throw new ArgumentException(
                            $"Matrix must be symmetric for this decomposition. " +
                            $"Elements at ({i}, {j}) and ({j}, {i}) differ by {diff}.",
                            nameof(matrix));
                    }
                }
            }
        }
    }

    /// <summary>
    /// Computes the Frobenius norm of a matrix (square root of sum of squared elements).
    /// </summary>
    /// <param name="matrix">The matrix to compute the norm for.</param>
    /// <returns>The Frobenius norm of the matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Frobenius norm is a way to measure the "size" or "magnitude"
    /// of a matrix. It's like measuring the length of a vector, but for a matrix.
    /// </para>
    /// <para>
    /// It's calculated by:
    /// 1. Squaring every element in the matrix
    /// 2. Adding all those squares together
    /// 3. Taking the square root of that sum
    /// </para>
    /// <para>
    /// This is useful for measuring how different two matrices are (by computing the norm
    /// of their difference) or for checking convergence in iterative algorithms.
    /// </para>
    /// </remarks>
    protected T FrobeniusNorm(Matrix<T> matrix)
    {
        T sumOfSquares = NumOps.Zero;

        // VECTORIZED: Process each row as a vector for SIMD optimization
        for (int i = 0; i < matrix.Rows; i++)
        {
            Vector<T> row = matrix.GetRow(i);
            // Dot product of row with itself gives sum of squares for that row
            T rowSumOfSquares = row.DotProduct(row);
            sumOfSquares = NumOps.Add(sumOfSquares, rowSumOfSquares);
        }

        return NumOps.Sqrt(sumOfSquares);
    }

    /// <summary>
    /// Checks if two values are approximately equal within a tolerance.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <param name="tolerance">The maximum acceptable difference.</param>
    /// <returns>True if the values are within tolerance; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In numerical computing, we rarely check if two floating-point
    /// numbers are exactly equal because of rounding errors. Instead, we check if they're
    /// "close enough" within some small tolerance.
    /// </para>
    /// <para>
    /// For example, if tolerance is 0.001, then 1.0001 and 1.0002 would be considered equal,
    /// but 1.0 and 1.002 would not.
    /// </para>
    /// </remarks>
    protected bool ApproximatelyEqual(T a, T b, T tolerance)
    {
        T diff = NumOps.Abs(NumOps.Subtract(a, b));
        return NumOps.LessThanOrEquals(diff, tolerance);
    }
}
