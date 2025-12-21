namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// A wrapper class that adapts a real-valued matrix decomposition to work with complex numbers.
/// </summary>
/// <typeparam name="T">The numeric data type used for the real and imaginary parts (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This class allows you to use existing matrix decomposition algorithms with complex numbers
/// by wrapping a real-valued decomposition. Note that this implementation only works with
/// matrices that have real values (the imaginary parts are all zero).
/// </para>
/// <para>
/// <b>For Beginners:</b> This is a wrapper that lets you apply regular matrix decomposition methods
/// to complex numbers. Complex numbers have both real and imaginary parts (like 3 + 4i), but this
/// implementation currently works best when the imaginary parts are zero.
/// </para>
/// <para>
/// Real-world applications:
/// - Transitioning between real and complex number computations
/// - Testing complex-valued algorithms with real-valued data
/// - Quantum mechanics simulations with real-valued initial conditions
/// </para>
/// </remarks>
public class ComplexMatrixDecomposition<T> : MatrixDecompositionBase<Complex<T>>
{
    /// <summary>
    /// The underlying real-valued matrix decomposition.
    /// </summary>
    private readonly IMatrixDecomposition<T> _baseDecomposition;

    /// <summary>
    /// Operations for the numeric type T (like addition, multiplication, etc.).
    /// </summary>
    private readonly INumericOperations<T> _realOps;

    /// <summary>
    /// Creates a new complex matrix decomposition by wrapping a real-valued decomposition.
    /// </summary>
    /// <param name="baseDecomposition">The real-valued matrix decomposition to wrap.</param>
    public ComplexMatrixDecomposition(IMatrixDecomposition<T> baseDecomposition)
        : base(ConvertToComplexMatrix(baseDecomposition.A, MathHelper.GetNumericOperations<T>()))
    {
        _baseDecomposition = baseDecomposition;
        _realOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Converts a real-valued matrix to a complex matrix with zero imaginary parts.
    /// </summary>
    private static Matrix<Complex<T>> ConvertToComplexMatrix(Matrix<T> realMatrix, INumericOperations<T> ops)
    {
        var complexMatrix = new Matrix<Complex<T>>(realMatrix.Rows, realMatrix.Columns);

        // VECTORIZED: Process each row as a vector operation
        for (int i = 0; i < realMatrix.Rows; i++)
        {
            Vector<T> realRow = realMatrix.GetRow(i);
            // Convert to array first to avoid nested Select operations for better performance
            Complex<T>[] complexArray = realRow.Select(val => new Complex<T>(val, ops.Zero)).ToArray();
            Vector<Complex<T>> complexRow = new Vector<Complex<T>>(complexArray);
            complexMatrix.SetRow(i, complexRow);
        }

        return complexMatrix;
    }

    /// <summary>
    /// Decomposition is handled by the wrapped base decomposition.
    /// </summary>
    protected override void Decompose()
    {
        // Decomposition is performed by the wrapped _baseDecomposition
    }

    // A property is inherited from MatrixDecompositionBase<Complex<T>>

    /// <summary>
    /// Calculates the inverse of the original matrix.
    /// </summary>
    /// <remarks>
    /// The inverse of a matrix A is another matrix A^-1 such that A * A^-1 = I,
    /// where I is the identity matrix. This method uses the base decomposition
    /// to calculate the inverse and then converts it to complex form.
    /// </remarks>
    /// <returns>The inverse of the original matrix as a complex matrix.</returns>
    public override Matrix<Complex<T>> Invert()
    {
        var baseInverse = _baseDecomposition.Invert();
        var complexInverse = new Matrix<Complex<T>>(baseInverse.Rows, baseInverse.Columns);

        // VECTORIZED: Process each row as a vector operation
        for (int i = 0; i < baseInverse.Rows; i++)
        {
            Vector<T> realRow = baseInverse.GetRow(i);
            Vector<Complex<T>> complexRow = new Vector<Complex<T>>(realRow.Select(val => new Complex<T>(val, _realOps.Zero)));
            complexInverse.SetRow(i, complexRow);
        }

        return complexInverse;
    }

    /// <summary>
    /// Solves a linear system of equations Ax = b, where A is the decomposed matrix.
    /// </summary>
    /// <remarks>
    /// This method extracts the real parts of the input vector b, solves the system
    /// using the base decomposition, and then converts the solution back to complex form.
    ///
    /// Note: This implementation only works correctly when the input vector b has zero
    /// imaginary parts. For fully complex systems, a different implementation would be needed.
    /// </remarks>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    public override Vector<Complex<T>> Solve(Vector<Complex<T>> b)
    {
        // VECTORIZED: Extract real parts using vector Select
        var realB = new Vector<T>(b.Select(c => c.Real));

        // Solve using base decomposition
        var realSolution = _baseDecomposition.Solve(realB);

        // VECTORIZED: Convert solution to complex using vector Select
        var complexSolution = new Vector<Complex<T>>(realSolution.Select(val => new Complex<T>(val, _realOps.Zero)));

        return complexSolution;
    }
}
