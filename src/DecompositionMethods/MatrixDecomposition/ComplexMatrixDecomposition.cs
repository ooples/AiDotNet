namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// A wrapper class that adapts a real-valued matrix decomposition to work with complex numbers.
/// </summary>
/// <remarks>
/// This class allows you to use existing matrix decomposition algorithms with complex numbers
/// by wrapping a real-valued decomposition. Note that this implementation only works with
/// matrices that have real values (the imaginary parts are all zero).
/// </remarks>
/// <typeparam name="T">The numeric data type used for the real and imaginary parts (e.g., float, double).</typeparam>
public class ComplexMatrixDecomposition<T> : IMatrixDecomposition<Complex<T>>
{
    /// <summary>
    /// The underlying real-valued matrix decomposition.
    /// </summary>
    private readonly IMatrixDecomposition<T> _baseDecomposition = default!;
    
    /// <summary>
    /// Operations for the numeric type T (like addition, multiplication, etc.).
    /// </summary>
    private readonly INumericOperations<T> _ops = default!;

    /// <summary>
    /// Creates a new complex matrix decomposition by wrapping a real-valued decomposition.
    /// </summary>
    /// <param name="baseDecomposition">The real-valued matrix decomposition to wrap.</param>
    public ComplexMatrixDecomposition(IMatrixDecomposition<T> baseDecomposition)
    {
        _baseDecomposition = baseDecomposition;
        _ops = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the original matrix that was decomposed, converted to complex form.
    /// </summary>
    /// <remarks>
    /// This property returns the original matrix from the base decomposition,
    /// but converted to a complex matrix where all values have zero imaginary parts.
    /// </remarks>
    public Matrix<Complex<T>> A
    {
        get
        {
            var baseA = _baseDecomposition.A;
            var complexA = new Matrix<Complex<T>>(baseA.Rows, baseA.Columns);

            for (int i = 0; i < baseA.Rows; i++)
            {
                for (int j = 0; j < baseA.Columns; j++)
                {
                    complexA[i, j] = new Complex<T>(baseA[i, j], _ops.Zero);
                }
            }

            return complexA;
        }
    }

    /// <summary>
    /// Calculates the inverse of the original matrix.
    /// </summary>
    /// <remarks>
    /// The inverse of a matrix A is another matrix A⁻¹ such that A × A⁻¹ = I, 
    /// where I is the identity matrix. This method uses the base decomposition
    /// to calculate the inverse and then converts it to complex form.
    /// </remarks>
    /// <returns>The inverse of the original matrix as a complex matrix.</returns>
    public Matrix<Complex<T>> Invert()
    {
        var baseInverse = _baseDecomposition.Invert();
        var complexInverse = new Matrix<Complex<T>>(baseInverse.Rows, baseInverse.Columns);

        for (int i = 0; i < baseInverse.Rows; i++)
        {
            for (int j = 0; j < baseInverse.Columns; j++)
            {
                complexInverse[i, j] = new Complex<T>(baseInverse[i, j], _ops.Zero);
            }
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
    public Vector<Complex<T>> Solve(Vector<Complex<T>> b)
    {
        // Extract real parts of b
        var realB = new Vector<T>(b.Length);
        for (int i = 0; i < b.Length; i++)
        {
            realB[i] = b[i].Real;
        }

        // Solve using base decomposition
        var realSolution = _baseDecomposition.Solve(realB);

        // Convert solution to complex
        var complexSolution = new Vector<Complex<T>>(realSolution.Length);
        for (int i = 0; i < realSolution.Length; i++)
        {
            complexSolution[i] = new Complex<T>(realSolution[i], _ops.Zero);
        }

        return complexSolution;
    }
}