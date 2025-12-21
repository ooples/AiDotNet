namespace AiDotNet.Extensions;

/// <summary>
/// Provides extension methods for matrix decomposition operations, enhancing their functionality.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Matrix decomposition is a way to break down complex matrices into simpler components,
/// making certain mathematical operations easier. This class adds helpful methods to work with these
/// decompositions in your AI applications.
/// 
/// Think of matrix decomposition like factoring a number (e.g., 12 = 3 × 4), but for matrices.
/// These decompositions are important in many AI algorithms for solving equations efficiently.
/// </remarks>
public static class MatrixDecompositionExtensions
{
    /// <summary>
    /// Converts a regular matrix decomposition to a complex-valued matrix decomposition.
    /// </summary>
    /// <typeparam name="T">The numeric data type used in the original decomposition (e.g., float, double).</typeparam>
    /// <param name="decomposition">The original matrix decomposition to convert.</param>
    /// <returns>A complex-valued version of the original matrix decomposition.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Complex numbers are numbers with both a real part and an "imaginary" part.
    /// They're useful in advanced mathematics and certain AI algorithms.
    /// 
    /// This method takes a regular matrix decomposition (using normal numbers) and converts it to
    /// use complex numbers instead. This is helpful when your AI algorithm needs to work with
    /// complex mathematical operations.
    /// 
    /// Example usage:
    /// <code>
    /// // Assuming you have a matrix decomposition for a real-valued matrix
    /// var realDecomposition = Matrix.Decompose(myMatrix);
    /// 
    /// // Convert it to work with complex numbers
    /// var complexDecomposition = realDecomposition.ToComplexDecomposition();
    /// 
    /// // Now you can use complexDecomposition for operations requiring complex numbers
    /// </code>
    /// 
    /// You typically need this when working with certain advanced algorithms like Fourier transforms,
    /// eigenvalue problems, or when analyzing oscillatory systems.
    /// </remarks>
    public static IMatrixDecomposition<Complex<T>> ToComplexDecomposition<T>(this IMatrixDecomposition<T> decomposition)
    {
        return new ComplexMatrixDecomposition<T>(decomposition);
    }
}
