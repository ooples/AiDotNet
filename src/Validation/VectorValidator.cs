namespace AiDotNet.Validation;

/// <summary>
/// Provides validation methods for vectors and related operations.
/// </summary>
public static class VectorValidator
{
    /// <summary>
    /// Validates that the vector length matches the expected length.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The vector to validate.</param>
    /// <param name="expectedLength">The expected length of the vector.</param>
    /// <param name="component">Optional. The component performing the validation.</param>
    /// <param name="operation">Optional. The operation being performed.</param>
    /// <exception cref="VectorLengthMismatchException">Thrown when the vector length doesn't match the expected length.</exception>
    public static void ValidateLength<T>(Vector<T> vector, int expectedLength, string component = "", string operation = "")
    {
        var (resolvedComponent, resolvedOperation) = ValidationHelper<T>.ResolveCallerInfo(component, operation);

        if (vector.Length != expectedLength)
        {
            throw new VectorLengthMismatchException(expectedLength, vector.Length, resolvedComponent, resolvedOperation);
        }
    }

    /// <summary>
    /// Validates that the vector length matches the expected size based on the shape.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The vector to validate.</param>
    /// <param name="expectedShape">The expected shape when the vector is reshaped.</param>
    /// <param name="component">Optional. The component performing the validation.</param>
    /// <param name="operation">Optional. The operation being performed.</param>
    /// <exception cref="VectorLengthMismatchException">Thrown when the vector length doesn't match the product of the expected shape dimensions.</exception>
    public static void ValidateLengthForShape<T>(Vector<T> vector, int[] expectedShape, string component = "", string operation = "")
    {
        var (resolvedComponent, resolvedOperation) = ValidationHelper<T>.ResolveCallerInfo(component, operation);
        int expectedLength = expectedShape.Aggregate(1, (a, b) => a * b);

        if (vector.Length != expectedLength)
        {
            throw new VectorLengthMismatchException(expectedLength, vector.Length, resolvedComponent, resolvedOperation);
        }
    }
}
