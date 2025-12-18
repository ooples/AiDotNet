namespace AiDotNet.Exceptions;

/// <summary>
/// Exception thrown when a vector's length doesn't match the expected value.
/// </summary>
public class VectorLengthMismatchException : AiDotNetException
{
    /// <summary>
    /// The expected length of the vector.
    /// </summary>
    public int ExpectedLength { get; }

    /// <summary>
    /// The actual length of the vector.
    /// </summary>
    public int ActualLength { get; }

    /// <summary>
    /// The component where the length mismatch occurred.
    /// </summary>
    public string Component { get; }

    /// <summary>
    /// The operation being performed when the mismatch was detected.
    /// </summary>
    public string Operation { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="VectorLengthMismatchException"/> class.
    /// </summary>
    /// <param name="expectedLength">The expected length of the vector.</param>
    /// <param name="actualLength">The actual length of the vector.</param>
    /// <param name="component">The component where the length mismatch occurred.</param>
    /// <param name="operation">The operation being performed when the mismatch was detected.</param>
    public VectorLengthMismatchException(int expectedLength, int actualLength, string component, string operation)
        : base(FormatMessage(expectedLength, actualLength, component, operation))
    {
        ExpectedLength = expectedLength;
        ActualLength = actualLength;
        Component = component;
        Operation = operation;
    }

    private static string FormatMessage(int expectedLength, int actualLength, string component, string operation)
    {
        return $"Vector length mismatch in {component} during {operation}: Expected length {expectedLength}, but got {actualLength}.";
    }
}
