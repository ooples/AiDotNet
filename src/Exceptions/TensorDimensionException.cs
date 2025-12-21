namespace AiDotNet.Exceptions;

/// <summary>
/// Exception thrown when a tensor's dimension doesn't match the expected value.
/// </summary>
public class TensorDimensionException : AiDotNetException
{
    /// <summary>
    /// The index of the dimension that doesn't match.
    /// </summary>
    public int DimensionIndex { get; }

    /// <summary>
    /// The expected value of the dimension.
    /// </summary>
    public int ExpectedValue { get; }

    /// <summary>
    /// The actual value of the dimension.
    /// </summary>
    public int ActualValue { get; }

    /// <summary>
    /// The component where the dimension mismatch occurred.
    /// </summary>
    public string Component { get; }

    /// <summary>
    /// The operation being performed when the mismatch was detected.
    /// </summary>
    public string Operation { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorDimensionException"/> class.
    /// </summary>
    /// <param name="dimensionIndex">The index of the dimension that doesn't match.</param>
    /// <param name="expectedValue">The expected value of the dimension.</param>
    /// <param name="actualValue">The actual value of the dimension.</param>
    /// <param name="component">The component where the dimension mismatch occurred.</param>
    /// <param name="operation">The operation being performed when the mismatch was detected.</param>
    public TensorDimensionException(int dimensionIndex, int expectedValue, int actualValue, string component, string operation)
        : base(FormatMessage(dimensionIndex, expectedValue, actualValue, component, operation))
    {
        DimensionIndex = dimensionIndex;
        ExpectedValue = expectedValue;
        ActualValue = actualValue;
        Component = component;
        Operation = operation;
    }

    private static string FormatMessage(int dimensionIndex, int expectedValue, int actualValue, string component, string operation)
    {
        return $"Dimension mismatch in {component} during {operation}: " +
               $"Expected dimension {dimensionIndex} to be {expectedValue}, but got {actualValue}.";
    }
}
