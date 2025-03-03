namespace AiDotNet.Exceptions;

/// <summary>
/// Exception thrown when a tensor's shape doesn't match the expected shape.
/// </summary>
public class TensorShapeMismatchException : AiDotNetException
{
    /// <summary>
    /// The expected shape of the tensor.
    /// </summary>
    public int[] ExpectedShape { get; }

    /// <summary>
    /// The actual shape of the tensor.
    /// </summary>
    public int[] ActualShape { get; }

    /// <summary>
    /// The component where the shape mismatch occurred.
    /// </summary>
    public string Component { get; }

    /// <summary>
    /// The operation being performed when the mismatch was detected.
    /// </summary>
    public string Operation { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorShapeMismatchException"/> class.
    /// </summary>
    /// <param name="expectedShape">The expected shape of the tensor.</param>
    /// <param name="actualShape">The actual shape of the tensor.</param>
    /// <param name="component">The component where the shape mismatch occurred.</param>
    /// <param name="operation">The operation being performed when the mismatch was detected.</param>
    public TensorShapeMismatchException(int[] expectedShape, int[] actualShape, string component, string operation)
        : base(FormatMessage(expectedShape, actualShape, component, operation))
    {
        ExpectedShape = expectedShape;
        ActualShape = actualShape;
        Component = component;
        Operation = operation;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorShapeMismatchException"/> class with a simplified context.
    /// </summary>
    /// <param name="expectedShape">The expected shape of the tensor.</param>
    /// <param name="actualShape">The actual shape of the tensor.</param>
    /// <param name="context">Additional context about where the mismatch occurred.</param>
    public TensorShapeMismatchException(int[] expectedShape, int[] actualShape, string context)
        : this(expectedShape, actualShape, context, "operation")
    {
    }

    private static string FormatMessage(int[] expectedShape, int[] actualShape, string component, string operation)
    {
        return $"Shape mismatch in {component} during {operation}: " +
               $"Expected shape [{string.Join(", ", expectedShape)}], " +
               $"but got [{string.Join(", ", actualShape)}].";
    }
}