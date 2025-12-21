namespace AiDotNet.Exceptions;

/// <summary>
/// Exception thrown when input data dimensions are invalid for a specific algorithm or operation.
/// </summary>
public class InvalidInputDimensionException : AiDotNetException
{
    /// <summary>
    /// The component where the dimension mismatch occurred.
    /// </summary>
    public string Component { get; }

    /// <summary>
    /// The operation being performed when the mismatch was detected.
    /// </summary>
    public string Operation { get; }

    /// <summary>
    /// Creates a new instance of the InvalidInputDimensionException class.
    /// </summary>
    public InvalidInputDimensionException() : base()
    {
        Component = "Unknown";
        Operation = "Unknown";
    }

    /// <summary>
    /// Creates a new instance of the InvalidInputDimensionException class with a specified error message.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public InvalidInputDimensionException(string message) : base(message)
    {
        Component = "Unknown";
        Operation = "Unknown";
    }

    /// <summary>
    /// Creates a new instance of the InvalidInputDimensionException class with a specified error message
    /// and a reference to the inner exception that is the cause of this exception.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public InvalidInputDimensionException(string message, Exception innerException)
        : base(message, innerException)
    {
        Component = "Unknown";
        Operation = "Unknown";
    }

    /// <summary>
    /// Creates a new instance of the InvalidInputDimensionException class with a specified error message
    /// and context information about where the exception occurred.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="component">The component where the dimension mismatch occurred.</param>
    /// <param name="operation">The operation being performed when the mismatch was detected.</param>
    public InvalidInputDimensionException(string message, string component, string operation)
        : base(FormatMessage(message, component, operation))
    {
        Component = component;
        Operation = operation;
    }

    /// <summary>
    /// Creates a new instance of the InvalidInputDimensionException class with a specified error message,
    /// context information, and a reference to the inner exception.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="component">The component where the dimension mismatch occurred.</param>
    /// <param name="operation">The operation being performed when the mismatch was detected.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public InvalidInputDimensionException(string message, string component, string operation, Exception innerException)
        : base(FormatMessage(message, component, operation), innerException)
    {
        Component = component;
        Operation = operation;
    }

    private static string FormatMessage(string message, string component, string operation)
    {
        return $"Dimension error in {component} during {operation}: {message}";
    }
}
