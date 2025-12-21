namespace AiDotNet.Exceptions;

/// <summary>
/// Exception thrown when input data contains invalid values such as NaN or infinity.
/// </summary>
public class InvalidDataValueException : AiDotNetException
{
    /// <summary>
    /// The component where the invalid data was detected.
    /// </summary>
    public string Component { get; }

    /// <summary>
    /// The operation being performed when the invalid data was detected.
    /// </summary>
    public string Operation { get; }

    /// <summary>
    /// Creates a new instance of the InvalidDataValueException class.
    /// </summary>
    public InvalidDataValueException() : base()
    {
        Component = "Unknown";
        Operation = "Unknown";
    }

    /// <summary>
    /// Creates a new instance of the InvalidDataValueException class with a specified error message.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public InvalidDataValueException(string message) : base(message)
    {
        Component = "Unknown";
        Operation = "Unknown";
    }

    /// <summary>
    /// Creates a new instance of the InvalidDataValueException class with a specified error message
    /// and a reference to the inner exception that is the cause of this exception.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public InvalidDataValueException(string message, Exception innerException)
        : base(message, innerException)
    {
        Component = "Unknown";
        Operation = "Unknown";
    }

    /// <summary>
    /// Creates a new instance of the InvalidDataValueException class with a specified error message
    /// and context information about where the exception occurred.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="component">The component where the invalid data was detected.</param>
    /// <param name="operation">The operation being performed when the invalid data was detected.</param>
    public InvalidDataValueException(string message, string component, string operation)
        : base(FormatMessage(message, component, operation))
    {
        Component = component;
        Operation = operation;
    }

    /// <summary>
    /// Creates a new instance of the InvalidDataValueException class with a specified error message,
    /// context information, and a reference to the inner exception.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="component">The component where the invalid data was detected.</param>
    /// <param name="operation">The operation being performed when the invalid data was detected.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public InvalidDataValueException(string message, string component, string operation, Exception innerException)
        : base(FormatMessage(message, component, operation), innerException)
    {
        Component = component;
        Operation = operation;
    }

    private static string FormatMessage(string message, string component, string operation)
    {
        return $"Invalid data value in {component} during {operation}: {message}";
    }
}
