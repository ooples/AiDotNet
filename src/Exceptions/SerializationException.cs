namespace AiDotNet.Exceptions;

/// <summary>
/// Exception thrown when there's an error during serialization or deserialization of a neural network.
/// </summary>
public class SerializationException : AiDotNetException
{
    /// <summary>
    /// The component where the serialization error occurred.
    /// </summary>
    public string Component { get; }

    /// <summary>
    /// The operation being performed when the error was detected (e.g., "Serialize", "Deserialize").
    /// </summary>
    public string Operation { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="SerializationException"/> class.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="component">The component where the serialization error occurred.</param>
    /// <param name="operation">The operation being performed when the error was detected.</param>
    public SerializationException(string message, string component, string operation)
        : base(FormatMessage(message, component, operation))
    {
        Component = component;
        Operation = operation;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SerializationException"/> class with an inner exception.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="component">The component where the serialization error occurred.</param>
    /// <param name="operation">The operation being performed when the error was detected.</param>
    /// <param name="innerException">The inner exception that caused this exception.</param>
    public SerializationException(string message, string component, string operation, Exception innerException)
        : base(FormatMessage(message, component, operation), innerException)
    {
        Component = component;
        Operation = operation;
    }

    private static string FormatMessage(string message, string component, string operation)
    {
        return $"Serialization error in {component} during {operation}: {message}";
    }
}