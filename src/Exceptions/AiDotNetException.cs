namespace AiDotNet.Exceptions;

/// <summary>
/// Base exception for all AiDotNet-specific exceptions.
/// </summary>
public class AiDotNetException : Exception
{
    /// <summary>
    /// Initializes a new instance of the <see cref="AiDotNetException"/> class.
    /// </summary>
    public AiDotNetException() : base() { }

    /// <summary>
    /// Initializes a new instance of the <see cref="AiDotNetException"/> class with a specified error message.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public AiDotNetException(string message) : base(message) { }

    /// <summary>
    /// Initializes a new instance of the <see cref="AiDotNetException"/> class with a specified error message
    /// and a reference to the inner exception that is the cause of this exception.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public AiDotNetException(string message, Exception innerException) : base(message, innerException) { }
}
