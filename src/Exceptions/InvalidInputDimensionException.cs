namespace AiDotNet.Exceptions;

/// <summary>
/// Exception thrown when input data dimensions are invalid for a specific algorithm or operation.
/// </summary>
public class InvalidInputDimensionException : Exception
{
    /// <summary>
    /// Creates a new instance of the InvalidInputDimensionException class.
    /// </summary>
    public InvalidInputDimensionException() : base() { }

    /// <summary>
    /// Creates a new instance of the InvalidInputDimensionException class with a specified error message.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public InvalidInputDimensionException(string message) : base(message) { }

    /// <summary>
    /// Creates a new instance of the InvalidInputDimensionException class with a specified error message
    /// and a reference to the inner exception that is the cause of this exception.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public InvalidInputDimensionException(string message, Exception innerException) 
        : base(message, innerException) { }
}