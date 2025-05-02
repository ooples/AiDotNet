namespace AiDotNet.Exceptions;

/// <summary>
/// Custom exception for errors that occur during error statistics calculation.
/// </summary>
[Serializable]
public class ErrorStatsException : AiDotNetException
{
    /// <summary>
    /// Initializes a new instance of the ErrorStatsException class.
    /// </summary>
    public ErrorStatsException() { }

    /// <summary>
    /// Initializes a new instance of the ErrorStatsException class with a specified error message.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public ErrorStatsException(string message) : base(message) { }

    /// <summary>
    /// Initializes a new instance of the ErrorStatsException class with a specified error message
    /// and a reference to the inner exception that is the cause of this exception.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public ErrorStatsException(string message, Exception innerException) : base(message, innerException) { }
}