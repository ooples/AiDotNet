namespace AiDotNet.Exceptions;

/// <summary>
/// Exception thrown when model training operations fail.
/// </summary>
public class ModelTrainingException : AiDotNetException
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ModelTrainingException"/> class.
    /// </summary>
    public ModelTrainingException() : base() { }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelTrainingException"/> class with a specified error message.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public ModelTrainingException(string message) : base(message) { }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelTrainingException"/> class with a specified error message
    /// and a reference to the inner exception that is the cause of this exception.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public ModelTrainingException(string message, Exception innerException) : base(message, innerException) { }
}
