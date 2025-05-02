namespace AiDotNet.Exceptions;

/// <summary>
/// Custom exception for errors that occur during prediction statistics calculation.
/// </summary>
/// <remarks>
/// <para>
/// This exception is thrown when errors occur during the calculation of prediction statistics.
/// It provides additional context about what went wrong during the calculation process.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is a specialized error type specifically for problems that occur
/// when calculating prediction statistics. Using custom exceptions like this helps you
/// handle specific error types differently in your code.
/// </para>
/// </remarks>
[Serializable]
public class PredictionStatsException : AiDotNetException
{
    /// <summary>
    /// Initializes a new instance of the PredictionStatsException class.
    /// </summary>
    public PredictionStatsException() { }

    /// <summary>
    /// Initializes a new instance of the PredictionStatsException class with a specified error message.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    public PredictionStatsException(string message) : base(message) { }

    /// <summary>
    /// Initializes a new instance of the PredictionStatsException class with a specified error message
    /// and a reference to the inner exception that is the cause of this exception.
    /// </summary>
    /// <param name="message">The message that describes the error.</param>
    /// <param name="innerException">The exception that is the cause of the current exception.</param>
    public PredictionStatsException(string message, Exception innerException) : base(message, innerException) { }
}