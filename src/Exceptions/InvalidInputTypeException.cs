namespace AiDotNet.Exceptions;

/// <summary>
/// Exception thrown when a neural network receives an input type that doesn't match its requirements.
/// </summary>
public class InvalidInputTypeException : AiDotNetException
{
    /// <summary>
    /// The expected input type for the neural network.
    /// </summary>
    public InputType ExpectedInputType { get; }

    /// <summary>
    /// The actual input type provided.
    /// </summary>
    public InputType ActualInputType { get; }

    /// <summary>
    /// The type of neural network that requires the specific input type.
    /// </summary>
    public string NetworkType { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="InvalidInputTypeException"/> class.
    /// </summary>
    /// <param name="expectedInputType">The expected input type for the neural network.</param>
    /// <param name="actualInputType">The actual input type provided.</param>
    /// <param name="networkType">The type of neural network that requires the specific input type.</param>
    public InvalidInputTypeException(InputType expectedInputType, InputType actualInputType, string networkType)
        : base(FormatMessage(expectedInputType, actualInputType, networkType))
    {
        ExpectedInputType = expectedInputType;
        ActualInputType = actualInputType;
        NetworkType = networkType;
    }

    private static string FormatMessage(InputType expectedInputType, InputType actualInputType, string networkType)
    {
        return $"{networkType} requires {expectedInputType} input, but received {actualInputType} input.";
    }
}
