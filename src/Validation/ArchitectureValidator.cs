namespace AiDotNet.Validation;

/// <summary>
/// Provides validation methods for neural network architectures.
/// </summary>
public static class ArchitectureValidator
{
    /// <summary>
    /// Validates that the architecture has the expected input type.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="architecture">The neural network architecture to validate.</param>
    /// <param name="expectedInputType">The expected input type.</param>
    /// <param name="networkType">The type of neural network being validated.</param>
    /// <exception cref="InvalidInputTypeException">Thrown when the input type doesn't match the expected type.</exception>
    public static void ValidateInputType<T>(NeuralNetworkArchitecture<T> architecture, InputType expectedInputType, string networkType)
    {
        if (architecture.InputType != expectedInputType)
        {
            throw new InvalidInputTypeException(expectedInputType, architecture.InputType, networkType);
        }
    }
}
