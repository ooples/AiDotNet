using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the CapsuleNetwork.
/// </summary>
public class CapsuleNetworkOptions : NeuralNetworkOptions
{
    private double _learningRate = 0.01;

    /// <summary>
    /// Learning rate for SGD parameter updates during training. Default: 0.01.
    /// Must be finite and greater than zero.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if the value is NaN, infinite, or not greater than 0.</exception>
    public double LearningRate
    {
        get => _learningRate;
        set
        {
            if (double.IsNaN(value) || double.IsInfinity(value) || value <= 0d)
                throw new ArgumentOutOfRangeException(nameof(LearningRate),
                    "LearningRate must be finite and greater than 0.");
            _learningRate = value;
        }
    }
}
