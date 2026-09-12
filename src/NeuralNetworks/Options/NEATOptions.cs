using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the NEAT neural network.
/// </summary>
public class NEATOptions : ModelHyperparameterOptions
{

    /// <summary>
    /// Gets or sets crossover rate. Default: <c>0.75</c>.
    /// </summary>
    public double CrossoverRate { get; set; } = 0.75;

    /// <summary>
    /// Gets or sets mutation rate. Default: <c>0.1</c>.
    /// </summary>
    public double MutationRate { get; set; } = 0.1;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        if (CrossoverRate < 0.0 || CrossoverRate > 1.0)
        {
            throw new ArgumentException(
                $"{GetType().Name}.CrossoverRate is {CrossoverRate}, but it must be between 0.0 and 1.0.",
                OptionsParameterName);
        }
        if (MutationRate < 0.0 || MutationRate > 1.0)
        {
            throw new ArgumentException(
                $"{GetType().Name}.MutationRate is {MutationRate}, but it must be between 0.0 and 1.0.",
                OptionsParameterName);
        }
    }
}
