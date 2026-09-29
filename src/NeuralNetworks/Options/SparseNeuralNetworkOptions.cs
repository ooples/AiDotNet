using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the SparseNeuralNetwork.
/// </summary>
public class SparseNeuralNetworkOptions : ModelHyperparameterOptions
{

    /// <summary>
    /// Gets or sets sparsity. Default: <c>0.9</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> What fraction of the connections are switched off.</para>
    /// </remarks>
    public double Sparsity { get; set; } = 0.9;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        if (Sparsity < 0.0 || Sparsity > 1.0)
        {
            throw new ArgumentException(
                $"{GetType().Name}.Sparsity is {Sparsity}, but it must be between 0.0 and 1.0.",
                OptionsParameterName);
        }
    }
}
