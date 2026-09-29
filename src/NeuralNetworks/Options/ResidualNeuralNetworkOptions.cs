using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the ResidualNeuralNetwork.
/// </summary>
public class ResidualNeuralNetworkOptions : ModelHyperparameterOptions
{

    /// <summary>
    /// Gets or sets batch size. Default: <c>32</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many examples the model looks at before it updates itself once.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets epochs. Default: <c>10</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many complete passes the model makes over your training data.</para>
    /// </remarks>
    public int Epochs { get; set; } = 10;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(BatchSize, nameof(BatchSize));
        Require(Epochs, nameof(Epochs));
    }
}
