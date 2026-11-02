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
    /// Gets or sets how many residual blocks the default stack has. Default: <c>3</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Used only when the architecture supplies no layers of its own. Each block is two layers with an
    /// identity shortcut around them (He et al. 2016).
    /// </para>
    /// <para><b>For Beginners:</b> More blocks make the network deeper; the shortcuts keep a deep stack
    /// trainable.</para>
    /// </remarks>
    public int NumResidualBlocks { get; set; } = 3;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(BatchSize, nameof(BatchSize));
        Require(Epochs, nameof(Epochs));
    }
}
