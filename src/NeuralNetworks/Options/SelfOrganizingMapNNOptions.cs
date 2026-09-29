using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the SelfOrganizingMap neural network.
/// </summary>
public class SelfOrganizingMapNNOptions : ModelHyperparameterOptions
{

    /// <summary>
    /// Gets or sets total epochs. Default: <c>1000</c>.
    /// </summary>
    public int TotalEpochs { get; set; } = 1000;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(TotalEpochs, nameof(TotalEpochs));
    }
}
