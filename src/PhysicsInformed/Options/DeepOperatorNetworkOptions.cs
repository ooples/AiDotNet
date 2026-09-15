using AiDotNet.Models.Options;

namespace AiDotNet.PhysicsInformed.Options;

/// <summary>
/// Configuration options for the DeepOperatorNetwork.
/// </summary>
public class DeepOperatorNetworkOptions : PhysicsInformedOptions
{

    /// <summary>
    /// Gets or sets latent dimension. Default: <c>128</c>.
    /// </summary>
    public int LatentDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets num sensors. Default: <c>100</c>.
    /// </summary>
    public int NumSensors { get; set; } = 100;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(LatentDimension, nameof(LatentDimension));
        Require(NumSensors, nameof(NumSensors));
    }
}
