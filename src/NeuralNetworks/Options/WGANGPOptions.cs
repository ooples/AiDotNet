using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the WGANGP neural network.
/// </summary>
public class WGANGPOptions : NeuralNetworkOptions
{
    /// <summary>Initializes the paper defaults.</summary>
    public WGANGPOptions()
    {
    }

    /// <summary>Initializes an independent copy of another configuration.</summary>
    public WGANGPOptions(WGANGPOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
    }

    /// <summary>Gets or sets Adam's learning rate (Algorithm 1 default: 1e-4).</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets Adam's first-moment decay (Algorithm 1 default: 0).</summary>
    public double Beta1 { get; set; } = 0.0;

    /// <summary>Gets or sets Adam's second-moment decay (Algorithm 1 default: 0.9).</summary>
    public double Beta2 { get; set; } = 0.9;
}
