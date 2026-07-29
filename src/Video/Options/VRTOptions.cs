using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the VRT video restoration model.
/// </summary>
public class VRTOptions : NeuralNetworkOptions
{
    /// <summary>Initializes the paper/released-training defaults.</summary>
    public VRTOptions()
    {
    }

    /// <summary>Initializes an independent copy of another VRT configuration.</summary>
    public VRTOptions(VRTOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        CharbonnierEpsilon = other.CharbonnierEpsilon;
    }

    /// <summary>
    /// Gets or sets Adam's initial learning rate. The released VRT training
    /// configuration uses 4e-4.
    /// </summary>
    public double LearningRate { get; set; } = 4e-4;

    /// <summary>
    /// Gets or sets the Charbonnier smoothing constant. The paper defines
    /// the restoration objective with epsilon equal to 1e-3.
    /// </summary>
    public double CharbonnierEpsilon { get; set; } = 1e-3;
}
