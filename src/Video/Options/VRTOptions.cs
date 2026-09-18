using AiDotNet.Models.Options;

using AiDotNet.Video.Restoration;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the VRT video restoration model.
/// </summary>
public class VRTOptions : VideoHyperparameterOptions
{
    /// <summary>Initializes the paper/released-training defaults.</summary>
    public VRTOptions()
    {
        EmbedDim = 120;
        NumFrames = 6;
        NumBlocks = 8;
        ScaleFactor = 4;
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
        NumBlocks = other.NumBlocks;
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

    /// <summary>
    /// Gets or sets the num blocks.
    /// </summary>
    public int NumBlocks { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
    }
}
