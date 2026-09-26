using AiDotNet.Models.Options;

using AiDotNet.Video.FrameInterpolation;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the RIFE frame interpolation model.
/// </summary>
public class RIFEOptions : VideoHyperparameterOptions
{
    /// <summary>Initializes the ECCV 2022 training defaults.</summary>
    public RIFEOptions()
    {
        NumFeatures = 64; // DefaultNumFeatures
        NumFlowBlocks = 3; // DefaultNumFlowBlocks
    }

    /// <summary>Initializes an independent copy of another RIFE configuration.</summary>
    /// <param name="other">The configuration to copy.</param>
    public RIFEOptions(RIFEOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
            NumFlowBlocks = other.NumFlowBlocks;
    }

    /// <summary>
    /// Gets or sets AdamW's initial learning rate. Huang et al. train RIFE
    /// from 1e-4 to 1e-5 with cosine annealing.
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets AdamW's decoupled weight decay. The ECCV paper uses 1e-4.
    /// </summary>
    public double WeightDecay { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the num flow blocks.
    /// </summary>
    public int NumFlowBlocks { get; set; }

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
