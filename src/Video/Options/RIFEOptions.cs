using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the RIFE frame interpolation model.
/// </summary>
public class RIFEOptions : NeuralNetworkOptions
{
    /// <summary>Initializes the ECCV 2022 training defaults.</summary>
    public RIFEOptions()
    {
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
}
