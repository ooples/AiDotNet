using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the XMem video segmentation model.
/// </summary>
public class XMemOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with the paper's training defaults.</summary>
    public XMemOptions()
    {
    }

    /// <summary>Initializes a new instance by copying another XMem options instance.</summary>
    /// <param name="other">The options to copy.</param>
    public XMemOptions(XMemOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>
    /// Gets or sets the AdamW learning rate. The XMem paper uses 1e-5.
    /// </summary>
    public double LearningRate { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets AdamW's decoupled weight decay. The XMem paper uses 0.05.
    /// </summary>
    public double WeightDecay { get; set; } = 0.05;
}
