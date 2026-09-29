using AiDotNet.Models.Options;

using AiDotNet.Video.Segmentation;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the XMem video segmentation model.
/// </summary>
public class XMemOptions : VideoHyperparameterOptions
{
    /// <summary>Initializes a new instance with the paper's training defaults.</summary>
    public XMemOptions()
    {
        NumFeatures = 256;
        SensoryMemorySize = 3;
        WorkingMemorySize = 10;
        LongTermMemorySize = 100;
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
        SensoryMemorySize = other.SensoryMemorySize;
        WorkingMemorySize = other.WorkingMemorySize;
        LongTermMemorySize = other.LongTermMemorySize;
    }

    /// <summary>
    /// Gets or sets the AdamW learning rate. The XMem paper uses 1e-5.
    /// </summary>
    public double LearningRate { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets AdamW's decoupled weight decay. The XMem paper uses 0.05.
    /// </summary>
    public double WeightDecay { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the sensory memory size.
    /// </summary>
    public int SensoryMemorySize { get; set; }

    /// <summary>
    /// Gets or sets the working memory size.
    /// </summary>
    public int WorkingMemorySize { get; set; }

    /// <summary>
    /// Gets or sets the long term memory size.
    /// </summary>
    public int LongTermMemorySize { get; set; }

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
