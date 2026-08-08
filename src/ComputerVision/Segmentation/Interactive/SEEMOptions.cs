using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Interactive;

/// <summary>
/// Configuration options for SEEM interactive segmentation.
/// </summary>
/// <remarks>
/// <para>
/// By default, the model-size selector uses the released Focal-T configuration (96-channel stem,
/// stage depths 2/2/6/2, and a 512-wide decoder). Set the optional topology properties to load a
/// custom or reduced checkpoint without changing the production defaults.
/// </para>
/// <para><b>For Beginners:</b> These options configure SEEM and allow every native stage size to be customized.</para>
/// </remarks>
public class SEEMOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SEEMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SEEMOptions(SEEMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        ChannelDimensions = other.ChannelDimensions?.ToArray();
        StageDepths = other.StageDepths?.ToArray();
        DecoderDimension = other.DecoderDimension;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>Gets or sets custom four-stage Focal channel widths, or null to use the selected released variant.</summary>
    public int[]? ChannelDimensions { get; set; }

    /// <summary>Gets or sets custom four-stage Focal block depths, or null to use the selected released variant.</summary>
    public int[]? StageDepths { get; set; }

    /// <summary>Gets or sets a custom mask-decoder width, or null to use the released 512-wide decoder.</summary>
    public int? DecoderDimension { get; set; }

    /// <summary>Gets or sets the AdamW learning rate used by native training.</summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the AdamW weight decay used by native training.</summary>
    public double WeightDecay { get; set; } = 0.01;
}
