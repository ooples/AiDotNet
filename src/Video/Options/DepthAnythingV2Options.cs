using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the DepthAnythingV2 video model.
/// </summary>
public class DepthAnythingV2Options : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with research-paper defaults selected by model size.</summary>
    public DepthAnythingV2Options()
    {
    }

    /// <summary>Initializes a new instance by copying another options instance.</summary>
    public DepthAnythingV2Options(DepthAnythingV2Options other)
    {
        ArgumentNullException.ThrowIfNull(other);
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        NumFeatures = other.NumFeatures;
        NumEncoderBlocks = other.NumEncoderBlocks;
    }

    /// <summary>
    /// Gets or sets the ViT feature dimension. A null value uses the selected
    /// Depth Anything V2 model-size default.
    /// </summary>
    public int? NumFeatures { get; set; }

    /// <summary>
    /// Gets or sets the number of DINOv2 transformer encoder blocks. A null value
    /// uses the selected model-size default.
    /// </summary>
    public int? NumEncoderBlocks { get; set; }
}
