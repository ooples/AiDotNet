using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for the SegNeXt semantic segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SegNeXt options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. SegNeXt uses a purely convolutional architecture
/// with multi-scale attention — no transformers needed — making it one of the most efficient
/// semantic segmentation models available.
/// </para>
/// </remarks>
public class SegNeXtOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SegNeXtOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SegNeXtOptions(SegNeXtOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
