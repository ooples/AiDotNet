using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Referring;

/// <summary>
/// Configuration options for PixelLM pixel-level reasoning segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the PixelLM model. Default values follow the original paper settings.</para>
/// </remarks>
public class PixelLMOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public PixelLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PixelLMOptions(PixelLMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
