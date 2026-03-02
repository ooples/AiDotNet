using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Mamba;

/// <summary>
/// Configuration options for ViM-UNet Vision Mamba + U-Net biomedical segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ViMUNet model. Default values follow the original paper settings.</para>
/// </remarks>
public class ViMUNetOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ViMUNetOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ViMUNetOptions(ViMUNetOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
