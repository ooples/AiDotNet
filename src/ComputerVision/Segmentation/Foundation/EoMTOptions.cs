using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the EoMT (Encoder-only Mask Transformer) model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> EoMT removes the pixel and transformer decoders used by Mask2Former,
/// placing mask queries directly inside a plain ViT (DINOv2). This yields 4.4x faster inference.
/// Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class EoMTOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public EoMTOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EoMTOptions(EoMTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
