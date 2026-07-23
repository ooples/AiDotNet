using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the Mask2Former universal segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mask2Former options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. Mask2Former is a universal model that can perform
/// semantic, instance, and panoptic segmentation with a single architecture by using
/// masked cross-attention in its transformer decoder.
/// </para>
/// </remarks>
public class Mask2FormerOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public Mask2FormerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Mask2FormerOptions(Mask2FormerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
