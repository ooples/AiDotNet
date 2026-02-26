using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the U2Seg model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> U2Seg is a unified unsupervised segmentation framework that performs
/// instance, semantic, and panoptic segmentation without requiring any human annotations.
/// Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class U2SegOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public U2SegOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public U2SegOptions(U2SegOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
