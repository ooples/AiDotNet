using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the Mask DINO model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mask DINO unifies detection and segmentation in one framework.
/// Options inherit from NeuralNetworkOptions and provide defaults suitable for most use cases.
/// </para>
/// </remarks>
public class MaskDINOOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MaskDINOOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MaskDINOOptions(MaskDINOOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
