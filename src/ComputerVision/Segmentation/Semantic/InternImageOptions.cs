using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for the InternImage semantic segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> InternImage options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. InternImage is a large-scale CNN that uses Deformable
/// Convolution v3 (DCNv3) to compete with Vision Transformers on dense prediction tasks.
/// </para>
/// </remarks>
public class InternImageOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public InternImageOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public InternImageOptions(InternImageOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
