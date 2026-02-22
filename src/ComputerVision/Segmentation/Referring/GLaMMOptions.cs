using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Referring;

/// <summary>
/// Configuration options for GLaMM grounding language model segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GLaMM model. Default values follow the original paper settings.</para>
/// </remarks>
public class GLaMMOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public GLaMMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GLaMMOptions(GLaMMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
