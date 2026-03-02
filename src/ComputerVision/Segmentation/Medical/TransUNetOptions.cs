using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// Configuration options for TransUNet medical segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the TransUNet model. Default values follow the original paper settings.</para>
/// </remarks>
public class TransUNetOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public TransUNetOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TransUNetOptions(TransUNetOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
