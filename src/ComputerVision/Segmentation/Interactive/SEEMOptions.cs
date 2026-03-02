using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Interactive;

/// <summary>
/// Configuration options for SEEM interactive segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SEEM model. Default values follow the original paper settings.</para>
/// </remarks>
public class SEEMOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SEEMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SEEMOptions(SEEMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
