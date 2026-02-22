using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Mamba;

/// <summary>
/// Configuration options for VMamba visual state space model segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VMamba model. Default values follow the original paper settings.</para>
/// </remarks>
public class VMambaOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public VMambaOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VMambaOptions(VMambaOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
