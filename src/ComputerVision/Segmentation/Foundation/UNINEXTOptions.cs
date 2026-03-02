using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the UNINEXT model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> UNINEXT reformulates 10+ instance perception tasks as object discovery
/// and retrieval. It achieves SOTA on 20+ benchmarks. Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class UNINEXTOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public UNINEXTOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UNINEXTOptions(UNINEXTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
