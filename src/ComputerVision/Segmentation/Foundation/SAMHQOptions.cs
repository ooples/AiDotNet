using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the SAM-HQ (High-Quality Segment Anything) model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM-HQ extends SAM with a High-Quality output token for significantly
/// better mask boundaries. Options inherit from NeuralNetworkOptions and provide defaults
/// suitable for most use cases.
/// </para>
/// </remarks>
public class SAMHQOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SAMHQOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SAMHQOptions(SAMHQOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
