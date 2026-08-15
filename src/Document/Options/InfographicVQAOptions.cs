using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the InfographicVQA document model.
/// </summary>
public class InfographicVQAOptions : DocumentNeuralNetworkOptions
{
    /// <summary>Initializes an options instance with default values.</summary>
    public InfographicVQAOptions() { }

    /// <summary>Initializes an options instance by copying inherited configuration.</summary>
    /// <param name="other">The source options.</param>
    public InfographicVQAOptions(InfographicVQAOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }
}
