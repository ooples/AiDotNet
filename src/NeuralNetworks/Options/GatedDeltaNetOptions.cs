using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the GatedDeltaNetLanguageModel.
/// </summary>
public class GatedDeltaNetOptions : NeuralNetworkOptions
{
    public GatedDeltaNetOptions() { }

    public GatedDeltaNetOptions(GatedDeltaNetOptions other)
    {
        ArgumentNullException.ThrowIfNull(other);
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }
}
