using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the GLALanguageModel.
/// </summary>
public class GLAOptions : NeuralNetworkOptions
{
    public GLAOptions() { }

    public GLAOptions(GLAOptions other)
    {
        ArgumentNullException.ThrowIfNull(other);
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }
}
