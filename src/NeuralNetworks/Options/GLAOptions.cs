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
        if (other is null)
            throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }
}
