using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the SiameseNetwork.
/// </summary>
public class SiameseNetworkOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Size of the shared-tower embedding compared by the Siamese similarity head.
    /// </summary>
    public int EmbeddingSize { get; set; } = 64;
}
