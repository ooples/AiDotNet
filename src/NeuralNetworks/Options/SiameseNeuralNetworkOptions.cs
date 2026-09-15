using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the SiameseNeuralNetwork.
/// </summary>
public class SiameseNeuralNetworkOptions : ModelHyperparameterOptions
{

    /// <summary>
    /// Gets or sets embedding dimension. Default: <c>768</c>.
    /// </summary>
    public int EmbeddingDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets max sequence length. Default: <c>512</c>.
    /// </summary>
    public int MaxSequenceLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets vocab size. Default: <c>30522</c>.
    /// </summary>
    public int VocabSize { get; set; } = 30522;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(EmbeddingDimension, nameof(EmbeddingDimension));
        Require(MaxSequenceLength, nameof(MaxSequenceLength));
        Require(VocabSize, nameof(VocabSize));
    }
}
