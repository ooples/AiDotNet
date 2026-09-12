using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the TransformerEmbeddingNetwork.
/// </summary>
public class TransformerEmbeddingOptions : EmbeddingModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TransformerEmbeddingOptions"/> class carrying
    /// this model's shipped defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They are the values this
    /// model has always used, moved here from its constructor so they can be seen and
    /// changed in one place.
    /// </para>
    /// <para>
    /// Carried over unchanged. Whether each matches the published paper is verified, and
    /// corrected where it does not, in a later phase of issue #2090.
    /// </para>
    /// </remarks>
    public TransformerEmbeddingOptions()
    {
        VocabSize = 30522;
        EmbeddingDimension = 768;
        MaxSequenceLength = 512;
        NumLayers = 12;
        NumHeads = 12;
        FeedForwardDim = 3072;
        EmbeddingPoolingStrategy = EmbeddingPoolingStrategy.Mean;
        MaxGradNorm = 1.0;
    }


    /// <summary>
    /// Gets or sets the pooling strategy.
    /// </summary>
    public EmbeddingPoolingStrategy EmbeddingPoolingStrategy { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
    }
}
