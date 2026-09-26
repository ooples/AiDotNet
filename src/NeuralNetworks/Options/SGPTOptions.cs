using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the SGPT model.
/// </summary>
public class SGPTOptions : TransformerEmbeddingOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="SGPTOptions"/> class carrying
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
    public SGPTOptions()
    {
        VocabSize = 50257;
        MaxSequenceLength = 1024;
        NumLayers = 12;
        FeedForwardDim = 3072;
        EmbeddingPoolingStrategy = EmbeddingPoolingStrategy.Mean;
        MaxGradNorm = 1.0;
    }
}
