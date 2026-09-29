using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the SimCSE model.
/// </summary>
public class SimCSEOptions : TransformerEmbeddingOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="SimCSEOptions"/> class carrying
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
    public SimCSEOptions()
    {
        Type = SimCSEType.Unsupervised;
        VocabSize = 30522;
        EmbeddingDimension = 768;
        MaxSequenceLength = 512;
        NumLayers = 12;
        NumHeads = 12;
        FeedForwardDim = 3072;
        DropoutRate = 0.1;
        EmbeddingPoolingStrategy = EmbeddingPoolingStrategy.ClsToken;
        MaxGradNorm = 1.0;
    }


    /// <summary>
    /// Gets or sets the type.
    /// </summary>
    public SimCSEType Type { get; set; }

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; }
}
