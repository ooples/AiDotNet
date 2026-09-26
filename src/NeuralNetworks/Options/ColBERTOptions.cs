using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the ColBERT model.
/// </summary>
public class ColBERTOptions : TransformerEmbeddingOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ColBERTOptions"/> class carrying
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
    public ColBERTOptions()
    {
        VocabSize = 30522;
        OutputDimension = 128;
        MaxSequenceLength = 512;
        NumLayers = 12;
        NumHeads = 12;
        FeedForwardDim = 3072;
        MaxGradNorm = 1.0;
    }


    /// <summary>
    /// Gets or sets the output dimension.
    /// </summary>
    public int OutputDimension { get; set; }
}
