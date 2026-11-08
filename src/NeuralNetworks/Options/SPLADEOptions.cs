using AiDotNet.Attributes;
using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the SPLADE model.
/// </summary>
[PaperDefault(nameof(VocabSize), 30522, Source = "Formal et al. 2021, Sec. 3.1: the BERT WordPiece vocabulary, |V| = 30522")]
[PaperDefault(nameof(MaxSequenceLength), 256, Source = "Formal et al. 2021, Sec. 4: a maximum length of 256 for input sequences")]
public class SPLADEOptions : TransformerEmbeddingOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="SPLADEOptions"/> class carrying
    /// this model's shipped defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They were moved here from
    /// the model's constructor so they can be seen and changed in one place.
    /// </para>
    /// <para>
    /// <c>MaxSequenceLength</c> is the paper's 256 (Sec. 4); it was 512, so
    /// inputs longer than 256 tokens are now truncated unless it is set higher. The vocabulary is the paper's
    /// too. The model sizes are carried over unchanged; checking them against the paper is issue #2090's.
    /// </para>
    /// </remarks>
    public SPLADEOptions()
    {
        VocabSize = 30522;
        EmbeddingDimension = 768;
        MaxSequenceLength = 256;
        NumLayers = 12;
        NumHeads = 12;
        FeedForwardDim = 3072;
        MaxGradNorm = 1.0;
    }
}
