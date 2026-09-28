using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the Matryoshka Embedding model.
/// </summary>
public class MatryoshkaEmbeddingOptions : TransformerEmbeddingOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MatryoshkaEmbeddingOptions"/> class carrying
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
    public MatryoshkaEmbeddingOptions()
    {
        VocabSize = 30522;
        // 1536 was assigned to MaxEmbeddingDimension here, which nothing read, while the property
        // that actually sizes the model -- EmbeddingDimension, inherited from
        // TransformerEmbeddingOptions -- was left at its 768 default. So a model whose own
        // documentation and whose ModelFamily test both describe a 1536-wide output was built 768
        // wide, and EmbedResized rejected every rung of the nesting ladder above 768.
        EmbeddingDimension = 1536;
        NestedDimensions = [64, 128, 256, 512, 768, 1024, 1536];
        MaxSequenceLength = 512;
        NumLayers = 12;
        NumHeads = 12;
        FeedForwardDim = 3072;
        EmbeddingPoolingStrategy = EmbeddingPoolingStrategy.ClsToken;
        MaxGradNorm = 1.0;
    }


    // MaxEmbeddingDimension was declared here, set to 1536, and never read. It duplicated the
    // inherited EmbeddingDimension, which is the actual width of the embedding this model
    // produces and which TransformerEmbeddingOptions sets to 768 -- so the two disagreed, and a
    // caller adjusting the one that did nothing would have seen no effect. The ceiling on the
    // nesting ladder is EmbeddingDimension, and Validate below enforces it against that.

    /// <summary>
    /// Gets or sets the nesting ladder — the prefix widths a single embedding can be
    /// truncated to (Kusupati et al. 2022).
    /// </summary>
    /// <value>Defaults to 64, 128, 256, 512, 768, 1024, 1536.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Matryoshka training makes one embedding usable at several
    /// sizes: the first 64 numbers are a good 64-dimensional embedding on their own, the
    /// first 128 a better one, and so on. This lists the sizes that are trained to work.
    /// </para>
    /// </remarks>
    public int[] NestedDimensions { get; set; } = [];

    /// <summary>
    /// Throws if the nesting ladder cannot be honoured by an embedding of
    /// <see cref="EmbeddingModelOptions.EmbeddingDimension"/>.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// When the ladder is empty, contains a non-positive width, or names a width wider
    /// than <see cref="EmbeddingModelOptions.EmbeddingDimension"/>.
    /// </exception>
    public override void Validate()
    {
        base.Validate();

        if (NestedDimensions is null || NestedDimensions.Length == 0)
        {
            throw new ArgumentException(
                "NestedDimensions must name at least one nesting width.",
                nameof(NestedDimensions));
        }

        foreach (int dimension in NestedDimensions)
        {
            if (dimension <= 0)
            {
                throw new ArgumentException(
                    $"NestedDimensions must be positive; got {dimension}.",
                    nameof(NestedDimensions));
            }

            if (dimension > EmbeddingDimension)
            {
                throw new ArgumentException(
                    $"NestedDimensions contains {dimension}, which exceeds EmbeddingDimension "
                    + $"{EmbeddingDimension}. A prefix cannot be wider than the embedding it is "
                    + "taken from, and EmbedResized rejects any dimension above it.",
                    nameof(NestedDimensions));
            }
        }
    }
}
