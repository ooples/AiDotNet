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
        MaxQueryLength = 32;
        NumLayers = 12;
        NumHeads = 12;
        FeedForwardDim = 3072;
        MaxGradNorm = 1.0;
    }


    /// <summary>
    /// Gets or sets the output dimension.
    /// </summary>
    public int OutputDimension { get; set; }

    /// <summary>
    /// Gets or sets the fixed number of tokens a query is encoded as (N<sub>q</sub>).
    /// </summary>
    /// <remarks>
    /// <para>
    /// ColBERT's query encoder (Khattab &amp; Zaharia 2020, section 3.2) truncates a query to N<sub>q</sub>
    /// tokens and pads a shorter one with [MASK] tokens up to N<sub>q</sub>. The paper calls this query
    /// augmentation: the encoder learns to fill the masks with terms that expand the query. The default
    /// is the paper's 32.
    /// </para>
    /// <para><b>For Beginners:</b> Every search query becomes exactly this many token vectors. Short
    /// queries get placeholder tokens that the model learns to use for related words.</para>
    /// </remarks>
    public int MaxQueryLength { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when a required size is zero or negative.</exception>
    public override void Validate()
    {
        base.Validate();
        if (MaxQueryLength <= 0 || MaxQueryLength > MaxSequenceLength)
        {
            throw new ArgumentException(
                $"MaxQueryLength must be in [1, MaxSequenceLength={MaxSequenceLength}]; got {MaxQueryLength}.");
        }
    }
}
