using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the FastText model.
/// </summary>
public class FastTextOptions : EmbeddingModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FastTextOptions"/> class carrying
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
    public FastTextOptions()
    {
        VocabSize = 10000;
        BucketSize = 2000000;
        MinNgramLength = 3;
        MaxNgramLength = 6;
        EmbeddingDimension = 100;
        MaxSequenceLength = 512;
        MaxGradNorm = 1.0;
    }


    /// <summary>
    /// Gets or sets the bucket size.
    /// </summary>
    public int BucketSize { get; set; }

    /// <summary>
    /// Gets or sets the shortest character n-gram a word is split into.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Bojanowski et al. 2017 (section 3.2) extract every n-gram with 3 &lt;= n &lt;= 6 from the word
    /// wrapped in the boundary symbols &lt; and &gt;. The default is that paper's 3.
    /// </para>
    /// <para><b>For Beginners:</b> FastText understands a word partly through its pieces, so "where"
    /// also contributes "&lt;wh", "whe", "her" and so on. This is the length of the shortest piece.</para>
    /// </remarks>
    public int MinNgramLength { get; set; }

    /// <summary>
    /// Gets or sets the longest character n-gram a word is split into; 0 turns subwords off.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The default is the paper's 6. As in the reference implementation (its <c>maxn</c> setting),
    /// 0 disables subwords, leaving a plain word-embedding model.
    /// </para>
    /// </remarks>
    public int MaxNgramLength { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
        if (MaxNgramLength != 0 && (MinNgramLength < 1 || MaxNgramLength < MinNgramLength))
        {
            throw new ArgumentException(
                $"FastText n-gram lengths must satisfy 1 <= MinNgramLength <= MaxNgramLength, or MaxNgramLength = 0 to " +
                $"turn subwords off; got MinNgramLength={MinNgramLength}, MaxNgramLength={MaxNgramLength}.");
        }
    }
}
