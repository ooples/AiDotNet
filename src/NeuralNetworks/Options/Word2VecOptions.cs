using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the Word2Vec model.
/// </summary>
public class Word2VecOptions : EmbeddingModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Word2VecOptions"/> class carrying
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
    public Word2VecOptions()
    {
        VocabSize = 10000;
        EmbeddingDimension = 100;
        WindowSize = 5;
        MaxSequenceLength = 512;
        Type = Word2VecType.SkipGram;
        MaxGradNorm = 1.0;
    }


    /// <summary>
    /// Gets or sets the window size.
    /// </summary>
    public int WindowSize { get; set; }

    /// <summary>
    /// Gets or sets the type.
    /// </summary>
    public Word2VecType Type { get; set; }

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
